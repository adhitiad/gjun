import asyncio
import json
import logging

from langchain_groq import ChatGroq

from brain import InternalSignalBus
from config import settings

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("LLM-Strategist")


class LLMStrategist:
    def __init__(self, signal_bus: InternalSignalBus):
        # Menerima signal_bus (InternalSignalBus) sebagai pengganti Redis
        self.bus = signal_bus

        # Inisialisasi LLM (Groq)
        try:
            self.llm = ChatGroq(
                api_key=settings.GROQ_API_KEY, model="llama3-70b-8192", temperature=0.1
            )
            self.llm_active = True
        except Exception as e:
            logger.error(f"❌ Groq Init Failed: {e}")
            self.llm_active = False

    async def validate_signal(self, data):
        """Chain of Thought Validation menggunakan Llama-3"""
        if not self.llm_active:
            return {"decision": "APPROVED", "reason": "LLM Offline (Bypass)"}

        try:
            # Hitung Risk:Reward manual untuk konteks LLM
            entry = float(data.get("entry", 0))
            sl = float(data.get("sl", 0))
            tp = float(data.get("tp", 0))
            atr = float(data.get("atr", 0))

            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr_ratio = reward / risk if risk > 0 else 0

            prompt = f"""
            You are a Senior Forex Risk Manager. Validate this trade signal.
            
            TRADE:
            - Pair: {data.get('symbol')}
            - Action: {data.get('action')}
            - Entry: {entry:.5f}
            - ATR: {atr:.5f}
            - Risk/Reward: 1:{rr_ratio:.2f}

            TASK:
            1. Is the Risk:Reward ratio healthy (>= 1:1.5)?
            2. Is the trade logical given the volatility?
            
            OUTPUT JSON ONLY:
            {{
                "decision": "APPROVED" or "REJECTED",
                "reason": "Short explanation (max 10 words)"
            }}
            """

            # Panggil LLM
            response = await self.llm.ainvoke(prompt)

            # Bersihkan output (kadang LLM menyertakan markdown ```json)
            content = response.content.replace("```json", "").replace("```", "").strip()

            # Parsing JSON
            analysis = json.loads(content)
            return analysis

        except Exception as e:
            logger.error(f"LLM Validation Error: {e}")
            # Jika error, default ke APPROVED agar trading tidak macet, atau REJECTED jika ingin aman
            return {"decision": "APPROVED", "reason": "LLM Error (Auto-Approve)"}

    async def run(self):
        logger.info("🧠 LLM Strategist Started (Standalone Mode).")

        while True:
            # 1. AMBIL DATA DARI BUS (QUEUE)
            # Di sini kita menunggu sampai ada data masuk dari Brain
            raw_data = await self.bus.get()

            # Data dari InternalSignalBus sudah berupa Dictionary,
            # jadi TIDAK PERLU cek msg.type atau json.loads()

            if not raw_data:
                continue

            logger.info(f"🤔 Validating signal for {raw_data.get('symbol')}...")

            # 2. VALIDASI DENGAN LLM
            result = await self.validate_signal(raw_data)

            # 3. KEPUTUSAN
            if result["decision"] == "APPROVED":
                final_payload = {
                    **raw_data,
                    "llm_reason": result["reason"],
                    "status": "VALIDATED",
                    "validator": "Llama-3-70b",
                }

                # Kirim ke Notifier (atau Executor jika ada)
                # Karena ini standalone, kita bisa print atau
                # kembalikan ke bus dengan flag baru jika ada consumer lain.
                # Untuk saat ini, kita log saja.
                logger.info(
                    f"✅ SIGNAL APPROVED: {raw_data['symbol']} | Reason: {result['reason']}"
                )

                # Opsi: Jika Anda ingin Notifier mengambil ini, Anda bisa buat queue terpisah
                # atau modifikasi Notifier untuk menerima sinyal tervalidasi.

            else:
                logger.warning(
                    f"⛔ SIGNAL REJECTED: {raw_data['symbol']} | Reason: {result['reason']}"
                )


# Bagian ini hanya untuk testing jika file dijalankan langsung
if __name__ == "__main__":

    asyncio.run(LLMStrategist(signal_bus=InternalSignalBus()).run())
