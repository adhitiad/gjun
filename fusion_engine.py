import asyncio
import json
import logging

import redis.asyncio as redis

from config import settings
from logging_config import setup_logger
from portfolio_manager import portfolio

logger = setup_logger("Fusion-V2")


class EnterpriseFusion:
    def __init__(self):
        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.r = redis.Redis(**kwargs)
        # Bobot Voting (Total 1.0)
        self.WEIGHTS = {
            "BRAIN_V1": 0.4,  # Technical (Transformer)
            "BRAIN_RL": 0.3,  # Adaptive (PPO)
            "LLM": 0.3,  # Fundamental (Llama3)
        }

    async def get_latest_decisions(self):
        """Mengumpulkan suara dari semua anggota dewan"""
        # 1. LLM Analysis
        llm_raw = await self.r.get("llm:analysis")
        llm_vote = 0  # 0=Hold, 1=Buy, -1=Sell
        if llm_raw:
            d = json.loads(llm_raw)
            if d["bias"] == "BULLISH":
                llm_vote = 1
            elif d["bias"] == "BEARISH":
                llm_vote = -1

        return {"LLM": llm_vote}

    async def process_signal(self, tech_signal):
        """Proses setiap sinyal teknikal masuk"""
        data = json.loads(tech_signal["data"])
        source = data.get("source", "UNKNOWN")
        action = data.get("action", "HOLD")
        symbol = data.get("symbol")

        # Mapping sinyal teknikal (-1, 0, 1)
        tech_vote = 1 if action == "BUY" else -1 if action == "SELL" else 0

        # 1. Cek Macro (Veto Power)
        macro_status = await self.r.get("macro:status")
        if macro_status == "DANGER":
            logger.warning(f"🛡️ SIGNAL BLOCKED by Macro Engine (High Spread/News)")
            return

        # 2. Ambil Suara Lain (LLM)
        votes = await self.get_latest_decisions()

        # 3. Hitung Weighted Score
        # Mulai dengan score dari si pengirim sinyal
        w_source = self.WEIGHTS.get(source, 0.2)
        final_score = tech_vote * w_source

        # Tambahkan suara LLM
        final_score += votes["LLM"] * self.WEIGHTS["LLM"]

        # Tambahkan suara Brain lain (jika disimpan di Redis, logic kompleks opsional)
        # Untuk simplifikasi, kita anggap konfirmasi silang terjadi jika arah sama

        # 4. Threshold Keputusan
        # > 0.4 -> BUY, < -0.4 -> SELL
        # Kenapa 0.4? Karena jika Tech (0.4) dan LLM (0.3) sepakat = 0.7 (Strong)
        # Jika Tech Buy (0.4) tapi LLM Sell (-0.3) = 0.1 (Weak/Hold) -> Filter Noise!

        final_action = "HOLD"
        confidence = abs(final_score)

        if final_score > 0.35:
            final_action = "BUY"
        elif final_score < -0.35:
            final_action = "SELL"

        if final_action == "HOLD":
            return

        # 5. Risk Check via Portfolio Manager
        allowed, lots, reason = await portfolio.calculate_allocation(symbol, confidence)

        if allowed:
            logger.info(
                f"⚖️ FUSION DECISION: {final_action} | Score: {final_score:.2f} | Lots: {lots}"
            )

            # Publikasikan Perintah Eksekusi
            order_payload = {
                "final_action": final_action,
                "symbol": symbol,
                "lots": lots,
                "confidence": confidence,
                "sl": 0.0,  # Akan dihitung Executor/MT5 atau dynamic
                "tp": 0.0,
                "reason": f"Fusion Score: {final_score:.2f} | {reason}",
            }
            await self.r.publish(
                settings.CHANNEL_AI_ANALYSIS, json.dumps(order_payload)
            )
        else:
            logger.warning(f"⛔ Risk Rejection: {reason}")

    async def run(self):
        ps = self.r.pubsub()
        await ps.subscribe(settings.CHANNEL_SIGNALS)  # Channel Raw Signals
        logger.info("⚖️ Enterprise Fusion Engine Active (Weighted Voting)")

        async for msg in ps.listen():
            if msg["type"] == "message":
                try:
                    await self.process_signal(msg)
                except Exception as e:
                    logger.error(f"Fusion Error: {e}")


if __name__ == "__main__":
    asyncio.run(EnterpriseFusion().run())
