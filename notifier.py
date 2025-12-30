import asyncio
import json
import logging

from telegram import Bot

from brain import InternalSignalBus
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NotifierModule")


# --- NOTIFIER MODULE (PENGIRIM TELEGRAM) ---
class NotifierModule:
    def __init__(self, signal_bus: InternalSignalBus):
        self.bus = signal_bus
        self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        self.chat_id = settings.TELEGRAM_CHAT_ID

    async def send_telegram(self, data):
        if not self.chat_id:
            return

        icon = "🟢" if "BUY" in data["action"] else "🔴"
        msg = (
            f"{icon} <b>AI SIGNAL</b>\n"
            f"Symbol: <b>{data['symbol']}</b>\n"
            f"Action: {data['action']}\n"
            f"Entry: {data['entry']:.5f}\n"
            f"TP: {data['tp']:.5f}\n"
            f"SL: {data['sl']:.5f}\n"
            f"Conf: {data['confidence']:.1f}%\n"
            f"Time: {settings.TIMEFRAME}"
        )
        try:
            await self.bot.send_message(
                chat_id=self.chat_id, text=msg, parse_mode="HTML"
            )
            logger.info(f"📲 Notification sent for {data['symbol']}")
        except Exception as e:
            logger.error(f"Telegram Error: {e}")

    async def run(self):
        # Tes Koneksi
        try:
            await self.bot.send_message(
                self.chat_id,
                "🔔 <b>System Restarted</b>\nMode: Standalone (No Redis)",
                parse_mode="HTML",
            )
            logger.info("✅ Telegram Connected")
        except Exception as e:
            logger.error(f"❌ Telegram Failed: {e}")

        logger.info("📲 Notifier Module Watching...")
        while True:
            # Tunggu sinyal dari Internal Bus
            signal = await self.bus.get()
            if signal:
                # Di sini bisa ditambahkan logika Fusion (Filter) sederhana
                # Misal: Cek jika confidence < 75 skip, dsb.
                await self.send_telegram(signal)
                continue
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    bus = InternalSignalBus()
    loop = asyncio.get_event_loop()
    notifier = NotifierModule(bus)
    loop.run_until_complete(notifier.run())
