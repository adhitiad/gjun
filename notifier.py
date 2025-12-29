import asyncio
import json

import redis.asyncio as redis
from telegram import Bot

from config import settings


class Notifier:
    async def run(self):
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        ps = r.pubsub()
        await ps.subscribe(settings.CHANNEL_CONFIRMATION)
        bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)

        print("📲 Notifier Ready...")
        # --- TAMBAHKAN BLOK INI UNTUK CEK KONEKSI TELEGRAM ---
        try:
            await bot.send_message(
                chat_id=settings.TELEGRAM_CHAT_ID,
                text="🔔 **SYSTEM ONLINE**\nNotifier service is active and connected to Telegram.",
            )
            print("✅ Telegram Test Message Sent!")
        except Exception as e:
            print(f"❌ Telegram Connection Failed: {e}")
        # -----------------------------------------------------

        async for msg in ps.listen():
            if msg["type"] == "message":
                data = json.loads(msg["data"])

                icon = "🟢" if "BUY" in data["action"] else "🔴"
                if "LIMIT" in data["order_type"]:
                    icon = "⏳"

                text = (
                    f"{icon} <b>AI SIGNAL CONFIRMED</b>\n"
                    f"Verified by: Groq Llama-3\n\n"
                    f"💎 <b>{data['symbol']}</b>\n"
                    f"Type: {data['order_type']}\n"
                    f"Price: {data['entry']:.5f}\n"
                    f"🎯 TP: {data['tp']:.5f}\n"
                    f"🛡 SL: {data['sl']:.5f}\n\n"
                    f"🧠 <b>Reason:</b> {data.get('llm_reason', '-')}"
                )

                try:
                    await bot.send_message(
                        chat_id=settings.TELEGRAM_CHAT_ID, text=text, parse_mode="HTML"
                    )
                except Exception as e:
                    print(f"Telegram Error: {e}")


if __name__ == "__main__":
    asyncio.run(Notifier().run())
if __name__ == "__main__":
    asyncio.run(Notifier().run())
