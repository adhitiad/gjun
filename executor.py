import asyncio
import json
import logging
import redis.asyncio as redis
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Signal-Logger")

class TradeExecutor:
    def __init__(self):
        self.r = None

    async def connect_redis(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT,
            db=settings.REDIS_DB, decode_responses=True
        )

    async def execute(self, s):
        """Hanya mencatat dan memberitahu Telegram"""
        action = s.get("action", "HOLD")
        if action == "HOLD": return

        symbol = s.get("symbol")
        price = float(s.get("entry_price", 0))
        
        # Log ke layar VPS
        logger.info(f"🔔 SIGNAL GENERATED: {symbol} | {action} | {price}")

        # Kirim ke Telegram (Notifier akan menangkap ini)
        if self.r:
            await self.r.publish(settings.CHANNEL_CONFIRMATION, json.dumps({
                "event": "ORDER_FILLED", # Pakai event ini agar Notifier memproses
                "symbol": symbol,
                "action": action,
                "price": price,
                "tp": s.get("tp", 0),
                "sl": s.get("sl", 0),
                "status": "SIGNAL ONLY"
            }))

    async def run(self):
        await self.connect_redis()
        ps = self.r.pubsub()
        await ps.subscribe(settings.CHANNEL_AI_ANALYSIS)
        logger.info("📡 Signal Generator Ready...")

        async for msg in ps.listen():
            if msg["type"] == "message":
                try:
                    await self.execute(json.loads(msg["data"]))
                except Exception as e:
                    logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(TradeExecutor().run())