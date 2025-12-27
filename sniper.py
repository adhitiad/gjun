import asyncio
import json

import redis.asyncio as redis
import websockets

from config import settings
from logging_config import setup_logger

logger = setup_logger("Sniper")


class Sniper:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )
        self.fast_price = 0

    async def binance_listener(self):
        uri = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
        async with websockets.connect(uri) as ws:
            while True:
                data = json.loads(await ws.recv())
                self.fast_price = float(data["p"])
                # Arbitrage logic check here

    async def run(self):
        logger.info("🔫 Sniper Armed")
        await self.binance_listener()


if __name__ == "__main__":
    asyncio.run(Sniper().run())
