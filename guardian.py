import asyncio

import redis.asyncio as redis

from config import settings
from logging_config import setup_logger

logger = setup_logger("Guardian")


async def watch():
    r = redis.Redis(
        host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
    )
    logger.info("👼 Guardian Watching")
    while True:
        # Cek PnL, jika rugi > 3% hari ini, shutdown
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(watch())
