import asyncio

import redis.asyncio as redis
import requests

from config import settings
from logging_config import setup_logger

logger = setup_logger("SocialSensor")


async def run():
    kwargs = {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "decode_responses": True,
    }
    if settings.REDIS_PASSWORD:
        kwargs["password"] = settings.REDIS_PASSWORD
    r = redis.Redis(**kwargs)
    logger.info("🐦 Social Sensor Started")
    while True:
        try:
            res = requests.get("https://api.alternative.me/fng/", timeout=10).json()
            val = int(res["data"][0]["value"])
            norm = (val - 50) / 50.0
            await r.set("social:fng_score", norm)
        except Exception as e:
            logger.error(f"API Error: {e}")
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(run())
