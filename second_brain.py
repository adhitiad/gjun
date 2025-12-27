import asyncio
import datetime
import json
import logging
from typing import cast

import pandas_ta as ta
import redis.asyncio as redis
import yfinance as yf
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr

from config import settings
from knowledge_base import memory_bank
from sentiment_engine import GoogleNewsScraper, GroqAnalyzer
from state_manager import state_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionBrain")


class AIStrategist:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.6,
            model="llama3-70b-8192",
            api_key=SecretStr(settings.GROQ_API_KEY),
        )
        self.news = GoogleNewsScraper()
        self.sent = GroqAnalyzer()
        self.memory = memory_bank
        self.latest_v1 = {"action": "HOLD"}
        self.r = None
        self.active = True

    async def connect(self):
        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.r = redis.Redis(**kwargs)

    async def listen(self):
        if not self.r:
            await self.connect()
        self.r = cast(redis.Redis, self.r)
        ps = self.r.pubsub()
        await ps.subscribe(settings.CHANNEL_SIGNALS, settings.CHANNEL_SYSTEM)
        async for msg in ps.listen():
            if msg["type"] == "message":
                try:
                    d = json.loads(msg["data"])
                    ch = msg["channel"]
                    if ch == settings.CHANNEL_SIGNALS and self.active:
                        self.latest_v1 = d
                    elif ch == settings.CHANNEL_SYSTEM:
                        evt = d.get("event")
                        if evt == "TRAINING_START":
                            self.active = False
                        elif evt == "TRAINING_COMPLETED":
                            self.active = True
                            self.latest_v1 = {"action": "HOLD"}
                except:
                    pass

    async def analyze(self):
        if not self.active:
            return

        # 1. AMBIL CONFIG TERBARU DARI REDIS
        config = await state_manager.get_asset_config()
        active_symbol = config["symbol"]
        active_tf = config["timeframe"]

        safe = await state_manager.check_circuit_breaker()
        if safe["status"] == "STOP":
            await self.publish(
                {
                    "alignment": "DANGER",
                    "final_action": "HALTED",
                    "reasoning": safe["reason"],
                }
            )
            return

        try:
            df = yf.Ticker(active_symbol).history(period="5d", interval=active_tf)

            if df.empty:
                return
            df.ta.atr(length=14, append=True)
            atr = df[f"ATRr_14"].iloc[-1]
            price = df["Close"].iloc[-1]
        except:
            return

        h = self.news.get_headlines(settings.ACTIVE_SYMBOL, settings.ASSET_TYPE)
        score, summary = self.sent.analyze(h)
        sent_lbl = (
            "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Validate signals with ATR. RISK: SL=1.5xATR, TP=2xATR. OUTPUT JSON: {{alignment, final_action, entry_price, tp, sl, reasoning, solution}}",
                ),
                (
                    "user",
                    f"PRICE:{price} | V1:{self.latest_v1.get('action')} | SENT:{sent_lbl} | ATR:{atr} | NEWS:{summary}",
                ),
            ]
        )

        try:
            chain = prompt | self.llm | JsonOutputParser()
            res = await chain.ainvoke({"atr": atr})
            await self.publish(res)
            self.memory.store_memory(
                f"Act:{res.get('final_action')} Res:{res.get('alignment')}",
                {"symbol": settings.ACTIVE_SYMBOL},
            )
        except Exception as e:
            logger.error(e)

    async def publish(self, data):
        if self.r:
            data["timestamp"] = str(datetime.datetime.now())
            await self.r.publish(settings.CHANNEL_AI_ANALYSIS, json.dumps(data))

    async def run(self):
        self.memory.init_pinecone()
        await self.connect()
        asyncio.create_task(self.listen())
        while True:
            await self.analyze()
            await asyncio.sleep(300)


if __name__ == "__main__":
    asyncio.run(AIStrategist().run())
