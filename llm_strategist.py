import asyncio
import json
import logging

import redis.asyncio as redis
from langchain_groq import ChatGroq

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM-Strategist")


class LLMStrategist:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        # Model Llama3 70B sangat bagus untuk reasoning
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY, model="llama3-70b-8192", temperature=0.1
        )

    async def validate_signal(self, data):
        """Chain of Thought Validation"""
        risk = abs(data["entry"] - data["sl"])
        reward = abs(data["tp"] - data["entry"])
        rr_ratio = reward / risk if risk > 0 else 0

        prompt = f"""
        You are a Senior Forex Risk Manager. Validate this trade signal.
        
        TRADE:
        - Pair: {data['symbol']}
        - Action: {data['action']} ({data['order_type']})
        - ATR: {data['atr']:.5f}
        - Risk/Reward: 1:{rr_ratio:.2f}

        TASK: Think step-by-step.
        1. Is the Risk:Reward ratio healthy (>= 1:1.5)?
        2. Is the Stop Loss logical given the ATR volatility?
        
        OUTPUT JSON ONLY:
        {{
            "decision": "APPROVED" or "REJECTED",
            "reason": "Short explanation"
        }}
        """

        try:
            logger.info(f"🤔 LLM Thinking about {data['symbol']}...")
            response = await self.llm.ainvoke(prompt)
            # Bersihkan output markdown
            content = response.content.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(content)
            return analysis
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {"decision": "APPROVED", "reason": "LLM Offline (Auto-Approve)"}

    async def run(self):
        logger.info("🧠 LLM Strategist Started.")
        ps = self.r.pubsub()
        await ps.subscribe(settings.CHANNEL_AI_ANALYSIS)

        async for msg in ps.listen():
            if msg["type"] == "message":
                raw_data = json.loads(msg["data"])

                # Validasi dengan LLM
                result = await self.validate_signal(raw_data)

                if result["decision"] == "APPROVED":
                    final_payload = {
                        **raw_data,
                        "llm_reason": result["reason"],
                        "status": "VALIDATED",
                    }
                    # Publish ke Notifier & Frontend
                    await self.r.publish(
                        settings.CHANNEL_CONFIRMATION, json.dumps(final_payload)
                    )
                    logger.info(f"✅ APPROVED: {raw_data['symbol']}")
                else:
                    logger.warning(
                        f"⛔ VETOED: {raw_data['symbol']} - {result['reason']}"
                    )


if __name__ == "__main__":
    asyncio.run(LLMStrategist().run())
