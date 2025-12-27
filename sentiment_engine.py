import asyncio
import json
import logging
import urllib.parse

import feedparser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import settings
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GNewsSentiment....")

client_nvdia = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    api_key=settings.NVIDIA_API_KEY,  # Ambil dari config
    temperature=0.6,
    top_p=0.7,
    max_completion_tokens=5135,
)


class GoogleNewsScraper:
    def __init__(self):
        self.base_url = (
            "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
        )

    def get_headlines(self, symbol, asset_type):
        try:
            clean = symbol.split("/")[0].replace("=X", "")
            query = (
                f"{clean} crypto market news when:1d"
                if asset_type == "CRYPTO"
                else f"{clean} market news when:1d"
            )
            if "XAU" in clean:
                query = "Gold price market news when:1d"

            feed = feedparser.parse(self.base_url.format(urllib.parse.quote(query)))
            headlines = [
                f"{e.title} ({e.source.get('title', '') if isinstance(e.source, dict) else ''})"
                for e in feed.entries[:7]
            ]
            logger.info(f"📰 Found {len(headlines)} news for {symbol}")
            return headlines
        except:
            return []


class GroqAnalyzer:
    def __init__(self):
        self.client = client_nvdia
        self.model = "mixtral-8x7b-32768"

    def analyze(self, headlines):
        if not headlines:
            return 0, "No news"
        prompt = f"Analyze sentiment (-1.0 to 1.0) and summary (max 10 words) for: {json.dumps(headlines)}. JSON format: {{'score': float, 'summary': string}}"
        try:
            res = self.client.invoke(
                [{"role": "user", "content": prompt}],
                kwargs={"response_format": {"type": "json_object"}},
            )
            if res.additional_kwargs and "reasoning_content" in res.additional_kwargs:
                logger.info(res.additional_kwargs["reasoning_content"])
            if isinstance(res.content, dict):
                data = res.content
            elif isinstance(res.content, str):
                data = json.loads(res.content)
            else:
                raise ValueError("Invalid response content type")
            return float(data.get("score", 0)), data.get("summary", "")
        except:
            return 0, "Error"


async def run():
    scraper = GoogleNewsScraper()
    analyzer = GroqAnalyzer()
    logger.info("🤖 Sentiment Engine Started...")
    while True:
        try:
            h = scraper.get_headlines(settings.ACTIVE_SYMBOL, settings.ASSET_TYPE)
            if h:
                score, summary = analyzer.analyze(h)
                if streamor.r:
                    await streamor.r.publish(
                        "channel_sentiment",
                        json.dumps(
                            {
                                "symbol": settings.ACTIVE_SYMBOL,
                                "sentiment_score": score,
                                "summary": summary,
                                "headline_count": len(h),
                                "source": "Google News",
                            }
                        ),
                    )
                logger.info(f"🧠 Sentiment: {score}")
            await asyncio.sleep(300)
        except:
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(run())
