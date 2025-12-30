import asyncio
import json
import logging

import pandas as pd
import redis.asyncio as redis
import yfinance as yf

from brain import InternalSignalBus
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketStreamer")


class MarketStreamer:
    def __init__(self, signal_bus: InternalSignalBus):
        self.signal_bus = signal_bus
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.running = False

    async def connect(self):
        try:
            await self.signal_bus.connect()
            logger.info(f"✅ Streamer connected to Redis DB {settings.REDIS_DB}")
            self.running = True
        except redis.exceptions.ResponseError as e:
            if "index is out of range" in str(e):
                logger.error("❌ Redis DB Index Error. Fallback to DB 0.")
                self.r = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=0,
                    password=settings.REDIS_PASSWORD,
                    decode_responses=True,
                )
                await self.signal_bus.connect()
                logger.info("✅ Connected to Redis DB 0 (Fallback)")
                self.running = True
        except Exception as e:
            logger.error(f"❌ Redis Connection Failed: {e}")

    async def _fetch_single_symbol(self, symbol):
        """Helper untuk download satu per satu jika bulk gagal"""
        try:
            df = yf.download(
                symbol,
                period="5d",
                interval="15m",
                progress=False,
                threads=False,  # PENTING: Matikan threads
            )
            if df.empty:
                return None

            # Cleaning
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]
            last_row = df.iloc[-1]

            return {
                "symbol": symbol,
                "time": str(last_row.name),
                "open": float(last_row.get("open", 0)),
                "high": float(last_row.get("high", 0)),
                "low": float(last_row.get("low", 0)),
                "close": float(last_row.get("close", 0)),
                "volume": float(last_row.get("volume", 0)),
            }
        except Exception:
            return None

    async def consume_market_data(self):
        if not self.running:
            return []

        data_batch = []
        symbols_str = " ".join(settings.ACTIVE_SYMBOLS)

        try:
            # 1. COBA BULK DOWNLOAD (threads=False agar stabil)
            df = yf.download(
                symbols_str,
                period="2y",
                interval="1h",
                progress=False,
                group_by="ticker",
                threads=False,  # FIX: Matikan threading untuk menghindari error NoneType
            )

            # Jika dataframe kosong atau rusak, force fallback
            if df.empty:
                raise ValueError("Empty Bulk Data")

            for symbol in settings.ACTIVE_SYMBOLS:
                try:
                    # Parsing MultiIndex DataFrame
                    if len(settings.ACTIVE_SYMBOLS) > 1:
                        if symbol not in df.columns.levels[0]:
                            # Jika simbol hilang di bulk, coba fetch manual
                            manual_data = await self._fetch_single_symbol(symbol)
                            if manual_data:
                                data_batch.append(manual_data)
                            continue

                        ticker_df = df[symbol].copy()
                    else:
                        ticker_df = df.copy()

                    ticker_df.dropna(inplace=True)
                    if ticker_df.empty:
                        continue

                    last_row = ticker_df.iloc[-1]
                    ticker_df.columns = [c.lower() for c in ticker_df.columns]

                    candle = {
                        "symbol": symbol,
                        "time": str(last_row.name),
                        "open": float(last_row.get("open", 0)),
                        "high": float(last_row.get("high", 0)),
                        "low": float(last_row.get("low", 0)),
                        "close": float(last_row.get("close", 0)),
                        "volume": float(last_row.get("volume", 0)),
                    }
                    data_batch.append(candle)

                except Exception:
                    continue

        except Exception as e:
            logger.warning(
                f"⚠️ Bulk download failed/partial ({e}). Switching to Serial Mode..."
            )
            # 2. FALLBACK: SERIAL DOWNLOAD (Satu per satu)
            # Ini lebih lambat tapi menjamin data tidak crash semua
            for sym in settings.ACTIVE_SYMBOLS:
                candle = await self._fetch_single_symbol(sym)
                if candle:
                    data_batch.append(candle)
                await asyncio.sleep(0.1)  # Beri jeda sedikit

        return data_batch

    async def push_signal(self, signal_data):
        await self.signal_bus.put(
            {"channel": settings.CHANNEL_AI_ANALYSIS, "data": signal_data}
        )


signal_bus = InternalSignalBus()
streamor = MarketStreamer(signal_bus)
