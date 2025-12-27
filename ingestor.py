import asyncio
import logging
import yfinance as yf
import pandas as pd
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YF-Ingestor")

class DataFetcher:
    async def fetch_market_data(self, symbol, days=730):
        """Ambil history dari Yahoo Finance"""
        logger.info(f"📥 Downloading {symbol} ({days} days)...")
        try:
            # Gunakan interval 1h untuk history panjang (2 tahun)
            # karena yfinance 15m dibatasi hanya 60 hari terakhir.
            # Untuk training pola jangka panjang, 1h sangat bagus.
            df = await asyncio.to_thread(
                yf.download,
                tickers=symbol,
                period=f"{days}d",
                interval="1h",
                progress=False,
                auto_adjust=True
            )

            if df.empty: return pd.DataFrame()

            # Bersihkan format kolom
            df.reset_index(inplace=True)
            df.columns = [c.lower() for c in df.columns]
            
            # Standardisasi nama kolom waktu
            if 'date' in df.columns: df.rename(columns={'date': 'time'}, inplace=True)
            if 'datetime' in df.columns: df.rename(columns={'datetime': 'time'}, inplace=True)
            
            # Hapus volume 0 (Market Libur)
            if 'volume' in df.columns:
                df = df[df['volume'] > 0]

            # Reset Index (Stitching)
            df.reset_index(drop=True, inplace=True)
            
            logger.info(f"✅ Loaded {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

class DataIngestor:
    def __init__(self):
        self.fetcher = DataFetcher()

    async def ingest(self, symbol):
        df = await self.fetcher.fetch_market_data(symbol)
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return

        # Proses data lebih lanjut...
        # ...
        
        # Simpan data ke Redis
        await self.save_to_redis(df, symbol)

    async def save_to_redis(self, df, symbol):
        # Simpan DataFrame ke Redis
        pass