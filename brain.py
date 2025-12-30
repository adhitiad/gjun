import asyncio
import datetime
import json
import logging
import os

import pandas as pd
import redis.asyncio as redis
import torch
import torch.nn.functional as F
import yfinance as yf
from cycler import L

from config import settings
from database import MarketTick, SessionLocal, init_db
from features import processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain-Engine")


class InternalSignalBus:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def put(self, signal):
        await self.queue.put(signal)

    async def get(self):
        return await self.queue.get()


class Brain:
    def __init__(self, signal_bus: InternalSignalBus):
        self.models = {}
        self.signal_bus = signal_bus
        # Koneksi ke Redis untuk mengirim sinyal
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )

    async def load_models(self):
        """Memuat model PyTorch yang sudah dilatih."""
        for sym in settings.ACTIVE_SYMBOLS:
            path = settings.get_model_path(sym)
            try:
                if os.path.exists(path):
                    model = TimeSeriesTransformer(input_dim=settings.FEATURE_DIM)
                    model.load_state_dict(torch.load(path))
                    model.eval()
                    self.models[sym] = model
                    logger.info(f"🧠 Model Loaded: {sym}")
                else:
                    logger.warning(
                        f"⚠️ Model not found for {sym} at {path}. (Train first!)"
                    )
            except Exception as e:
                logger.warning(f"Model load error {sym}: {e}")

    def calculate_strategy(self, action_idx, current_price, atr):
        """Menghitung Entry, SL, dan TP berdasarkan ATR."""
        signal_type = "HOLD"
        order_type = "MARKET"

        if action_idx == 1:
            signal_type = "BUY"
        elif action_idx == 2:
            signal_type = "SELL"

        if signal_type == "HOLD":
            return None

        # Gunakan ATR untuk SL/TP dynamic (Volatilitas)
        safe_atr = atr if atr > 0 else current_price * 0.001
        sl_dist = safe_atr * settings.ATR_MULTIPLIER_SL
        tp_dist = sl_dist * settings.RISK_REWARD_RATIO

        entry_price = current_price

        if signal_type == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        return {
            "action": signal_type,
            "order_type": order_type,
            "entry": entry_price,
            "tp": tp,
            "sl": sl,
            "atr": safe_atr,
        }

    async def fetch_live_data(self, symbol):
        """Mengambil data LIVE langsung dari YFinance tanpa lewat DB dulu."""
        try:
            # Ambil data sedikit lebih banyak untuk perhitungan indikator
            df = await asyncio.to_thread(
                yf.download,
                tickers=symbol,
                period="5d",
                interval=settings.TIMEFRAME,
                progress=False,
                threads=False,
            )

            if df.empty:
                return None

            # Tambahkan kolom-kolom fitur lainnya sesuai kebutuhan
            # df["feature_name"] = processor.calculate_feature(df)

            # --- CLEANING DATA ---
            df.reset_index(inplace=True)

            # Handle MultiIndex jika ada
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]

            # Standardisasi nama kolom waktu
            rename_map = {
                "date": "time",
                "datetime": "time",
                "timestamp": "time",
                "index": "time",
            }
            df.rename(columns=rename_map, inplace=True)

            if "time" not in df.columns:
                df.rename(columns={df.columns[0]: "time"}, inplace=True)

            # Feedback visual di log
            # logging.info(
            #     f"✅ Data Fetched: {symbol}, ({len(df)} candles), {df['time'].iloc[-1]}"
            # )

            return df
        except Exception as e:
            logger.error(f"YF Download Error {symbol}: {e}")
            return None

    async def save_history_to_db(self, symbol, df):
        """Menyimpan data history ke SQLite secara background (Asynchronous)."""
        if df is None or df.empty:
            return
        await asyncio.to_thread(self._sync_save, symbol, df)

    def _sync_save(self, symbol, df):
        """Fungsi sinkron untuk menyimpan tick ke database SQLite."""
        db = SessionLocal()
        try:
            # Simpan 5 candle terakhir saja untuk efisiensi
            # (Agar database tetap punya arsip history)
            for _, row in df.tail(5).iterrows():
                ts = pd.to_datetime(row["time"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)

                tick = MarketTick(
                    time=ts,
                    symbol=symbol,
                    price=float(row["close"]),
                    volume=float(row["volume"]) if "volume" in row else 0.0,
                )
                db.merge(tick)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    async def run(self):
        # 1. Inisialisasi Database (Create Table jika belum ada)
        await asyncio.to_thread(init_db)

        # 2. Load Model AI
        await self.load_models()

        logger.info(
            f"🧠 Brain Engine Started (Source: Direct YFinance | TF: {settings.TIMEFRAME})"
        )

        while True:
            for sym in settings.ACTIVE_SYMBOLS:

                # A. AMBIL DATA LIVE
                df = await self.fetch_live_data(sym)
                if df is None or len(df) < settings.SEQ_LEN:
                    continue

                # B. SIMPAN HISTORY KE DB (Background Task)
                asyncio.create_task(self.save_history_to_db(sym, df))

                # C. PROSES AI (Feature Engineering)
                try:
                    df_processed, scaled = processor.process(df, sym, is_training=False)

                except Exception as e:
                    logger.error(f"Feature Error {sym}: {e}")
                    continue

                if len(scaled) < settings.SEQ_LEN:
                    continue

                # D. INFERENCE (Prediksi)
                if sym in self.models:
                    try:
                        tensor = torch.FloatTensor(
                            scaled[-settings.SEQ_LEN :]
                        ).unsqueeze(0)

                        with torch.no_grad():
                            logits = self.models[sym](tensor)
                            probs = F.softmax(logits, dim=1)
                            top_p, top_class = torch.max(probs, dim=1)
                            pred = top_class.item()
                            conf = top_p.item() * 100

                        # Filter Confidence > 70%
                        if pred != 0 and conf > 70:
                            last_row = df_processed.iloc[-1]
                            strat = self.calculate_strategy(
                                pred, last_row["close"], last_row.get("atr", 0.0)
                            )

                            if strat:
                                # --- UPDATE PAYLOAD SESUAI PERMINTAAN ---
                                # Menyertakan Timeframe dan Sumber Data
                                payload = {
                                    "symbol": sym,
                                    **strat,
                                    "confidence": conf,
                                    "timestamp": str(last_row["time"]),
                                    "source": "YFinance_Direct",  # Sumber Data
                                    "timeframe": settings.TIMEFRAME,  # Timeframe (misal: 1h)
                                }

                                logger.info(
                                    f"⚡ SIGNAL: {sym} {strat['action']} | TF: {settings.TIMEFRAME} | Conf: {conf:.1f}%"
                                )

                                # Kirim Sinyal ke Redis
                                await self.signal_bus.put(
                                    {
                                        "channel": settings.CHANNEL_AI_ANALYSIS,
                                        "data": json.dumps(payload),
                                    }
                                )

                    except Exception as e:
                        logger.error(f"Inference Error {sym}: {e}")

            # Jeda agar tidak terkena limit Yahoo Finance
            await asyncio.sleep(60)


if __name__ == "__main__":
    signal_bus = InternalSignalBus()
    asyncio.run(Brain(signal_bus).run())
