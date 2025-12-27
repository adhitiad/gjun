# FILE: brain.py
import asyncio
import json
import logging
import os

import pandas as pd
import redis.asyncio as redis
import torch
import torch.nn.functional as F

from config import settings

# PENTING: Import fungsi fetch baru dari database yang telah kita buat sebelumnya
from database import fetch_recent_data
from features import processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain-Engine")


class Brain:
    def __init__(self):
        self.models = {}
        # Kita tidak butuh buffer deque lagi karena DB sudah menyimpan history
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )

    async def load_models(self):
        """Memuat model PyTorch yang sudah dilatih untuk setiap pair."""
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
        """Menghitung SL/TP berdasarkan volatilitas (ATR)."""
        signal_type = "HOLD"
        order_type = "MARKET"

        if action_idx == 1:
            signal_type = "BUY"
        elif action_idx == 2:
            signal_type = "SELL"

        if signal_type == "HOLD":
            return None

        # Gunakan ATR untuk SL/TP dynamic
        # Jika ATR 0 (karena data tick belum cukup), fallback ke 0.1% harga sebagai pengaman
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

    async def run(self):
        await self.load_models()

        logger.info(
            f"🧠 Brain Engine Started (Source: Database | TF: {settings.TIMEFRAME})"
        )

        while True:
            # Loop untuk setiap symbol aktif di config
            for sym in settings.ACTIVE_SYMBOLS:

                # ---------------------------------------------------------
                # 1. AMBIL DATA DARI DB (Replacing Streamor)
                # ---------------------------------------------------------
                # Ambil data secukupnya: Sequence Length + 50 candle extra untuk indikator (MA/RSI)
                required_len = settings.SEQ_LEN + 50

                # Gunakan to_thread agar query DB tidak memblokir async loop utama
                try:
                    df = await asyncio.to_thread(fetch_recent_data, sym, required_len)
                except Exception as e:
                    logger.error(f"DB Fetch Error {sym}: {e}")
                    continue

                # Cek apakah data cukup untuk diproses model
                if df.empty or len(df) < settings.SEQ_LEN:
                    # Data belum cukup di DB, skip dulu
                    # logger.debug(f"Waiting for more data: {sym}")
                    continue

                # ---------------------------------------------------------
                # 2. FEATURE ENGINEERING
                # ---------------------------------------------------------
                try:
                    # is_training=False agar menggunakan scaler yang tersimpan
                    df_processed, scaled = processor.process(df, sym, is_training=False)
                except Exception as e:
                    logger.error(f"Feature Processing Error {sym}: {e}")
                    continue

                # Pastikan hasil scaling valid
                if len(scaled) < settings.SEQ_LEN:
                    continue

                # ---------------------------------------------------------
                # 3. INFERENCE (PREDIKSI AI)
                # ---------------------------------------------------------
                if sym in self.models:
                    try:
                        # Ambil sequence terakhir sesuai panjang input model
                        tensor = torch.FloatTensor(
                            scaled[-settings.SEQ_LEN :]
                        ).unsqueeze(
                            0
                        )  # Tambah batch dimension

                        with torch.no_grad():
                            logits = self.models[sym](tensor)
                            probs = F.softmax(logits, dim=1)
                            top_p, top_class = torch.max(probs, dim=1)
                            pred = top_class.item()
                            conf = top_p.item() * 100

                        # Filter Confidence (> 70%) untuk mengurangi False Signal
                        if pred != 0 and conf > 70:
                            last_row = df_processed.iloc[-1]
                            last_price = last_row["close"]
                            last_atr = last_row.get("atr", 0.0)

                            strat = self.calculate_strategy(pred, last_price, last_atr)

                            if strat:
                                payload = {
                                    "symbol": sym,
                                    **strat,
                                    "confidence": conf,
                                    "timestamp": str(last_row["time"]),
                                    "source": "BRAIN_DB",  # Penanda bahwa ini dari DB Engine
                                }

                                logger.info(
                                    f"⚡ SIGNAL GENERATED: {sym} {strat['action']} (Conf: {conf:.1f}%)"
                                )

                                # Kirim ke Redis agar ditangkap Fusion Engine / Executor / Telegram
                                await self.r.publish(
                                    settings.CHANNEL_AI_ANALYSIS, json.dumps(payload)
                                )

                    except Exception as e:
                        logger.error(f"Inference Error {sym}: {e}")

            # ---------------------------------------------------------
            # 4. SLEEP INTERVAL
            # ---------------------------------------------------------
            # Penting: Sleep agar tidak membebani database dengan query berlebihan.
            # 5 detik cukup responsif untuk timeframe 15m atau 1h.
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(Brain().run())
