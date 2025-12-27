# brain.py (Bagian run method saja yang perlu diperhatikan, tapi ini Full File agar aman)
import asyncio
import json
import logging
import os
from collections import deque

import numpy as np
import pandas as pd
import redis.asyncio as redis
import torch
import torch.nn.functional as F

from config import settings
from features import processor
from model import TimeSeriesTransformer
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain-Engine")


class Brain:
    def __init__(self):
        self.models = {}
        self.buffers = {
            s: deque(maxlen=settings.SEQ_LEN + 50) for s in settings.ACTIVE_SYMBOLS
        }

        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )

    async def load_models(self):
        for sym in settings.ACTIVE_SYMBOLS:
            path = settings.get_model_path(sym)
            try:
                if os.path.exists(path):
                    model = TimeSeriesTransformer(input_dim=settings.FEATURE_DIM)
                    model.load_state_dict(torch.load(path))
                    model.eval()
                    self.models[sym] = model
                    logger.info(f"🧠 Model Loaded: {sym}")
                # Silent warning if model not found (maybe not trained yet)
            except Exception as e:
                logger.warning(f"Model error {sym}: {e}")

    def calculate_strategy(self, action_idx, current_price, atr):
        signal_type = "HOLD"
        order_type = "MARKET"

        if action_idx == 1:
            signal_type = "BUY"
        elif action_idx == 2:
            signal_type = "SELL"

        if signal_type == "HOLD":
            return None

        sl_dist = atr * settings.ATR_MULTIPLIER_SL
        tp_dist = sl_dist * settings.RISK_REWARD_RATIO

        entry_price = current_price

        # Pending Order Check
        if atr > (current_price * 0.001):
            if signal_type == "BUY":
                order_type = "BUY LIMIT"
                entry_price = current_price - (atr * settings.PENDING_ORDER_BUFFER)
            else:
                order_type = "SELL LIMIT"
                entry_price = current_price + (atr * settings.PENDING_ORDER_BUFFER)

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
            "atr": atr,
        }

    async def run(self):
        await self.load_models()
        await streamor.connect()
        logger.info("🧠 Brain Engine Started (YF Data)")

        while True:
            candles = await streamor.consume_market_data()
            if not candles:
                await asyncio.sleep(2)  # Sleep lebih lama jika data kosong
                continue

            for c in candles:
                sym = c["symbol"]
                if sym not in self.buffers:
                    continue

                # Masukkan ke buffer
                self.buffers[sym].append(c)

                # Hanya proses jika buffer sudah cukup penuh untuk sequence
                if len(self.buffers[sym]) < settings.SEQ_LEN:
                    continue

                # Convert ke DF
                df = pd.DataFrame(list(self.buffers[sym]))

                # --- SAFETY CHECK SEBELUM PROSES ---
                # Pastikan kolom numeric valid
                if "close" not in df.columns:
                    continue

                try:
                    df_processed, scaled = processor.process(df, sym)
                except Exception as e:
                    logger.error(f"Feature Error {sym}: {e}")
                    continue

                if len(scaled) < settings.SEQ_LEN:
                    continue

                # Inference Logic
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

                        if pred != 0 and conf > 70:
                            last_price = c["close"]
                            # Ambil ATR terakhir (Safe access)
                            last_atr = df_processed.iloc[-1].get(
                                "atr", last_price * 0.001
                            )

                            strat = self.calculate_strategy(pred, last_price, last_atr)

                            if strat:
                                payload = {
                                    "symbol": sym,
                                    **strat,
                                    "confidence": conf,
                                    "timestamp": c.get("time", ""),
                                }

                                logger.info(
                                    "⚡ RAW SIGNAL: %s %s",
                                    payload["symbol"],
                                    payload["action"],
                                )
                                await self.r.publish(
                                    settings.CHANNEL_AI_ANALYSIS, json.dumps(payload)
                                )
                    except Exception as e:
                        logger.error(f"Inference Error {sym}: {e}")

            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(Brain().run())
