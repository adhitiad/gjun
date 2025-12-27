import asyncio
import json
import logging
import os
from collections import deque

import numpy as np
import pandas as pd
import redis.asyncio as redis
from stable_baselines3 import PPO

from config import settings
from features import processor
from state_manager import state_manager
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain-RL-Multi")


class BrainRL:
    def __init__(self):
        # Dictionary untuk menyimpan model per symbol
        # Contoh: {'EURUSD': <PPO Object>, 'XAUUSD': <PPO Object>}
        self.models = {}
        self.buffers = {}

        # Init buffer per symbol
        for sym in settings.ACTIVE_SYMBOLS:
            self.buffers[sym] = deque(
                maxlen=200
            )  # RL butuh data terakhir untuk feature extraction

        self.r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )

    async def load_all_agents(self):
        """Load semua model RL yang tersedia untuk active symbols"""
        for sym in settings.ACTIVE_SYMBOLS:
            # Nama file harus match dengan yang di train_rl.py
            model_path = os.path.join(settings.BASE_DIR, "data", f"rl_model_{sym}.zip")

            if os.path.exists(model_path):
                try:
                    # Load model
                    self.models[sym] = PPO.load(model_path)
                    logger.info(f"🤖 Agent RL Loaded: {sym}")
                except Exception as e:
                    logger.error(f"❌ Failed to load RL agent {sym}: {e}")
            else:
                logger.warning(f"⚠️ RL Model not found for {sym}. Skipping.")

    async def run(self):
        await self.load_all_agents()
        await streamor.connect()
        logger.info(f"🧠 Brain RL Running for: {settings.ACTIVE_SYMBOLS}")

        while True:
            # Terima data market
            candles = await streamor.consume_market_data()
            if not candles:
                await asyncio.sleep(0.01)
                continue

            for c in candles:
                symbol = c["symbol"]

                # Skip jika symbol tidak aktif atau agent-nya belum ada
                if symbol not in settings.ACTIVE_SYMBOLS or symbol not in self.models:
                    continue

                # Masukkan buffer
                self.buffers[symbol].append(c)

                # Cek Posisi Aktif (Jangan trade jika sudah ada posisi)
                if await state_manager.get_active_position(symbol):
                    continue

                # Butuh minimal data untuk scaling
                if len(self.buffers[symbol]) < 20:
                    continue

                try:
                    # 1. Prepare Data (Single Row Inference)
                    # Kita ambil snapshot terakhir, tapi processor butuh df untuk scaling
                    df = pd.DataFrame(list(self.buffers[symbol]))

                    # Process: scaling menggunakan scaler spesifik symbol
                    _, scaled_data = processor.process(df, symbol=symbol)

                    if len(scaled_data) == 0:
                        continue

                    # Ambil data point terakhir sebagai observasi
                    obs = scaled_data[-1]

                    # 2. Predict Action using Specific Agent
                    agent = self.models[symbol]
                    action, _states = agent.predict(obs, deterministic=True)

                    # Map Action (0: HOLD, 1: BUY, 2: SELL)
                    final_act = "HOLD"
                    if action == 1:
                        final_act = "BUY"
                    elif action == 2:
                        final_act = "SELL"

                    if final_act != "HOLD":
                        logger.info(f"💡 RL Signal {symbol}: {final_act}")
                        # Push Signal dengan Source BRAIN_RL
                        await streamor.push_signal(
                            {
                                "action": final_act,
                                "symbol": symbol,
                                "confidence": 65.0,  # RL biasanya tidak output probability langsung di stable-baselines
                                "reason": "RL Agent Decision",
                                "source": "BRAIN_RL",
                                "type": "MARKET",
                            }
                        )

                except Exception as e:
                    logger.error(f"RL Prediction Error {symbol}: {e}")


if __name__ == "__main__":
    asyncio.run(BrainRL().run())
