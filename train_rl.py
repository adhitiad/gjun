import asyncio
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config import settings
from features import processor


# --- 1. REAL ENVIRONMENT SETUP ---
class ClinicalForexEnv(gym.Env):
    def __init__(self, df, features):
        super().__init__()
        self.df = df
        self.features = features
        self.current_step = 0
        self.max_steps = len(features) - 1

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(settings.FEATURE_DIM,), dtype=np.float32
        )

        self.position = 0
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        return self.features[self.current_step], {}

    def step(self, action):
        curr_price = self.df.iloc[self.current_step]["close"]
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        next_price = self.df.iloc[self.current_step]["close"]
        pct_change = (next_price - curr_price) / curr_price

        reward = 0
        spread_cost = 0.00015

        if action == 1:  # BUY
            if self.position == 1:
                reward = pct_change
            elif self.position == -1:
                reward = -pct_change
            else:
                self.position = 1
                reward = -spread_cost

        elif action == 2:  # SELL
            if self.position == -1:
                reward = -pct_change
            elif self.position == 1:
                reward = pct_change
            else:
                self.position = -1
                reward = -spread_cost

        else:  # HOLD
            if self.position == 1:
                reward = pct_change
            elif self.position == -1:
                reward = -pct_change
            else:
                reward = -0.00001

        if reward < 0:
            reward *= 2.0

        return self.features[self.current_step], reward, terminated, False, {}


# --- 2. FACTORY FUNCTION ---
def make_env(df, scaled, rank, seed=0):
    def _init():
        env = ClinicalForexEnv(df, scaled)
        log_file = os.path.join(settings.BASE_DIR, "logs", f"env_{rank}")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        env = Monitor(env, log_file)
        env.reset(seed=seed + rank)
        return env

    return _init


async def train_rl():
    print(f"🚀 Initializing Training Pipeline...")

    for symbol in settings.ACTIVE_SYMBOLS:
        print(f"\n🦾 Training RL Agent: {symbol}")

        # 1. Download Data
        try:
            df = yf.download(symbol, period="1y", interval="1h", progress=False)
        except Exception as e:
            print(f"❌ Download Error {symbol}: {e}")
            continue

        if df.empty:
            print(f"⚠️ Skipping {symbol}, no data found.")
            continue

        # --- FIX: DATA CLEANING (BULLETPROOF) ---
        # 1. Flatten MultiIndex jika ada (Masalah umum YF baru)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Reset Index agar Date menjadi kolom
        df.reset_index(inplace=True)

        # 3. Lowercase semua nama kolom
        df.columns = [str(c).lower().strip() for c in df.columns]

        # 4. Standarisasi nama kolom 'time'
        # YFinance biasanya memberi nama: date, datetime, atau timestamp
        rename_map = {
            "date": "time",
            "datetime": "time",
            "timestamp": "time",
            "index": "time",
        }
        df.rename(columns=rename_map, inplace=True)

        # 5. FINAL CHECK: Jika kolom 'time' masih tidak ada,
        # kita paksa kolom pertama (biasanya tanggal) menjadi 'time'
        if "time" not in df.columns:
            print(f"⚠️ 'time' column missing, forcing first column as time.")
            df.rename(columns={df.columns[0]: "time"}, inplace=True)

        # 6. Pastikan format datetime benar
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception as e:
            print(f"❌ Date Conversion Error {symbol}: {e}")
            continue

        # --- END FIX ---

        # 3. Feature Engineering
        try:
            df, scaled = processor.process(df, symbol, is_training=True)
        except Exception as e:
            print(f"❌ Feature Error {symbol}: {e}")
            # Debugging: Print kolom jika error lagi
            # print(f"Columns: {df.columns}")
            continue

        if len(scaled) < 100:
            print("⚠️ Not enough data points to train.")
            continue

        # 4. Setup Training Environment
        try:
            # Gunakan DummyVecEnv dulu agar lebih stabil di Windows/Scripting
            # (Subproc kadang butuh setup __name__ yang rumit di beberapa terminal)
            env = DummyVecEnv([make_env(df, scaled, 0)])

            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,  # Silent biar ga spam
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
            )

            # Training Loop
            model.learn(total_timesteps=15000)

            path = settings.get_rl_path(symbol)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
            print(f"✅ Saved RL Agent: {symbol}")

        except Exception as e:
            print(f"❌ Model Training Error {symbol}: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(train_rl())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(train_rl())
