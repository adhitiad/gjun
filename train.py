# FILE: train.py
import asyncio
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from features import processor
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")


async def train_single_pair(symbol):
    logger.info(f"🚀 START TRAINING: {symbol} (TF: {settings.TIMEFRAME})")

    # 1. Fetch Data
    try:
        # Download data
        df = await asyncio.to_thread(
            yf.download,
            tickers=symbol,
            period="2y",
            interval=settings.TIMEFRAME,
            progress=False,
            multi_level_index=False,  # Penting untuk versi baru yfinance
        )

        # --- LOGIKA CLEANING DATA (PERBAIKAN UTAMA) ---

        # 1. Jika kosong, stop
        if df.empty:
            logger.warning(f"⚠️ No data for {symbol}. Skipping.")
            return

        # 2. Reset Index agar Date/Datetime turun menjadi kolom biasa
        df.reset_index(inplace=True)

        # 3. Ratakan MultiIndex columns jika masih ada (misal: (Close, EURUSD=X))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 4. Ubah semua nama kolom jadi huruf kecil & hapus spasi
        df.columns = [str(c).lower().strip() for c in df.columns]

        # 5. Cari kolom waktu dan ubah jadi 'time'
        # Kemungkinan nama dari yfinance: 'date', 'datetime', 'timestamp', 'index'
        rename_map = {
            "date": "time",
            "datetime": "time",
            "timestamp": "time",
            "index": "time",
        }
        df.rename(columns=rename_map, inplace=True)

        # 6. Fallback (Jaga-jaga): Jika 'time' masih tidak ada, paksa kolom pertama jadi 'time'
        if "time" not in df.columns:
            logger.warning(
                f"⚠️ 'time' column missing for {symbol}, forcing first column."
            )
            df.rename(columns={df.columns[0]: "time"}, inplace=True)

        # 7. Pastikan format datetime
        df["time"] = pd.to_datetime(df["time"], utc=True)

        # -----------------------------------------------

    except Exception as e:
        logger.error(f"❌ Download/Cleaning Error {symbol}: {e}")
        return

    if len(df) < 500:
        logger.warning(f"⚠️ Not enough data points ({len(df)}) for {symbol}. Skipping.")
        return

    # 2. Process Features & Save Scaler
    try:
        df, scaled_data = processor.process(df, symbol=symbol, is_training=True)
    except Exception as e:
        logger.error(f"Feature Error {symbol}: {e}")
        # Debugging: Print kolom jika error lagi untuk diagnosa
        logger.error(f"Columns available: {list(df.columns)}")
        return

    if len(scaled_data) < settings.SEQ_LEN:
        return

    # 3. Create Sequences (X, y)
    X, y = [], []
    closes = df["close"].values
    threshold = 0.002

    limit = len(scaled_data) - settings.SEQ_LEN - settings.PREDICTION_WINDOW
    for i in range(limit):
        X.append(scaled_data[i : i + settings.SEQ_LEN])
        curr = closes[i + settings.SEQ_LEN]
        fut = closes[i + settings.SEQ_LEN + settings.PREDICTION_WINDOW]

        if curr == 0:
            continue

        diff = (fut - curr) / curr

        if diff > threshold:
            y.append(1)  # BUY
        elif diff < -threshold:
            y.append(2)  # SELL
        else:
            y.append(0)  # HOLD

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        logger.warning(f"⚠️ No valid sequences for {symbol}")
        return

    # 4. Convert to Tensor
    tensor_x = torch.FloatTensor(X)
    tensor_y = torch.LongTensor(y)

    # 5. Train Model
    model = TimeSeriesTransformer(input_dim=settings.FEATURE_DIM, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"🏋️‍♂️ Training {symbol}...")
    model.train()

    # Epoch diperbanyak sedikit agar hasil lebih matang
    for epoch in range(5):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # 6. Save Model
    save_path = settings.get_model_path(symbol)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model Saved: {save_path}")


async def main():
    logger.info("🔥 STARTING MASS TRAINING 🔥")
    logger.info(
        f"Pairs: {len(settings.ACTIVE_SYMBOLS)} | Timeframe: {settings.TIMEFRAME}"
    )

    for symbol in settings.ACTIVE_SYMBOLS:
        await train_single_pair(symbol)
        await asyncio.sleep(3)  # Jeda ringan

    logger.info("🎉 ALL TRAINED.")


if __name__ == "__main__":
    asyncio.run(main())
