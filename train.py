import asyncio
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from features import processor

# Import DataFetcher for single symbol fetching
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiTrainer")


async def train_single_pair(symbol):
    logger.info(f"🚀 START TRAINING: {symbol}")

    fetcher = yfinance.Ticker(symbol)

    # 1. Fetch Data
    # Fetch historical market data for the given symbol
    df = fetcher.history(period="max", interval="15m")
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()

    if df.empty:
        logger.error("❌ No data for %s", symbol)
        return

    # 2. Process Features & Save Scaler (Specific to Symbol)
    df, scaled_data = processor.process(df, symbol=symbol, is_training=True)

    # 3. Create Sequences
    X, y = [], []
    closes = df["close"].values
    threshold = settings.VOLATILITY_THRESHOLD

    for i in range(len(scaled_data) - settings.SEQ_LEN - settings.PREDICTION_WINDOW):
        X.append(scaled_data[i : i + settings.SEQ_LEN])
        curr = closes[i + settings.SEQ_LEN]
        fut = closes[i + settings.SEQ_LEN + settings.PREDICTION_WINDOW]
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
        return

    # 4. Convert to Tensor
    tensor_x = torch.FloatTensor(X)
    tensor_y = torch.LongTensor(y)

    # 5. Train Model
    model = TimeSeriesTransformer(input_dim=4, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    loader = DataLoader(
        TensorDataset(tensor_x, tensor_y), batch_size=settings.BATCH_SIZE, shuffle=True
    )

    logger.info(f"🏋️‍♂️ Training {symbol} for {settings.EPOCHS} epochs.... 🏋️‍♂️ ")
    model.train()
    for epoch in range(settings.EPOCHS):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            logger.info(
                f"🏋️‍♂️ Training {symbol} epoch {epoch + 1}/{settings.EPOCHS}.... Avg loss: {avg_loss:.4f} 🏋️‍♂️"
            )

    logger.info(f"🏋️‍♂️ Training {symbol} finished. Total loss: {total_loss:.4f} 🏋️‍♂️")

    # 6. Save Model (Specific to Symbol)
    save_path = settings.get_model_path(symbol)
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model saved: {save_path}")


async def main():
    # Loop semua pair di config
    for symbol in settings.ACTIVE_SYMBOLS:
        await train_single_pair(symbol)

    logger.info("🎉 ALL PAIRS TRAINED SUCCESSFULLY")


if __name__ == "__main__":
    asyncio.run(main())
