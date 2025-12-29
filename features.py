# features.py
import os

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler

from config import Settings

settings = Settings()


class FeatureEngineer:
    def __init__(self):
        self.feature_cols = [
            "returns",
            "rsi",
            "ema_dist",
            "atr",
            "macd",
            "macd_hist",
            "bb_p",
            "bb_w",
            "hour_sin",
            "hour_cos",
        ]

    def process(self, df, symbol, is_training=False):
        if df.empty:
            return df, np.empty((0, settings.FEATURE_DIM))
        df = df.copy()

        # FIX: Paksa konversi ke numeric untuk mencegah error 'isnan' pada string
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df.dropna(subset=["close"], inplace=True)
        if df.empty:
            return df, np.empty((0, settings.FEATURE_DIM))

        # Indicators
        df["returns"] = df["close"].pct_change(fill_method=None)
        df["rsi"] = df.ta.rsi(length=14) / 100.0

        ema = df.ta.ema(length=20)
        df["ema_dist"] = (df["close"] - ema) / ema

        # ATR Safe
        try:
            df["atr"] = df.ta.atr(length=14)
            df["atr"] = df["atr"].fillna(0.0)
        except:
            df["atr"] = 0.0

        # MACD
        try:
            macd = df.ta.macd(fast=12, slow=26, signal=9)
            if macd is not None:
                df["macd"] = macd.iloc[:, 0].fillna(0.0)
                df["macd_hist"] = macd.iloc[:, 1].fillna(0.0)
            else:
                df["macd"] = 0.0
                df["macd_hist"] = 0.0
        except:
            df["macd"] = 0.0
            df["macd_hist"] = 0.0

        # Bollinger
        try:
            bb = df.ta.bbands(length=20, std=2)
            if bb is not None:
                bbp = next((c for c in bb.columns if c.startswith("BBP")), None)
                bbb = next((c for c in bb.columns if c.startswith("BBB")), None)
                df["bb_p"] = bb[bbp].fillna(0.5) if bbp else 0.5
                df["bb_w"] = bb[bbb].fillna(0.0) if bbb else 0.0
            else:
                df["bb_p"] = 0.5
                df["bb_w"] = 0.0
        except:
            df["bb_p"] = 0.5
            df["bb_w"] = 0.0

        # Time Encoding
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        df.dropna(subset=["time"], inplace=True)
        hours = df["time"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

        df.dropna(inplace=True)
        df.columns = [c.lower() for c in df.columns]

        try:
            raw_data = df[self.feature_cols].values
        except KeyError:
            return df, np.empty((0, settings.FEATURE_DIM))

        path = settings.get_scaler_path(symbol)

        if is_training:
            if len(raw_data) < 10:
                return df, np.empty((0, settings.FEATURE_DIM))
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(raw_data)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(scaler, path)
        else:
            if os.path.exists(path) and len(raw_data) > 0:
                scaler = joblib.load(path)
                scaled_data = scaler.transform(raw_data)
            else:
                return df, np.empty((0, settings.FEATURE_DIM))

        return df, scaled_data


processor = FeatureEngineer()
