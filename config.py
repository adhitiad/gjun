# config.py
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- IDENTITAS ---
    APP_NAME: str = "Forex AI Enterprise (Hybrid Crypto/Forex)"
    VERSION: str = "4.1.0-Crypto"

    # --- DATA SOURCE (YFinance Tickers) ---
    ACTIVE_SYMBOLS: list = [
        # --- MAJOR FOREX ---
        "EURUSD=X",
        "GBPUSD=X",
        "JPY=X",
        # --- MAJOR CRYPTO ---
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "SOL-USD",
        "XRP-USD",
        "DOGE-USD",
        "ADA-USD",
        "TRX-USD",
        "DOT-USD",
        "LINK-USD",
        "BCH-USD",
        "LTC-USD",
        "XLM-USD",
        "XMR-USD",
        "ZEC-USD",
        # --- STABLECOINS & TOKENS ---
        # Note: Volatilitas USDT/USDC sangat rendah, AI mungkin jarang entry
        "USDT-USD",
        "USDC-USD",
    ]

    TIMEFRAME: str = "1h"

    # --- AI HYPERPARAMETERS ---
    SEQ_LEN: int = 60
    PREDICTION_WINDOW: int = 4
    FEATURE_DIM: int = 10

    # --- TRADING LOGIC (TP/SL) ---
    RISK_REWARD_RATIO: float = 2.0
    ATR_MULTIPLIER_SL: float = 1.5

    # Logic Pending Order:
    # Crypto sering spike tajam, kita perbesar buffer pending order sedikit
    PENDING_ORDER_BUFFER: float = 0.5

    # --- PATHS ---
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    def get_model_path(self, symbol):
        # Bersihkan simbol agar valid jadi nama file (Hapus =X, -, dll)
        clean_sym = symbol.replace("=X", "").replace("-", "").replace(".", "")
        return os.path.join(self.BASE_DIR, "data", f"model_v1_{clean_sym}.pth")

    def get_rl_path(self, symbol):
        clean_sym = symbol.replace("=X", "").replace("-", "").replace(".", "")
        return os.path.join(self.BASE_DIR, "data", f"rl_agent_{clean_sym}.zip")

    def get_scaler_path(self, symbol):
        clean_sym = symbol.replace("=X", "").replace("-", "").replace(".", "")
        return os.path.join(self.BASE_DIR, "data", f"scaler_{clean_sym}.pkl")

    # --- INFRASTRUCTURE --
    # --- INFRASTRUCTURE (FIXED) ---
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    # Tambahkan DB dan Password (default string kosong jika tidak ada di env)
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # --- API KEYS ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Channels
    CHANNEL_AI_ANALYSIS: str = "ai_raw_signals"
    CHANNEL_CONFIRMATION: str = "ai_validated_signals"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
