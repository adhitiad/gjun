# FILE: database.py
import datetime

import pandas as pd  # Pastikan install pandas: pip install pandas
from sqlalchemy import Column, DateTime, Float, String, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker
from tenacity import retry, stop_after_attempt, wait_fixed

from config import settings

Base = declarative_base()

# Create database engine
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Tabel Tick Data (TimescaleDB Hypertable)
class MarketTick(Base):
    __tablename__ = "market_ticks"
    # Time adalah primary index untuk TimescaleDB
    time = Column(
        DateTime(timezone=True),
        primary_key=True,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
    symbol = Column(String, primary_key=True)
    price = Column(Float)
    volume = Column(Float)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def init_db():
    inspector = inspect(engine)
    if not inspector.has_table("market_ticks"):
        print("🛠️ Creating table 'market_ticks'...")
        MarketTick.__table__.create(bind=engine)

        # --- MODIFIKASI UNTUK SQLITE ---
        # SQLite tidak support Hypertable. Cek dialek atau skip bagian ini jika menggunakan SQLite.
        if "sqlite" not in settings.DATABASE_URL:
            with engine.connect() as conn:
                conn.commit()
                try:
                    conn.execute(
                        text("SELECT create_hypertable('market_ticks', 'time');")
                    )
                    conn.execute(
                        text(
                            "SELECT add_retention_policy('market_ticks', INTERVAL '12 months');"
                        )
                    )
                    print("✅ TimescaleDB Hypertable & Retention Configured.")
                except SQLAlchemyError as e:
                    print(f"⚠️ Hypertable init skipped/error: {e}")
        else:
            print("ℹ️ Running in SQLite mode (No Hypertable optimization).")

    else:
        print("⚡ Table 'market_ticks' ready.")


def get_db():
    """
    Get a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- FUNGSI BARU UNTUK BRAIN ENGINE ---
def fetch_recent_data(symbol: str, limit: int = 100):
    """
    Mengambil data N candle terakhir dari database
    dan mengonversinya menjadi DataFrame untuk Brain Engine.
    """
    db = SessionLocal()
    try:
        # Query data terbaru urut waktu mundur, lalu ambil limit
        # Kita ambil 'price' sebagai 'close' karena struktur tick data sederhana
        query = text(
            """
            SELECT time, price as close, volume 
            FROM market_ticks 
            WHERE symbol = :symbol 
            ORDER BY time DESC 
            LIMIT :limit
        """
        )

        # Pandas read_sql otomatis mengonversi hasil query ke DataFrame
        df = pd.read_sql(query, db.bind, params={"symbol": symbol, "limit": limit})

        if df.empty:
            return pd.DataFrame()

        # Urutkan kembali dari lama ke baru (Ascending) untuk TimeSeries (Sequence)
        df = df.sort_values(by="time").reset_index(drop=True)

        # FIX: Isi kolom Open, High, Low (karena DB tick hanya punya Price/Close)
        # Agar features.py tidak error saat hitung indikator seperti ATR/BB
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]

        # Pastikan kolom time adalah datetime yang benar
        df["time"] = pd.to_datetime(df["time"])

        return df

    except Exception as e:
        print(f"❌ DB Fetch Error ({symbol}): {e}")
        return pd.DataFrame()
    finally:
        db.close()
