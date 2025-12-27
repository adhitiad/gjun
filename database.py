import datetime

from sqlalchemy import Column, DateTime, Float, String, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker
from tenacity import retry, stop_after_attempt, wait_fixed

from config import settings

Base = declarative_base()


# Tabel Tick Data (TimescaleDB Hypertable)
class MarketTick(Base):
    __tablename__ = "market_ticks"
    # Time adalah primary index untuk TimescaleDB
    time = Column(
        DateTime(timezone=True), primary_key=True, default=datetime.datetime.utcnow
    )
    symbol = Column(String, primary_key=True)
    price = Column(Float)
    volume = Column(Float)


# Setup Engine dengan Connection Pool
engine = create_engine(
    settings.DATABASE_URL, pool_size=20, max_overflow=10, pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def init_db():
    inspector = inspect(engine)
    if not inspector.has_table("market_ticks"):
        print("🛠️ Creating table 'market_ticks'...")
        MarketTick.__table__.create(bind=engine)

        # Konversi ke Hypertable (TimescaleDB Magic)
        with engine.connect() as conn:
            conn.commit()
            try:
                conn.execute(text("SELECT create_hypertable('market_ticks', 'time');"))
                # Level 1 Fix: Auto-delete data older than 1 year to save disk
                conn.execute(
                    text(
                        "SELECT add_retention_policy('market_ticks', INTERVAL '12 months');"
                    )
                )
                print("✅ TimescaleDB Hypertable & Retention Configured.")
            except SQLAlchemyError as e:
                print(f"⚠️ Hypertable init skipped/error: {e}")
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


# Alembic Migrations
# Alembic Migrations
# Alembic Migrations
