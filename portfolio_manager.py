import json
import logging

import redis.asyncio as redis

from config import settings
from logging_config import setup_logger

logger = setup_logger("PortfolioMgr")


class PortfolioManager:
    def __init__(self):
        self.r = None

    async def connect(self):
        if not self.r:
            self.r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )

    async def get_account_info(self):
        """Ambil info saldo real-time yang di-update oleh Executor"""
        await self.connect()
        # Executor MT5 sekarang rajin update key ini
        balance = float(await self.r.get("account_balance") or 0.0)
        equity = float(await self.r.get("account_equity") or 0.0)
        margin_free = float(await self.r.get("account_margin_free") or 0.0)

        # Fallback jika Redis kosong (belum sync)
        if balance == 0:
            logger.warning("⚠️ Saldo Redis Kosong. Defaulting to 1000.")
            balance = 1000.0

        return balance, equity, margin_free

    async def get_open_positions_count(self):
        """Hitung posisi aktif (bisa ambil dari Redis state manager)"""
        await self.connect()
        # Hitung key position:* (tapi state manager sekarang handle ini)
        # Lebih aman tanya State Manager atau cek key spesifik
        # Simplifikasi: Cek state
        state = await self.r.get(f"bot_state:{settings.ACTIVE_SYMBOLS[0]}")
        return 1 if state else 0

    async def calculate_allocation(self, symbol, confidence):
        """Logic Manajemen Risiko (Fixed Lot / Kelly)"""
        await self.connect()

        balance, equity, free_margin = await self.get_account_info()
        open_count = await self.get_open_positions_count()

        # 1. Cek Max Positions
        if open_count >= settings.MAX_OPEN_POSITIONS:
            return False, 0.0, f"⛔ Max Positions ({open_count})"

        # 2. Cek Margin Cukup
        if free_margin < 50:  # $50 safety buffer
            return False, 0.0, "⛔ Low Margin"

        # 3. Lot Sizing (PENTING UNTUK MT5)
        # OANDA pakai 'units' (1000, 5000). MT5 pakai 'lot' (0.01, 0.1).
        # Rumus sederhana: Risk 1% dari Equity
        risk_per_trade = 0.01  # 1%
        risk_amount = equity * risk_per_trade

        # Estimasi SL 200 point (20 pips)
        # 1 Lot Standard = $10 per pip (approx)
        # Value per pip untuk 1 lot ~ $10
        stop_loss_pips = 20
        pip_value_1_lot = 10.0  # Estimasi kasar EURUSD

        # Risk = Lots * SL_Pips * Pip_Value
        # Lots = Risk / (SL_Pips * Pip_Value)
        lots = risk_amount / (stop_loss_pips * pip_value_1_lot)

        # Rounding ke step 0.01
        lots = round(max(0.01, lots), 2)

        # Adjustment berdasarkan Confidence AI
        if confidence > 0.8:
            lots = round(lots * 1.2, 2)  # Aggressive
        elif confidence < 0.6:
            lots = round(lots * 0.5, 2)  # Conservative

        # Cap Max Lot (Safety)
        lots = min(lots, 1.0)  # Jangan lebih dari 1 lot

        return True, lots, f"✅ Alloc: {lots} Lots (${risk_amount:.2f} risk)"


portfolio = PortfolioManager()
