import asyncio
import datetime
import logging

import MetaTrader5 as mt5
import redis.asyncio as redis

from config import settings
from logging_config import setup_logger

logger = setup_logger("MacroEngine-MT5")


class MacroEngine:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.connected = False

    def _init_mt5_sync(self):
        if not mt5.initialize(
            path=settings.MT5_PATH,
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER,
        ):
            logger.error(f"MT5 Init Failed: {mt5.last_error()}")
            return False
        self.connected = True
        return True

    def _check_market_sync(self):
        """Fungsi sync untuk cek kondisi pasar (dijalankan di thread)"""
        if not self.connected:
            if not self._init_mt5_sync():
                return "DANGER", "MT5 Disconnected"

        status = "SAFE"
        msg = "Normal"

        try:
            # 1. Cek Spread (Indikasi News/Volatilitas)
            symbol = settings.ACTIVE_SYMBOLS[0]
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                info = mt5.symbol_info(symbol)
                if info:
                    spread = tick.ask - tick.bid
                    spread_points = spread / info.point

                    if spread_points > 50:  # Spread > 50 poin (5 pips)
                        status = "DANGER"
                        msg = f"High Spread ({spread_points:.1f} pts)"

            # 2. Cek Jam (Opsional: Hindari jam swap 23:55-00:05)
            # ... tambahkan logic jam server di sini jika perlu ...

        except Exception as e:
            logger.error(f"Macro Check Error: {e}")
            return "DANGER", "Check Error"

        return status, msg

    async def run(self):
        logger.info("📅 Macro Engine Started (Non-Blocking)")
        while True:
            # Gunakan to_thread agar loop utama tidak macet saat MT5 lag
            status, msg = await asyncio.to_thread(self._check_market_sync)

            # Update Redis
            await self.r.set("macro:status", status)
            await self.r.set("macro:next_event", msg)

            if status == "DANGER":
                logger.warning(f"🚨 MARKET DANGER: {msg}")

            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(MacroEngine().run())
