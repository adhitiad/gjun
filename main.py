import asyncio
import logging
from datetime import datetime

import MetaTrader5 as mt5
import redis.asyncio as redis

from config import settings

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | MACRO | %(message)s")
logger = logging.getLogger("MacroEngine")


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
        """Cek kondisi Spread & Waktu"""
        if not self.connected:
            if not self._init_mt5_sync():
                return "DANGER", "MT5 Disconnected"

        status = "SAFE"
        msg = "Normal Market"

        try:
            symbol = settings.ACTIVE_SYMBOLS[0]

            # 1. CEK SPREAD (FBS Check)
            # Ambil data tick terakhir
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)

            if tick and info:
                spread = tick.ask - tick.bid
                spread_points = spread / info.point

                # BATAS SPREAD INTRADAY: 25 Point (2.5 Pips)
                # Jika spread > 2.5 pips, DANGER. Jangan trading M15.
                if spread_points > 25:
                    status = "DANGER"
                    msg = f"High Spread ({spread_points:.1f} pts)"

            # 2. CEK JAM ROLLOVER (Swap Filter)
            # Hindari jam 23:50 - 01:00 waktu server (Spread gila-gilaan)
            current_server_time = mt5.symbol_info_tick(symbol).time
            dt_server = datetime.fromtimestamp(current_server_time)

            if (dt_server.hour == 23 and dt_server.minute >= 50) or (
                dt_server.hour == 0
            ):
                status = "DANGER"
                msg = "Rollover Time (High Volatility/Swap)"

        except Exception as e:
            logger.error(f"Macro Check Error: {e}")
            return "DANGER", "Check Error"

        return status, msg

    async def run(self):
        logger.info("🌍 Macro Engine Started (FBS Optimized)")
        while True:
            # Jalankan cek MT5 di thread terpisah agar tidak blocking
            status, msg = await asyncio.to_thread(self._check_market_sync)

            # Update status ke Redis
            await self.r.set("macro:status", status)
            await self.r.set("macro:reason", msg)

            if status == "DANGER":
                logger.warning(f"🚨 MARKET UNSAFE: {msg}")

            await asyncio.sleep(5)  # Cek setiap 5 detik


if __name__ == "__main__":
    asyncio.run(MacroEngine().run())
