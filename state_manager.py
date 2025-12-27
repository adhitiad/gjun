import json

import redis.asyncio as redis

from config import settings


class StateManager:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )

    def _get_key(self, symbol):
        """Membuat key unik: active_position:EURUSD"""
        return f"active_position:{symbol}"

    async def get_active_position(self, symbol):
        """Ambil posisi aktif spesifik untuk pair tertentu"""
        data = await self.r.get(self._get_key(symbol))
        return json.loads(data) if data else None

    async def set_active_position(self, symbol, side, entry, lot, tp, sl):
        """Simpan posisi aktif dengan key spesifik"""
        data = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "lot": lot,
            "tp": tp,
            "sl": sl,
            "status": "OPEN",
        }
        await self.r.set(self._get_key(symbol), json.dumps(data))

    async def clear_active_position(self, symbol):
        """Hapus posisi untuk pair tertentu"""
        await self.r.delete(self._get_key(symbol))


state_manager = StateManager()
