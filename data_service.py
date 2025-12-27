import asyncio
import datetime
import logging
from concurrent import futures

import grpc

from config import settings
from database import MarketTick, get_db, init_db
from logging_config import setup_logger

# Import hasil generate proto
from protos import market_pb2, market_pb2_grpc

logger = setup_logger("DataService-GRPC")


class MarketDataServicer(market_pb2_grpc.MarketDataServiceServicer):
    def SubmitTick(self, request, context):
        """Menerima 1 Tick dan simpan ke DB"""
        db = next(get_db())
        try:
            # Level 1 Fix: Pastikan Timezone UTC
            try:
                # Handle ISO string format
                ts = datetime.datetime.fromisoformat(request.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
            except:
                ts = datetime.datetime.now(datetime.timezone.utc)

            new_tick = MarketTick(
                time=ts,
                symbol=request.symbol,
                price=request.price,
                volume=request.volume,
            )
            db.add(new_tick)
            db.commit()
            return market_pb2.TickResponse(success=True, message="Saved")
        except Exception as e:
            logger.error(f"DB Insert Error: {e}")
            return market_pb2.TickResponse(success=False, message=str(e))
        finally:
            db.close()


async def serve():
    init_db()  # Init DB saat service nyala
    server = grpc.aio.server()
    market_pb2_grpc.add_MarketDataServiceServicer_to_server(
        MarketDataServicer(), server
    )

    # Listen di Port standar gRPC
    host = "[::]:50051"
    server.add_insecure_port(host)
    logger.info(f"🚀 gRPC Data Service running on {host}")

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
