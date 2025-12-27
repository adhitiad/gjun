import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("Test")

print("Print works")
logger.info("Logging works")

try:
    import torch

    logger.info("Torch imported")
except ImportError as e:
    logger.error(f"Torch failed: {e}")

try:
    import yfinance

    logger.info("Yfinance imported")
except ImportError as e:
    logger.error(f"Yfinance failed: {e}")
