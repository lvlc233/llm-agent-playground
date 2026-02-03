from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, colorize=True)

logger.info("Hello Loguru")
logger.info("<green>Green Text</green>")
print("Standard Print")
