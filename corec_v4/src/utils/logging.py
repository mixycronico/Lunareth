import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("corec.log")
    ]
)

logger = logging.getLogger("CoreC")

if os.getenv("ENVIRONMENT") == "production":
    pass