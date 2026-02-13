import logging
import os
from datetime import datetime

LOGS_DIR = "logs"

os.makedirs(LOGS_DIR, exist_ok=True)

# Fixed: Used single quotes inside strftime and added % before 'd'
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%y-%m_%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger