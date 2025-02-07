import os
from data_prep.utilities import setup_logging

# Define directories and constants.
CURRENT_DIR = os.getcwd()
DATA_DIR = os.path.join(CURRENT_DIR, "data")
LOG_DIR = os.path.join(CURRENT_DIR, "logs")
OUTPUT_DIR = os.path.join(DATA_DIR, "predictions")
TEMP_DIR = os.path.join(CURRENT_DIR, "temp")

# Optionally, initialize logging here.
# setup_logging(os.path.join(LOG_DIR, "FastRaceML.log"))

# Additional constants.
MAX_NPP = 10
MAX_NWRK = 12
