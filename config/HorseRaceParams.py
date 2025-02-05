# FastRaceML/HorseRaceParams.py

import os
from data_prep.utilities import setup_logging

# Example param definitions
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "Data")
log_dir = os.path.join(current_dir, "Log")
output_dir = os.path.join(data_dir, "predictions")
temp_dir = os.path.join(current_dir, "Temp")

# Possibly call setup_logging, or let pipeline handle that
# setup_logging(os.path.join(log_dir, "FastRaceML.log"))

# Additional constants or paths
max_npp = 10
max_nwrk = 12
