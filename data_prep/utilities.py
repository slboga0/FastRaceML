# FastRaceML/data_prep/utilities.py

"""
General utility functions for logging setup, file manipulation, 
and other reusables.
"""

import os
import logging
import configparser

def setup_logging(log_file=None, level=logging.INFO):
    """
    Configures Python logging for the entire application.
    If log_file is specified, logs will be written there, otherwise to console.
    """
    log_format = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(filename=log_file, level=level, format=log_format)
    else:
        logging.basicConfig(level=level, format=log_format)

    logging.info("Logging configured.")

def load_config():
    """
    Loads and returns configuration from 'config.ini', 
    which should be located in the same directory as this file.
    """
    config = configparser.ConfigParser()
    config_file_path = "config\config.ini"

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}")

    config.read(config_file_path)
    return config
