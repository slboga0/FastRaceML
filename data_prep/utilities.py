import os
import logging
import configparser
from pathlib import Path

def setup_logging(log_file=None, level=logging.INFO):
    """
    Configure logging for the application. If log_file is specified, logs are written there.
    """
    log_format = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=level, format=log_format)
    else:
        logging.basicConfig(level=level, format=log_format)
    logging.info("Logging configured.")

def load_config():
    """
    Load configuration from 'config/config.ini'.
    """
    config = configparser.ConfigParser()
    config_file_path = Path("config") / "config.ini"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_file_path}")
    config.read(config_file_path)
    return config
