import json
import os
from typing import Optional

import logging.config


def setup_logging(default_path: str = 'src/configs/logging/logging_config.json', default_level: int = logging.INFO) -> None:
    """
    Setup logging configuration.

    Args:
        default_path (str): The default path to the logging configuration file.
        default_level (int): The default logging level.
        env_key (str): The environment variable key to check for a custom logging configuration file path.

    Returns:
        None
    """
    path: str = default_path
    try:
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config: dict = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
    except Exception as e:
        print(f"Error occurred while setting up logging: {e}")
