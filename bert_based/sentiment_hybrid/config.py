from typing import Dict, Any
import yaml
import os
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")
CONFIG = load_config(CONFIG_PATH)