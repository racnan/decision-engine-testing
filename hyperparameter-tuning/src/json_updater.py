"""
Simple JSON Constants Updater.

This module provides a minimal implementation for updating constants
in a JSON configuration file based on parameters from the Bayesian optimization.
"""

import os
import json
import time
from typing import Dict

from .config import BOConfig


def update_json_constants(config: BOConfig, params: Dict[str, float]) -> bool:
    """
    Update constants in the JSON file specified in the config.

    Args:
        config: Configuration object containing JSON path
        params: Dictionary of parameter names and values to update (from Bayesian optimization)

    Returns:
        True if the update was successful, False otherwise
    """
    # Get JSON path from the config
    json_file_path = config.constants_file_path

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        return False

    try:
        # Read the JSON file
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Apply parameter updates with proper rounding
        for name, value in params.items():
            if isinstance(value, float):
                precision = config.get_parameter_precision(name)
                data[name] = round(value, precision)
            else:
                data[name] = value

        # Write the updated JSON back to the file
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=2)

        # Sleep briefly to allow the server to pick up changes
        print("Waiting 3 seconds for server to detect changes...")
        time.sleep(3)

        print(f"Successfully updated parameters in JSON file")
        return True

    except Exception as e:
        print(f"Error updating JSON file: {e}")
        return False
