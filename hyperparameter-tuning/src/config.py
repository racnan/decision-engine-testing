"""
Configuration module for the Bayesian Optimization system.

This module is responsible for:
1. Loading and parsing YAML configuration files
2. Validating configuration settings
3. Providing access to configuration values
"""

import os
import yaml
from typing import Dict, List, Any, Optional


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


class BOConfig:
    """
    Bayesian Optimization Configuration class.

    This class loads, validates, and provides access to the configuration settings
    specified in the YAML configuration file.
    """

    def __init__(self, config_path: str):
        """
        Initialize the configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._validate_config()
        self._param_precision_map = self._build_parameter_precision_map()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dict containing the parsed configuration

        Raises:
            ConfigurationError: If the file doesn't exist or can't be parsed
        """
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file: {e}")

    def _validate_config(self) -> None:
        """
        Validate the configuration to ensure all required fields are present.

        Raises:
            ConfigurationError: If the configuration is invalid
        """
        # Basic validation of required sections
        required_sections = [
            "experiment",
            "trials",
            "objectives",
            "baseline",
            "parameters",
        ]
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required section: {section}")

        # Validate that the TOML file path exists
        toml_path = (
            self.config.get("experiment", {}).get("constants_file", {}).get("path")
        )
        if toml_path and not os.path.exists(toml_path):
            raise ConfigurationError(f"TOML constants file not found: {toml_path}")
    
    def _build_parameter_precision_map(self) -> Dict[str, int]:
        """
        Build a mapping of parameter names to their precision values.
        
        Returns:
            Dictionary with parameter names as keys and precision values as values
        """
        precision_map = {}
        default_precision = 2  # Default precision if not specified
        
        for param in self.config.get("parameters", []):
            name = param.get("name")
            precision = param.get("precision", default_precision)
            precision_map[name] = int(precision)
            
        return precision_map

    # Basic accessor methods for configuration values

    @property
    def experiment_name(self) -> str:
        """Get the experiment name."""
        return self.config["experiment"]["name"]

    @property
    def output_csv_path(self) -> str:
        """Get the output CSV path."""
        return self.config["experiment"]["output_csv"]

    @property
    def constants_file_path(self) -> str:
        """Get the TOML constants file path."""
        return self.config["experiment"]["constants_file"]["path"]

    @property
    def constants_section(self) -> str:
        """Get the TOML constants section name."""
        return self.config["experiment"]["constants_file"]["section"]

    @property
    def startup_trials(self) -> int:
        """Get the number of startup trials."""
        return int(self.config["trials"]["startup"])

    @property
    def total_trials(self) -> int:
        """Get the total number of trials."""
        return int(self.config["trials"]["total"])

    @property
    def baseline_parameters(self) -> Dict[str, float]:
        """Get the baseline parameter values."""
        return self.config["baseline"]["parameters"]

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        """Get the parameters to optimize."""
        return self.config["parameters"]
        
    @property
    def objectives(self) -> Dict[str, str]:
        """Get the optimization objectives."""
        return {
            'success_rate': self.config['objectives']['success_rate'],
            'missed_savings': self.config['objectives']['missed_savings']
        }
        
    @property
    def aggregation_methods(self) -> Dict[str, str]:
        """
        Get the aggregation methods for metrics across multiple scenes.
        Uses mean aggregation by default to combine metrics from different scenes.
        """
        return {
            'success_rate': 'mean',
            'missed_savings': 'mean'
        }
    
    @property
    def precision(self) -> int:
        """
        Get the default decimal precision for parameters and metrics.
        
        Returns:
            Integer value representing number of decimal places.
            Defaults to 2 if not specified in the config.
        """
        # Default precision if not specified
        return 2
    
    def get_parameter_precision(self, param_name: str) -> int:
        """
        Get the precision for a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Integer value representing number of decimal places.
            If parameter not found, returns the default precision.
        """
        return self._param_precision_map.get(param_name, self.precision)
