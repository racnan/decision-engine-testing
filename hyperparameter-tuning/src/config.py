"""
Configuration management for hyperparameter tuning system.

This module handles loading and validating configuration from YAML files,
including parameter ranges, objectives, and experiment settings.
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ParameterConfig:
    """Configuration for a single parameter."""
    name: str
    type: str
    precision: int
    range: Dict[str, float]
    log_scale: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    name: str
    output_csv: str
    constants_file: Dict[str, str]


@dataclass
class TrialsConfig:
    """Configuration for optimization trials."""
    startup: int
    total: int
    parallel_jobs: int


@dataclass
class ObjectivesConfig:
    """Configuration for optimization objectives."""
    success_rate: str
    missed_savings: str
    aggregation: Dict[str, str]


@dataclass
class BaselineConfig:
    """Configuration for baseline parameters and results."""
    parameters: Dict[str, Any]
    results: List[Dict[str, Any]]


class HyperparameterConfig:
    """Main configuration class for hyperparameter tuning."""
    
    def __init__(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        self.experiment = ExperimentConfig(**config_data['experiment'])
        self.trials = TrialsConfig(**config_data['trials'])
        self.objectives = ObjectivesConfig(**config_data['objectives'])
        self.baseline = BaselineConfig(**config_data['baseline'])
        
        # Parse parameter configurations
        self.parameters = []
        for param_data in config_data['parameters']:
            param = ParameterConfig(**param_data)
            self.parameters.append(param)
        
        # Store original config data for reference
        self.config_data = config_data
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration for consistency and correctness."""
        # Validate parameter ranges
        for param in self.parameters:
            if param.range['low'] >= param.range['high']:
                raise ValueError(f"Invalid range for parameter {param.name}: "
                               f"low ({param.range['low']}) must be less than high ({param.range['high']})")
            
            if param.precision < 0:
                raise ValueError(f"Precision must be non-negative for parameter {param.name}")
        
        # Validate trials
        if self.trials.startup >= self.trials.total:
            raise ValueError("Startup trials must be less than total trials")
        
        if self.trials.parallel_jobs < 1:
            raise ValueError("Parallel jobs must be at least 1")
        
        # Validate objectives
        valid_directions = ['maximize', 'minimize']
        if self.objectives.success_rate not in valid_directions:
            raise ValueError(f"Invalid success_rate objective: {self.objectives.success_rate}")
        
        if self.objectives.missed_savings not in valid_directions:
            raise ValueError(f"Invalid missed_savings objective: {self.objectives.missed_savings}")
        
        # Validate aggregation methods
        valid_aggregations = ['mean', 'median', 'min', 'max']
        for obj, agg in self.objectives.aggregation.items():
            if agg not in valid_aggregations:
                raise ValueError(f"Invalid aggregation method for {obj}: {agg}")
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return [param.name for param in self.parameters]
    
    def get_parameter_bounds(self) -> List[tuple]:
        """Get list of (low, high) bounds for each parameter."""
        return [(param.range['low'], param.range['high']) for param in self.parameters]
    
    def get_parameter_types(self) -> List[str]:
        """Get list of parameter types."""
        return [param.type for param in self.parameters]
    
    def get_parameter_precisions(self) -> List[int]:
        """Get list of parameter precisions."""
        return [param.precision for param in self.parameters]
    
    def get_log_scale_flags(self) -> List[bool]:
        """Get list of log scale flags for each parameter."""
        return [param.log_scale for param in self.parameters]
    
    def format_parameter_value(self, param_name: str, value: float) -> Any:
        """
        Format a parameter value according to its type and precision.
        
        Args:
            param_name: Name of the parameter
            value: Raw float value to format
            
        Returns:
            Formatted value (int, float, etc.)
        """
        param = next((p for p in self.parameters if p.name == param_name), None)
        if param is None:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        if param.type == 'int':
            return int(round(value))
        elif param.type == 'float':
            return round(value, param.precision)
        else:
            return value
    
    def get_baseline_score(self) -> float:
        """
        Calculate baseline score from baseline results.
        
        Returns:
            Combined baseline score
        """
        if not self.baseline.results:
            return 0.0
        
        # Aggregate baseline results
        success_rates = [r['success_rate'] for r in self.baseline.results]
        missed_savings = [r['missed_savings'] for r in self.baseline.results]
        
        if self.objectives.aggregation['success_rate'] == 'mean':
            avg_success_rate = sum(success_rates) / len(success_rates)
        elif self.objectives.aggregation['success_rate'] == 'median':
            avg_success_rate = sorted(success_rates)[len(success_rates) // 2]
        else:
            avg_success_rate = success_rates[0]  # fallback
        
        if self.objectives.aggregation['missed_savings'] == 'mean':
            avg_missed_savings = sum(missed_savings) / len(missed_savings)
        elif self.objectives.aggregation['missed_savings'] == 'median':
            avg_missed_savings = sorted(missed_savings)[len(missed_savings) // 2]
        else:
            avg_missed_savings = missed_savings[0]  # fallback
        
        # Combined objective: success_rate - (missed_savings / 5.0)
        return avg_success_rate - (avg_missed_savings / 5.0)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"HyperparameterConfig(experiment={self.experiment.name}, " \
               f"parameters={len(self.parameters)}, trials={self.trials.total})"
