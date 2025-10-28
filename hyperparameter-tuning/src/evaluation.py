"""
Parameter evaluation system for hyperparameter tuning.

This module handles evaluating parameter sets by running multi-scene tests
and collecting performance metrics (success rate and missed savings).
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import json

from config import HyperparameterConfig
from json_updater import JSONUpdater


class ParameterEvaluator:
    """
    Evaluates parameter sets by running multi-scene tests and collecting metrics.
    
    This class handles:
    - Updating JSON constants files with new parameters
    - Running multi-scene simulations
    - Collecting and aggregating performance metrics
    - Calculating combined optimization scores
    """
    
    def __init__(self, config: HyperparameterConfig, scripts_dir: str = "scripts"):
        """
        Initialize the parameter evaluator.
        
        Args:
            config: Hyperparameter configuration object
            scripts_dir: Directory containing simulation scripts
        """
        self.config = config
        self.scripts_dir = scripts_dir
        
        # Initialize JSON updater
        self.json_updater = JSONUpdater(
            file_path=config.experiment.constants_file['path'],
            section=config.experiment.constants_file['section']
        )
        
        # Validate required scripts exist
        self._validate_scripts()
        
        # Scene configuration
        self.scene_dirs = self._discover_scene_directories()
        
    def _validate_scripts(self):
        """Validate that required scripts exist."""
        required_scripts = ['multi_scene_runner.py', 'results.py']
        
        for script in required_scripts:
            script_path = os.path.join(self.scripts_dir, script)
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Required script not found: {script_path}")
    
    def _discover_scene_directories(self) -> List[str]:
        """
        Discover available scene directories.
        
        Returns:
            List of scene directory paths
        """
        scene_dirs = []
        base_dir = "."
        
        for item in os.listdir(base_dir):
            if item.startswith('scene-') and os.path.isdir(item):
                # Check if scene has required files
                schema_path = os.path.join(item, 'schema.yaml')
                transactions_path = os.path.join(item, 'transactions.csv')
                
                if os.path.exists(schema_path) and os.path.exists(transactions_path):
                    scene_dirs.append(item)
        
        scene_dirs.sort()  # Ensure consistent ordering
        print(f"Discovered {len(scene_dirs)} scene directories: {scene_dirs}")
        return scene_dirs
    
    def evaluate_parameters(self, parameters: Dict[str, Any], 
                          num_scenes: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a set of parameters by running multi-scene tests.
        
        Args:
            parameters: Dictionary of parameter names to values
            num_scenes: Number of scenes to use (None for all available)
            
        Returns:
            Tuple of (combined_score, detailed_results)
        """
        print(f"\n=== Evaluating Parameters ===")
        for name, value in parameters.items():
            print(f"{name}: {value}")
        
        # Update JSON constants file
        print("\n--- Updating JSON constants file ---")
        try:
            self.json_updater.update_parameters(parameters, create_backup=True)
        except Exception as e:
            print(f"Error updating JSON file: {e}")
            return 0.0, {'error': str(e)}
        
        # Run multi-scene evaluation
        print("\n--- Running multi-scene evaluation ---")
        try:
            scene_results = self._run_multi_scene_evaluation(num_scenes)
        except Exception as e:
            print(f"Error running multi-scene evaluation: {e}")
            return 0.0, {'error': str(e)}
        
        # Calculate metrics
        print("\n--- Calculating performance metrics ---")
        metrics = self._calculate_metrics(scene_results)
        
        # Calculate combined score
        combined_score = self._calculate_combined_score(metrics)
        
        # Prepare detailed results
        detailed_results = {
            'parameters': parameters,
            'metrics': metrics,
            'combined_score': combined_score,
            'scene_results': scene_results,
            'num_scenes': len(scene_results)
        }
        
        print(f"\n--- Evaluation Results ---")
        print(f"Success Rate: {metrics['success_rate']:.2f}%")
        print(f"Missed Savings: ${metrics['missed_savings']:.2f}")
        print(f"Combined Score: {combined_score:.4f}")
        
        return combined_score, detailed_results
    
    def _run_multi_scene_evaluation(self, num_scenes: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run multi-scene evaluation using the multi_scene_runner script.
        
        Args:
            num_scenes: Number of scenes to evaluate
            
        Returns:
            List of scene results
        """
        # Determine scenes to use
        scenes_to_use = self.scene_dirs[:num_scenes] if num_scenes else self.scene_dirs
        
        if not scenes_to_use:
            raise ValueError("No scenes available for evaluation")
        
        # Prepare command
        script_path = os.path.join(self.scripts_dir, 'multi_scene_runner.py')
        cmd = [sys.executable, script_path] + scenes_to_use
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the multi-scene runner
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        print(f"Multi-scene evaluation completed in {elapsed_time:.1f} seconds")
        
        if result.returncode != 0:
            print(f"Multi-scene runner failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Multi-scene evaluation failed: {result.stderr}")
        
        # Parse results
        scene_results = self._parse_multi_scene_results(result.stdout)
        return scene_results
    
    def _parse_multi_scene_results(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse the output from multi_scene_runner.py.
        
        Args:
            output: Raw output from the multi-scene runner
            
        Returns:
            List of parsed scene results
        """
        scene_results = []
        
        # Try to parse as JSON first
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    result = json.loads(line.strip())
                    if 'scene' in result and 'success_rate' in result:
                        scene_results.append(result)
        except json.JSONDecodeError:
            print("Warning: Could not parse output as JSON, trying alternative parsing")
        
        # If JSON parsing failed, try to extract from CSV-like output
        if not scene_results:
            try:
                # Look for CSV-like lines
                lines = output.strip().split('\n')
                for line in lines:
                    if ',' in line and ('success_rate' in line or 'scene' in line.lower()):
                        # Simple CSV parsing
                        parts = line.split(',')
                        if len(parts) >= 3:
                            scene_results.append({
                                'scene': parts[0].strip(),
                                'success_rate': float(parts[1].strip()),
                                'missed_savings': float(parts[2].strip())
                            })
            except (ValueError, IndexError):
                print("Warning: Could not parse CSV-like output")
        
        # If still no results, try to read from output files
        if not scene_results:
            scene_results = self._read_results_from_files()
        
        print(f"Parsed {len(scene_results)} scene results")
        return scene_results
    
    def _read_results_from_files(self) -> List[Dict[str, Any]]:
        """
        Read results from output files generated by scenes.
        
        Returns:
            List of scene results from files
        """
        scene_results = []
        
        for scene_dir in self.scene_dirs:
            # Look for output_results.csv
            output_file = os.path.join(scene_dir, 'output_results.csv')
            
            if os.path.exists(output_file):
                try:
                    df = pd.read_csv(output_file)
                    
                    # Extract metrics (assuming specific column names)
                    success_rate = df['success_rate'].iloc[0] if 'success_rate' in df.columns else 0.0
                    missed_savings = df['missed_savings'].iloc[0] if 'missed_savings' in df.columns else 0.0
                    
                    scene_results.append({
                        'scene': scene_dir,
                        'success_rate': success_rate,
                        'missed_savings': missed_savings
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not read results from {output_file}: {e}")
        
        return scene_results
    
    def _calculate_metrics(self, scene_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregated metrics from scene results.
        
        Args:
            scene_results: List of scene result dictionaries
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not scene_results:
            return {'success_rate': 0.0, 'missed_savings': 0.0}
        
        # Extract values
        success_rates = [r['success_rate'] for r in scene_results]
        missed_savings = [r['missed_savings'] for r in scene_results]
        
        # Apply aggregation method from config
        agg_method = self.config.objectives.aggregation['success_rate']
        if agg_method == 'mean':
            avg_success_rate = np.mean(success_rates)
        elif agg_method == 'median':
            avg_success_rate = np.median(success_rates)
        elif agg_method == 'min':
            avg_success_rate = np.min(success_rates)
        elif agg_method == 'max':
            avg_success_rate = np.max(success_rates)
        else:
            avg_success_rate = np.mean(success_rates)  # default
        
        agg_method = self.config.objectives.aggregation['missed_savings']
        if agg_method == 'mean':
            avg_missed_savings = np.mean(missed_savings)
        elif agg_method == 'median':
            avg_missed_savings = np.median(missed_savings)
        elif agg_method == 'min':
            avg_missed_savings = np.min(missed_savings)
        elif agg_method == 'max':
            avg_missed_savings = np.max(missed_savings)
        else:
            avg_missed_savings = np.mean(missed_savings)  # default
        
        return {
            'success_rate': avg_success_rate,
            'missed_savings': avg_missed_savings,
            'success_rate_std': np.std(success_rates),
            'missed_savings_std': np.std(missed_savings),
            'num_scenes': len(scene_results)
        }
    
    def _calculate_combined_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate combined optimization score from metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Combined optimization score
        """
        success_rate = metrics['success_rate']
        missed_savings = metrics['missed_savings']
        
        # Combined objective: success_rate - (missed_savings / 5.0)
        # This gives higher weight to success rate while penalizing missed savings
        combined_score = success_rate - (missed_savings / 5.0)
        
        return combined_score
    
    def evaluate_baseline(self) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the baseline parameters from configuration.
        
        Returns:
            Tuple of (baseline_score, baseline_results)
        """
        print("\n=== Evaluating Baseline Parameters ===")
        
        baseline_params = self.config.baseline.parameters
        return self.evaluate_parameters(baseline_params)
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current parameters from the JSON constants file.
        
        Returns:
            Dictionary of current parameter values
        """
        return self.json_updater.get_all_parameters()
    
    def restore_parameters(self, parameters: Dict[str, Any]):
        """
        Restore specific parameters to the JSON constants file.
        
        Args:
            parameters: Dictionary of parameter names to values
        """
        print(f"\n--- Restoring parameters ---")
        for name, value in parameters.items():
            print(f"{name}: {value}")
        
        self.json_updater.update_parameters(parameters, create_backup=False)
    
    def cleanup_backups(self, keep_count: int = 5):
        """
        Clean up old backup files.
        
        Args:
            keep_count: Number of recent backups to keep
        """
        self.json_updater.cleanup_old_backups(keep_count)
    
    def __str__(self) -> str:
        """String representation of the evaluator."""
        return f"ParameterEvaluator(scenes={len(self.scene_dirs)}, " \
               f"config={self.config.experiment.name})"


class EvaluationCache:
    """
    Simple cache for parameter evaluations to avoid redundant computations.
    """
    
    def __init__(self):
        """Initialize the evaluation cache."""
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, params_key: str) -> Optional[Tuple[float, Dict[str, Any]]]:
        """
        Get cached evaluation result.
        
        Args:
            params_key: String key representing the parameters
            
        Returns:
            Cached result or None if not found
        """
        if params_key in self.cache:
            self.hit_count += 1
            return self.cache[params_key]
        
        self.miss_count += 1
        return None
    
    def put(self, params_key: str, result: Tuple[float, Dict[str, Any]]):
        """
        Store evaluation result in cache.
        
        Args:
            params_key: String key representing the parameters
            result: Evaluation result to cache
        """
        self.cache[params_key] = result
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate
        }
    
    @staticmethod
    def make_params_key(parameters: Dict[str, Any]) -> str:
        """
        Create a string key from parameters dictionary.
        
        Args:
            parameters: Dictionary of parameters
            
        Returns:
            String key for caching
        """
        # Sort parameters to ensure consistent key generation
        sorted_items = sorted(parameters.items())
        return str(sorted_items)
