#!/usr/bin/env python3
"""
Main hyperparameter optimization script.

This script runs Bayesian Optimization to find optimal parameters for
the decision engine routing algorithms.
"""

import argparse
import os
import sys
import json
import csv
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import HyperparameterConfig
from optimizer import BayesianOptimizer
from evaluation import ParameterEvaluator, EvaluationCache


class HyperparameterOptimization:
    """
    Main hyperparameter optimization orchestrator.
    """
    
    def __init__(self, config_path: str, scripts_dir: str = "scripts", 
                 results_dir: str = "results", use_cache: bool = True):
        """
        Initialize the optimization system.
        
        Args:
            config_path: Path to the configuration YAML file
            scripts_dir: Directory containing simulation scripts
            results_dir: Directory to store results
            use_cache: Whether to use evaluation caching
        """
        self.config_path = config_path
        self.scripts_dir = scripts_dir
        self.results_dir = results_dir
        self.use_cache = use_cache
        
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        self.config = HyperparameterConfig(config_path)
        print(f"Loaded configuration: {self.config}")
        
        # Initialize components
        self.optimizer = BayesianOptimizer(self.config)
        self.evaluator = ParameterEvaluator(self.config, scripts_dir)
        self.cache = EvaluationCache() if use_cache else None
        
        # Results storage
        self.optimization_results = []
        self.start_time = None
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
    
    def run_optimization(self, save_best: bool = False, num_scenes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete hyperparameter optimization.
        
        Args:
            save_best: Whether to save best parameters to JSON file
            num_scenes: Number of scenes to use for evaluation
            
        Returns:
            Dictionary with optimization results
        """
        self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"STARTING HYPERPARAMETER OPTIMIZATION")
        print(f"Experiment: {self.config.experiment.name}")
        print(f"Parameters: {', '.join(self.config.get_parameter_names())}")
        print(f"Total Trials: {self.config.trials.total}")
        print(f"Startup Trials: {self.config.trials.startup}")
        print(f"{'='*60}\n")
        
        # Phase 1: Evaluate baseline
        print("\n" + "="*50)
        print("PHASE 1: BASELINE EVALUATION")
        print("="*50)
        baseline_score, baseline_results = self.evaluator.evaluate_baseline()
        self._record_trial(0, self.config.baseline.parameters, baseline_score, baseline_results, "baseline")
        
        # Phase 2: Startup random trials
        print("\n" + "="*50)
        print("PHASE 2: STARTUP RANDOM TRIALS")
        print("="*50)
        for i in range(self.config.trials.startup):
            trial_num = i + 1
            print(f"\n--- Startup Trial {trial_num}/{self.config.trials.startup} ---")
            
            # Generate random parameters
            params = self._generate_random_parameters()
            
            # Evaluate parameters
            score, results = self._evaluate_with_cache(params, num_scenes)
            
            # Record trial
            self._record_trial(trial_num, params, score, results, "random")
            
            # Observe for optimizer
            param_array = self._params_to_array(params)
            self.optimizer.observe(param_array, score)
        
        # Phase 3: Bayesian optimization trials
        print("\n" + "="*50)
        print("PHASE 3: BAYESIAN OPTIMIZATION")
        print("="*50)
        remaining_trials = self.config.trials.total - self.config.trials.startup
        
        for i in range(remaining_trials):
            trial_num = self.config.trials.startup + i + 1
            print(f"\n--- Optimization Trial {trial_num}/{self.config.trials.total} ---")
            
            # Get suggested parameters from optimizer
            param_array = self.optimizer.suggest_next_parameters()
            params = self._array_to_params(param_array)
            
            # Evaluate parameters
            score, results = self._evaluate_with_cache(params, num_scenes)
            
            # Record trial
            self._record_trial(trial_num, params, score, results, "bayesian")
            
            # Observe for optimizer
            self.optimizer.observe(param_array, score)
            
            # Print progress
            best_params, best_score, best_iter = self.optimizer.get_best_parameters()
            improvement = score - baseline_score
            best_improvement = best_score - baseline_score
            
            print(f"Current Score: {score:.4f} (improvement: {improvement:+.4f})")
            print(f"Best Score: {best_score:.4f} (improvement: {best_improvement:+.4f}) at trial {best_iter}")
        
        # Phase 4: Final results
        print("\n" + "="*50)
        print("PHASE 4: FINAL RESULTS")
        print("="*50)
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Get final best parameters
        best_params, best_score, best_iter = self.optimizer.get_best_parameters()
        best_params_dict = self._array_to_params(best_params) if best_params is not None else None
        
        # Calculate improvements
        baseline_improvement = best_score - baseline_score
        
        # Prepare final results
        final_results = {
            'experiment': self.config.experiment.name,
            'config_path': self.config_path,
            'total_time': total_time,
            'total_trials': self.config.trials.total,
            'baseline': {
                'parameters': self.config.baseline.parameters,
                'score': baseline_score,
                'results': baseline_results
            },
            'best': {
                'parameters': best_params_dict,
                'score': best_score,
                'iteration': best_iter,
                'improvement_over_baseline': baseline_improvement
            },
            'optimization_history': self.optimization_results,
            'cache_stats': self.cache.get_stats() if self.cache else None
        }
        
        # Print summary
        self._print_final_summary(final_results)
        
        # Save results
        self._save_results(final_results)
        
        # Save best parameters to JSON if requested
        if save_best and best_params_dict:
            self._save_best_parameters(best_params_dict)
        
        return final_results
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within bounds."""
        params = {}
        
        for param_config in self.config.parameters:
            low = param_config.range['low']
            high = param_config.range['high']
            
            if param_config.log_scale:
                # Log scale sampling
                log_low = np.log(low)
                log_high = np.log(high)
                log_value = np.random.uniform(log_low, log_high)
                value = np.exp(log_value)
            else:
                # Linear scale sampling
                value = np.random.uniform(low, high)
            
            # Format according to parameter type and precision
            params[param_config.name] = self.config.format_parameter_value(
                param_config.name, value
            )
        
        return params
    
    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameters dictionary to numpy array."""
        param_names = self.config.get_parameter_names()
        return np.array([params[name] for name in param_names])
    
    def _array_to_params(self, param_array: np.ndarray) -> Dict[str, Any]:
        """Convert numpy array to parameters dictionary."""
        param_names = self.config.get_parameter_names()
        params = {}
        
        for i, name in enumerate(param_names):
            params[name] = self.config.format_parameter_value(name, param_array[i])
        
        return params
    
    def _evaluate_with_cache(self, params: Dict[str, Any], 
                           num_scenes: Optional[int] = None) -> tuple:
        """Evaluate parameters with optional caching."""
        if self.cache:
            params_key = EvaluationCache.make_params_key(params)
            cached_result = self.cache.get(params_key)
            
            if cached_result:
                print(f"Using cached result for parameters: {params}")
                return cached_result
        
        # Evaluate parameters
        score, results = self.evaluator.evaluate_parameters(params, num_scenes)
        
        if self.cache:
            self.cache.put(params_key, (score, results))
        
        return score, results
    
    def _record_trial(self, trial_num: int, params: Dict[str, Any], 
                     score: float, results: Dict[str, Any], trial_type: str):
        """Record a trial in the optimization history."""
        trial_record = {
            'trial': trial_num,
            'type': trial_type,
            'parameters': params.copy(),
            'score': score,
            'metrics': results.get('metrics', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_results.append(trial_record)
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final optimization summary."""
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        
        print(f"Experiment: {results['experiment']}")
        print(f"Total Time: {results['total_time']:.1f} seconds")
        print(f"Total Trials: {results['total_trials']}")
        
        baseline = results['baseline']
        best = results['best']
        
        print(f"\nBaseline Results:")
        print(f"  Parameters: {baseline['parameters']}")
        print(f"  Score: {baseline['score']:.4f}")
        
        print(f"\nBest Results:")
        print(f"  Parameters: {best['parameters']}")
        print(f"  Score: {best['score']:.4f}")
        print(f"  Improvement: {best['improvement_over_baseline']:+.4f}")
        print(f"  Found at trial: {best['iteration']}")
        
        if self.cache:
            cache_stats = results['cache_stats']
            print(f"\nCache Statistics:")
            print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
            print(f"  Cache Size: {cache_stats['cache_size']}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to files."""
        # Save detailed JSON results
        json_path = os.path.join(self.results_dir, f"{self.config.experiment.name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {json_path}")
        
        # Save CSV summary
        csv_path = os.path.join(self.results_dir, self.config.experiment.output_csv)
        self._save_csv_results(csv_path, results)
        print(f"CSV results saved to: {csv_path}")
    
    def _save_csv_results(self, csv_path: str, results: Dict[str, Any]):
        """Save optimization results to CSV file."""
        fieldnames = ['trial', 'type', 'score', 'improvement_over_baseline']
        
        # Add parameter columns
        param_names = self.config.get_parameter_names()
        fieldnames.extend(param_names)
        
        # Add metric columns
        fieldnames.extend(['success_rate', 'missed_savings', 'num_scenes'])
        fieldnames.append('timestamp')
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            baseline_score = results['baseline']['score']
            
            for trial in results['optimization_history']:
                row = {
                    'trial': trial['trial'],
                    'type': trial['type'],
                    'score': trial['score'],
                    'improvement_over_baseline': trial['score'] - baseline_score,
                    'timestamp': trial['timestamp']
                }
                
                # Add parameters
                for param_name in param_names:
                    row[param_name] = trial['parameters'].get(param_name, '')
                
                # Add metrics
                metrics = trial.get('metrics', {})
                row['success_rate'] = metrics.get('success_rate', '')
                row['missed_savings'] = metrics.get('missed_savings', '')
                row['num_scenes'] = metrics.get('num_scenes', '')
                
                writer.writerow(row)
    
    def _save_best_parameters(self, best_params: Dict[str, Any]):
        """Save best parameters to JSON file."""
        output_path = os.path.join(self.results_dir, 'best_params.json')
        
        # Prepare output data
        output_data = {
            'experiment': self.config.experiment.name,
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_path,
            'constants_file': self.config.experiment.constants_file,
            'best_parameters': best_params,
            'score': self.optimizer.best_score,
            'iteration': self.optimizer.best_iteration
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Best parameters saved to: {output_path}")
        
        # Also update the JSON constants file with best parameters
        try:
            self.evaluator.restore_parameters(best_params)
            print(f"Updated constants file with best parameters")
        except Exception as e:
            print(f"Warning: Could not update constants file: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('-c', '--config', required=True, 
                       help='Path to configuration YAML file')
    parser.add_argument('--save-best', action='store_true',
                       help='Save best parameters to JSON file')
    parser.add_argument('--scenes', type=int,
                       help='Number of scenes to use for evaluation')
    parser.add_argument('--scripts-dir', default='../scripts',
                       help='Directory containing simulation scripts')
    parser.add_argument('--results-dir', default='results',
                       help='Directory to store results')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable evaluation caching')
    
    args = parser.parse_args()
    
    # Validate configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize optimization
        optimization = HyperparameterOptimization(
            config_path=args.config,
            scripts_dir=args.scripts_dir,
            results_dir=args.results_dir,
            use_cache=not args.no_cache
        )
        
        # Run optimization
        results = optimization.run_optimization(
            save_best=args.save_best,
            num_scenes=args.scenes
        )
        
        print(f"\nOptimization completed successfully!")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
