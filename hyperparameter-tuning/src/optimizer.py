"""
Bayesian Optimization Core module.

This module implements the core Bayesian optimization algorithm, including:
1. Gaussian Process surrogate model
2. Acquisition functions (Expected Improvement)
3. Optimization loop
"""

import numpy as np
from typing import Dict, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import scipy.stats as stats
import pandas as pd
import os
import json

# Changed from relative import to absolute import
from src.config import BOConfig
from src.evaluation import evaluate_parameters_on_scenes, aggregate_metrics


class BayesianOptimizer:
    """
    Bayesian Optimization implementation using Gaussian Processes.

    This class implements the core Bayesian optimization algorithm for
    finding the optimal parameter values that maximize (or minimize)
    an objective function.
    """

    def __init__(self, config: BOConfig, black_box_func=None):
        """
        Initialize the Bayesian Optimizer.

        Args:
            config: Configuration object containing parameters, ranges, etc.
            black_box_func: Function or command to evaluate parameters
        """
        self.config = config
        self.black_box_func = black_box_func

        # Extract parameter names and ranges
        self.parameter_names = [p["name"] for p in config.parameters]

        self.parameter_ranges = {}
        self.log_scale_params = []
        for param in config.parameters:
            name = param["name"]
            param_range = param["range"]
            self.parameter_ranges[name] = (
                float(param_range["low"]),
                float(param_range["high"]),
            )

            if param.get("log_scale", False):
                self.log_scale_params.append(name)

        # Initialize Gaussian Process model with Matérn kernel
        # nu=2.5 corresponds to a twice-differentiable function (smoother than default)
        kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise to ensure numerical stability
            normalize_y=True,
            n_restarts_optimizer=10,
        )

        # Initialize data storage
        self.X_sample = []  # Normalized parameter vectors
        self.X_params = []  # Original parameter dictionaries
        self.y_sample = []  # Objective values
        self.metrics = []  # Full metrics dictionaries

        # Set objectives
        self.objectives = config.objectives

        # Track best result
        self.best_params = None
        self.best_value = -np.inf  # Assuming maximization
        self.best_metrics = None

        # Set balance weight for missed_savings in the objective function
        # Smaller value = more weight on missed_savings
        # Higher value = more weight on success_rate
        self.missed_savings_weight = 5.0

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """
        Normalize parameters to [0, 1] range for GP input.

        Args:
            params: Parameter dictionary

        Returns:
            Normalized parameter vector
        """
        x_normalized = []
        for name in self.parameter_names:
            value = params[name]
            low, high = self.parameter_ranges[name]

            # Handle log scale parameters
            if name in self.log_scale_params:
                log_low = np.log10(low)
                log_high = np.log10(high)
                normalized_value = (np.log10(value) - log_low) / (log_high - log_low)
            else:
                normalized_value = (value - low) / (high - low)

            x_normalized.append(normalized_value)

        return np.array(x_normalized).reshape(1, -1)

    def _denormalize_params(self, x_normalized: np.ndarray) -> Dict[str, float]:
        """
        Convert normalized parameters back to original range.

        Args:
            x_normalized: Normalized parameter vector

        Returns:
            Parameter dictionary with values in original ranges
        """
        params = {}
        for i, name in enumerate(self.parameter_names):
            low, high = self.parameter_ranges[name]
            value = x_normalized[0, i]

            # Handle log scale parameters
            if name in self.log_scale_params:
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[name] = 10 ** (log_low + value * (log_high - log_low))
            else:
                params[name] = low + value * (high - low)

            # For integer parameters, round to nearest integer
            if isinstance(low, int) and isinstance(high, int):
                params[name] = int(round(params[name]))

        return params

    def _sample_random_params(self) -> Dict[str, float]:
        """
        Sample random parameter values within the defined ranges.

        Returns:
            Parameter dictionary with random values
        """
        params = {}
        for name in self.parameter_names:
            low, high = self.parameter_ranges[name]

            # Handle log scale parameters
            if name in self.log_scale_params:
                log_low = np.log10(low)
                log_high = np.log10(high)
                value = 10 ** np.random.uniform(log_low, log_high)
            else:
                value = np.random.uniform(low, high)

            # For integer parameters, round to nearest integer
            if isinstance(low, int) and isinstance(high, int):
                value = int(round(value))

            params[name] = value

        return params

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Calculate expected improvement acquisition function.

        Args:
            X: Normalized parameter vectors to evaluate
            xi: Exploration-exploitation trade-off parameter

        Returns:
            Expected improvement values
        """
        # Get mean and standard deviation predictions from GP
        mu, sigma = self.gp.predict(X, return_std=True)

        # Reshape to ensure proper dimensions
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Get best value observed so far
        # Assuming maximization
        y_best = np.max(self.y_sample) if self.y_sample else 0.0

        # Calculate improvement
        improvement = mu - y_best - xi

        # Calculate Z score for expected improvement
        with np.errstate(divide="ignore", invalid="ignore"):
            z = improvement / sigma
            ei = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)

            # Handle division by zero or other numerical issues
            ei[sigma <= 1e-8] = 0.0

        return ei

    def suggest_next_parameters(self, n_restarts: int = 10) -> Dict[str, float]:
        """
        Suggest the next parameter values to evaluate.

        Args:
            n_restarts: Number of restarts for optimization

        Returns:
            Parameter dictionary with values to try next
        """
        # If no data yet, return random parameters
        if len(self.X_sample) < 2:
            return self._sample_random_params()

        # Update Gaussian Process model with available data
        X = np.vstack(self.X_sample)
        y = np.array(self.y_sample).reshape(-1, 1)
        self.gp.fit(X, y)

        # Generate random starting points for optimization
        best_x = None
        best_ei = -1

        # Sample random normalized parameters and find point with maximum EI
        for _ in range(n_restarts):
            x_random = np.random.rand(1, len(self.parameter_names))
            ei_value = self.expected_improvement(x_random)[0][0]

            if ei_value > best_ei:
                best_ei = ei_value
                best_x = x_random

        # Convert back to original parameter range
        return self._denormalize_params(best_x)

    def evaluate_and_update(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate a set of parameters and update the model.

        Args:
            params: Parameter dictionary to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Evaluate parameters across all scenes
        scene_metrics = evaluate_parameters_on_scenes(
            params, self.config, self.black_box_func
        )

        # Aggregate metrics according to config
        aggregated = aggregate_metrics(scene_metrics, self.config.aggregation_methods)

        # Get metrics
        success_rate = aggregated.get("success_rate", 0.0)
        missed_savings = aggregated.get("missed_savings", 0.0)

        # Create combined objective: maximize success_rate, minimize missed_savings
        # We scale missed_savings to be comparable to success_rate
        # Since we want to maximize the objective but minimize missed_savings,
        # we subtract missed_savings from success_rate
        objective_value = success_rate - (missed_savings / self.missed_savings_weight)

        print(
            f"Success Rate: {success_rate:.2f}, Missed Savings: ${missed_savings:.2f}"
        )
        print(
            f"Combined Objective: {success_rate:.2f} - ({missed_savings:.2f} / {self.missed_savings_weight}) = {objective_value:.2f}"
        )

        # Normalize parameters
        x_norm = self._normalize_params(params)

        # Store data
        self.X_sample.append(x_norm)
        self.X_params.append(params)
        self.y_sample.append(objective_value)
        self.metrics.append(
            {"params": params, "scene_metrics": scene_metrics, "aggregated": aggregated}
        )

        # Update best result based on combined objective
        if objective_value > self.best_value:
            self.best_value = objective_value
            self.best_params = params.copy()
            self.best_metrics = {
                "scene_metrics": scene_metrics,
                "aggregated": aggregated,
            }

            # Save the best parameters to JSON whenever we find a better result
            self._save_best_params_to_json()

        return {
            "params": params,
            "scene_metrics": scene_metrics,
            "aggregated": aggregated,
        }

    def _save_best_params_to_json(self):
        """
        Save the current best parameters to a JSON file.
        This ensures we always have the best parameters saved in a readable format.
        """
        # Create results directory if needed
        results_dir = os.path.join(
            os.path.dirname(self.config.config_path), "..", "results"
        )
        os.makedirs(results_dir, exist_ok=True)

        # Path to save the JSON file
        json_path = os.path.join(results_dir, "best_params.json")

        # Convert any numpy values to Python native types and round to parameter-specific precision
        best_params_dict = {}
        for k, v in self.best_params.items():
            if isinstance(v, np.number) or isinstance(v, float):
                precision = self.config.get_parameter_precision(k)
                best_params_dict[k] = round(float(v), precision)
            else:
                best_params_dict[k] = v

        # Create a dictionary with the best parameters and metrics
        data = {
            "best_params": best_params_dict,
            "metrics": {
                "success_rate": round(
                    float(self.best_metrics["aggregated"]["success_rate"]), 2
                ),
                "missed_savings": round(
                    float(self.best_metrics["aggregated"]["missed_savings"]), 2
                ),
                "combined_score": round(float(self.best_value), 2),
            },
        }

        # Save to JSON
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def run_optimization(
        self, n_startup_trials: int = 5, n_total_trials: int = 15
    ) -> Dict[str, Any]:
        """
        Run the full optimization loop.

        Args:
            n_startup_trials: Number of random initialization trials
            n_total_trials: Total number of trials (including startup)

        Returns:
            Dictionary with optimization results
        """
        print(f"Starting Bayesian Optimization with {n_total_trials} trials")
        print(f"Initial random exploration: {n_startup_trials} trials")
        print(f"Parameter space: {len(self.parameter_names)} dimensions")
        print(f"Parameters: {self.parameter_names}")

        # Print objectives
        print("\nOptimization Objectives:")
        print(
            f"- Success Rate: {self.objectives.get('success_rate', 'maximize')} (higher is better)"
        )
        print(
            f"- Missed Savings: {self.objectives.get('missed_savings', 'minimize')} (lower is better)"
        )
        print(
            f"Using combined objective: success_rate - (missed_savings / {self.missed_savings_weight})"
        )
        print(
            "  This balances both metrics, optimizing for high success rate and low missed savings"
        )

        # First, run baseline evaluation with default parameters
        baseline_params = self.config.baseline_parameters
        print("\n" + "=" * 80)
        print("BASELINE EVALUATION WITH DEFAULT PARAMETERS")
        print("=" * 80)
        print(f"Baseline parameters: {baseline_params}")

        baseline_result = self.evaluate_and_update(baseline_params)

        print(
            f"Baseline success rate: {baseline_result['aggregated']['success_rate']:.2f}"
        )
        print(
            f"Baseline missed savings: ${baseline_result['aggregated']['missed_savings']:.2f}"
        )
        print("=" * 80 + "\n")

        # Initial random exploration
        for i in range(n_startup_trials):
            print(f"\nStartup trial {i+1}/{n_startup_trials}:")
            params = self._sample_random_params()

            # Round parameters for display, using parameter-specific precision
            rounded_params = {}
            for k, v in params.items():
                if isinstance(v, float):
                    precision = self.config.get_parameter_precision(k)
                    rounded_params[k] = round(v, precision)
                else:
                    rounded_params[k] = v

            print(f"Parameters: {rounded_params}")

            result = self.evaluate_and_update(params)

            print(f"Success rate: {result['aggregated']['success_rate']:.2f}")
            print(f"Missed savings: ${result['aggregated']['missed_savings']:.2f}")

        # Bayesian optimization loop
        for i in range(n_startup_trials, n_total_trials):
            print(f"\nOptimization trial {i+1}/{n_total_trials}:")
            params = self.suggest_next_parameters()

            # Round parameters for display, using parameter-specific precision
            rounded_params = {}
            for k, v in params.items():
                if isinstance(v, float):
                    precision = self.config.get_parameter_precision(k)
                    rounded_params[k] = round(v, precision)
                else:
                    rounded_params[k] = v

            print(f"Suggested parameters: {rounded_params}")

            result = self.evaluate_and_update(params)

            print(f"Success rate: {result['aggregated']['success_rate']:.2f}")
            print(f"Missed savings: ${result['aggregated']['missed_savings']:.2f}")

        # Save results to CSV
        results_dir = os.path.join(
            os.path.dirname(self.config.config_path), "..", "results"
        )
        os.makedirs(results_dir, exist_ok=True)

        output_path = os.path.join(results_dir, self.config.output_csv_path)
        self._save_results_to_csv(output_path)

        # Extract baseline metrics for easy comparison
        baseline_metrics = next(
            (
                m["aggregated"]
                for m in self.metrics
                if m["params"] == self.config.baseline_parameters
            ),
            {"success_rate": 0.0, "missed_savings": 0.0},
        )

        # Calculate improvement over baseline
        success_rate_improvement = (
            (
                self.best_metrics["aggregated"]["success_rate"]
                - baseline_metrics["success_rate"]
            )
            / max(baseline_metrics["success_rate"], 0.001)
            * 100
        )

        # For missed_savings, lower is better, so improvement is negative
        missed_savings_improvement = (
            (
                baseline_metrics["missed_savings"]
                - self.best_metrics["aggregated"]["missed_savings"]
            )
            / max(baseline_metrics["missed_savings"], 0.001)
            * 100
        )

        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        print("Baseline vs. Best Parameters:")

        # Show success rate comparison
        print(
            f"Success Rate: {baseline_metrics['success_rate']:.2f} → {self.best_metrics['aggregated']['success_rate']:.2f} ({success_rate_improvement:+.2f}%)"
        )

        # Show missed savings comparison (improvement is when it decreases)
        print(
            f"Missed Savings: ${baseline_metrics['missed_savings']:.2f} → ${self.best_metrics['aggregated']['missed_savings']:.2f} ({missed_savings_improvement:+.2f}%)"
        )
        print("=" * 80 + "\n")

        return {
            "baseline_params": self.config.baseline_parameters,
            "baseline_metrics": baseline_metrics,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_metrics": self.best_metrics,
            "all_trials": self.metrics,
            "output_csv": output_path,
        }

    def _save_results_to_csv(self, output_path: str) -> None:
        """
        Save optimization results to a CSV file.

        Args:
            output_path: Path to save the CSV file
        """
        # Create a list of dictionaries for the DataFrame
        rows = []

        for i, (params, metrics_data) in enumerate(zip(self.X_params, self.metrics)):
            # Create a row with trial number and parameters
            row = {"trial": i + 1}

            # Round parameters using parameter-specific precision
            for k, v in params.items():
                if isinstance(v, float):
                    precision = self.config.get_parameter_precision(k)
                    row[k] = round(v, precision)
                else:
                    row[k] = v

            # Add aggregated metrics, rounded to 2 decimal places
            for key, value in metrics_data["aggregated"].items():
                if isinstance(value, float):
                    row[f"agg_{key}"] = round(value, 2)
                else:
                    row[f"agg_{key}"] = value

            # Add per-scene metrics, rounded to 2 decimal places
            for scene, scene_metrics in metrics_data["scene_metrics"].items():
                for metric_name, metric_value in scene_metrics.items():
                    if isinstance(metric_value, float):
                        row[f"{scene}_{metric_name}"] = round(metric_value, 2)
                    else:
                        row[f"{scene}_{metric_name}"] = metric_value

            rows.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def optimize(config_path: str, black_box_func=None) -> Dict[str, Any]:
    """
    Run Bayesian optimization using the specified config file.

    This is the main entry point for running the optimization.

    Args:
        config_path: Path to the YAML configuration file
        black_box_func: Function or command to evaluate parameters

    Returns:
        Dictionary with optimization results
    """
    # Load configuration
    config = BOConfig(config_path)

    # Create optimizer
    optimizer = BayesianOptimizer(config, black_box_func)

    # Run optimization
    return optimizer.run_optimization(
        n_startup_trials=config.startup_trials, n_total_trials=config.total_trials
    )
