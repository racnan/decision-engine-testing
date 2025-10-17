#!/usr/bin/env python3
"""
Main script for running Bayesian Optimization.

This script provides a command-line interface to run the
Bayesian Optimization process on the specified configuration.
"""

import os
import sys
import argparse
import yaml
import json

# Add the parent directory to the Python path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our modules
from src.config import BOConfig
from src.optimizer import optimize


def setup_parser():
    """Set up the argument parser for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization for parameter tuning"
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )

    parser.add_argument(
        "-b",
        "--blackbox",
        type=str,
        help="Command to run for evaluating parameters (optional)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the optimization results (optional)",
    )

    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save the best parameters to a JSON file",
    )

    return parser


def save_best_params(best_params, output_path):
    """
    Save the best parameters to a JSON file.

    Args:
        best_params: Dictionary of best parameter values
        output_path: Path to save the JSON file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert any numpy values to regular Python types and round to 2 decimal places
    best_params_dict = {}
    for name, value in best_params.items():
        # Handle numpy types and round float values
        try:
            if isinstance(value, float):
                best_params_dict[name] = round(float(value), 2)
            else:
                best_params_dict[name] = float(value)
                best_params_dict[name] = round(best_params_dict[name], 2)
        except (TypeError, ValueError):
            best_params_dict[name] = value
    
    # Create a dictionary with best parameters
    data = {
        "best_params": best_params_dict
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Best parameters saved to {output_path}")


def save_results_json(results, output_path):
    """
    Save the optimization results to a JSON file.

    Args:
        results: Dictionary with optimization results
        output_path: Path to save the JSON file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Helper function to convert and round values
    def convert_and_round(value):
        if hasattr(value, "item"):
            # Handle numpy values
            value = float(value)
        
        # Round float values to 2 decimal places
        if isinstance(value, float):
            return round(value, 2)
        return value
    
    # Apply conversion and rounding to nested dictionaries
    def process_dict(d):
        if not isinstance(d, dict):
            return convert_and_round(d)
        
        return {k: process_dict(v) if isinstance(v, dict) else convert_and_round(v) 
                for k, v in d.items()}

    # Clean up results for JSON serialization
    clean_results = {
        "baseline_params": process_dict(results.get("baseline_params", {})),
        "baseline_metrics": process_dict(results.get("baseline_metrics", {})),
        "best_params": process_dict(results["best_params"]),
        "best_value": convert_and_round(results["best_value"]),
        "best_metrics": {
            "aggregated": process_dict(results["best_metrics"]["aggregated"])
        },
        "output_csv": results["output_csv"],
    }

    # Save the results to JSON
    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"Optimization results saved to {output_path}")


def main():
    """Main function to run the optimization process."""
    # Set up the argument parser
    parser = setup_parser()
    args = parser.parse_args()

    # Check if the config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Run the optimization
    print(f"Starting optimization with config: {args.config}")
    if args.blackbox:
        print(f"Using black box command: {args.blackbox}")

    try:
        # Run the optimization
        results = optimize(args.config, args.blackbox)

        # Print baseline results
        print("\n=== Baseline Evaluation ===")
        print("Parameters:")
        for name, value in results.get("baseline_params", {}).items():
            if isinstance(value, float):
                print(f"  {name}: {value:.2f}")
            else:
                print(f"  {name}: {value}")
        
        print("Metrics:")
        for name, value in results.get("baseline_metrics", {}).items():
            if isinstance(value, float):
                print(f"  {name}: {value:.2f}")
            else:
                print(f"  {name}: {value}")

        # Print the best results
        print("\n=== Best Parameters Found ===")
        for name, value in results["best_params"].items():
            try:
                if isinstance(value, float):
                    print(f"{name}: {value:.2f}")
                else:
                    print(f"{name}: {float(value):.2f}")
            except (TypeError, ValueError):
                print(f"{name}: {value}")

        print("\nMetrics:")
        for name, value in results["best_metrics"]["aggregated"].items():
            try:
                if isinstance(value, float):
                    print(f"{name}: {value:.2f}")
                else:
                    print(f"{name}: {float(value):.2f}")
            except (TypeError, ValueError):
                print(f"{name}: {value}")
            
        # Print improvement
        if "baseline_metrics" in results:
            baseline_sr = results["baseline_metrics"].get("success_rate", 0)
            best_sr = results["best_metrics"]["aggregated"].get("success_rate", 0)
            if baseline_sr > 0:
                improvement = (best_sr - baseline_sr) / baseline_sr * 100
                print(f"\nImprovement over baseline: {improvement:+.2f}%")

        print(f"\nResults saved to: {results['output_csv']}")

        # Save the best parameters to JSON if requested
        if args.save_best:
            best_params_path = os.path.join(
                os.path.dirname(args.config), "..", "results", "best_params.json"
            )
            save_best_params(results["best_params"], best_params_path)

        # Save the results to JSON if requested
        if args.output:
            save_results_json(results, args.output)

    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
