#!/usr/bin/env python3
"""
Script to save the best parameters from optimization results in JSON format.

This script converts the best parameters from the optimization results CSV
to a simple JSON file, avoiding any serialization issues with NumPy types.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# Add the parent directory to the Python path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def find_best_params(csv_path):
    """
    Find the best parameters from the optimization results CSV.

    Args:
        csv_path: Path to the optimization results CSV

    Returns:
        Dictionary with best parameters
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Find the row with the highest success rate
    best_row_idx = df["agg_success_rate"].idxmax()
    best_row = df.iloc[best_row_idx]

    # All parameter columns are those that don't start with 'agg_' and don't contain 'scene'
    # This will include sr_threshold, spread_threshold, alpha, beta
    param_cols = [
        col
        for col in df.columns
        if col != "trial" and not col.startswith("agg_") and not "scene" in col
    ]

    # Create a dictionary with best parameters
    best_params = {param: float(best_row[param]) for param in param_cols}

    # Also include the success rate and missed savings
    metrics = {
        "success_rate": float(best_row["agg_success_rate"]),
        "missed_savings": float(best_row["agg_missed_savings"]),
    }

    return {"best_params": best_params, "metrics": metrics}


def save_to_json(data, output_path):
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        output_path: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert any NumPy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.number):
            return float(obj)
        else:
            return obj

    # Convert and save
    with open(output_path, "w") as f:
        json.dump(convert_numpy(data), f, indent=2)

    print(f"Best parameters saved to {output_path}")


def main():
    # Default paths relative to the project root
    default_csv = "hyperparameter-tuning/results/optimization_results.csv"
    default_output = "hyperparameter-tuning/results/best_params.json"

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Save best parameters from optimization results as JSON"
    )

    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default=default_csv,
        help=f"Path to the optimization results CSV file (default: {default_csv})",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=default_output,
        help=f"Path to save the JSON file (default: {default_output})",
    )

    # Parse arguments
    args = parser.parse_args()

    # Find best parameters and save to JSON
    best_params = find_best_params(args.csv)
    save_to_json(best_params, args.output)

    # Print the best parameters
    print("\nBest parameters:")
    for name, value in best_params["best_params"].items():
        print(f"{name}: {value}")

    print("\nMetrics:")
    for name, value in best_params["metrics"].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
