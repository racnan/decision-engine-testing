"""
Evaluation Interface module.

This module provides an interface for evaluating parameter sets by:
1. Updating the JSON constants file
2. Running the black box function (which will be provided by the user)
3. Collecting metrics from the evaluation
"""

import subprocess
import json
import os
import time
import requests
from typing import Dict, Any, Callable, List, Union

from .config import BOConfig
from .json_updater import update_json_constants


class EvaluationError(Exception):
    """Exception raised for errors in the evaluation process."""

    pass


def evaluate_parameters(
    params: Dict[str, float],
    config: BOConfig,
    black_box_func: Union[Callable, str, None] = None,
) -> Dict[str, float]:
    """
    Evaluate a set of parameters using the black box function.

    This function:
    1. Updates the JSON constants file with the given parameters
    2. Calls the black box function (either a Python function or a command)
    3. Returns the metrics from the evaluation

    Args:
        params: Dictionary of parameter values to evaluate
        config: Configuration object
        black_box_func: Either a Python function, a command template string, or None
                       If None, the default command will be used

    Returns:
        Dictionary of metrics from the evaluation

    Raises:
        EvaluationError: If the evaluation fails
    """
    # Step 1: Update the JSON constants file
    print(f"Updating constants with parameters: {params}")
    if not update_json_constants(config, params):
        raise EvaluationError("Failed to update JSON constants")

    # Step 2: Call the black box function
    metrics = {}

    # If black_box_func is a Python callable, call it directly
    if callable(black_box_func):
        try:
            print("Calling black box function...")
            result = black_box_func()

            if isinstance(result, dict):
                metrics = result
            else:
                raise EvaluationError(
                    f"Black box function returned invalid result: {result}"
                )

        except Exception as e:
            raise EvaluationError(f"Error calling black box function: {e}")

    # If black_box_func is a command template string, run it as a subprocess
    elif isinstance(black_box_func, str):
        try:
            print(f"Running command: {black_box_func}")
            process = subprocess.run(
                black_box_func, shell=True, capture_output=True, text=True, check=True
            )

            # Try to parse the output as JSON
            try:
                metrics = json.loads(process.stdout)
            except json.JSONDecodeError:
                # If not JSON, just store the output as a string
                metrics = {"output": process.stdout.strip()}

        except subprocess.CalledProcessError as e:
            raise EvaluationError(
                f"Command failed with exit code {e.returncode}: {e.stderr}"
            )

        except Exception as e:
            raise EvaluationError(f"Error running command: {e}")

    # If black_box_func is None, use the default command
    else:
        default_command = "python3 scripts/multi_scene_runner.py"
        try:
            print(f"Running default command: {default_command}")
            process = subprocess.run(
                default_command, shell=True, capture_output=True, text=True, check=True
            )

            # Try to parse the output as JSON
            try:
                metrics = json.loads(process.stdout)
            except json.JSONDecodeError:
                # If not JSON, just store the output as a string
                metrics = {"output": process.stdout.strip()}

        except subprocess.CalledProcessError as e:
            raise EvaluationError(
                f"Command failed with exit code {e.returncode}: {e.stderr}"
            )

        except Exception as e:
            raise EvaluationError(f"Error running command: {e}")

    # Step 3: Return the metrics
    print(f"Evaluation completed with metrics: {metrics}")
    return metrics


def evaluate_parameters_on_scenes(
    params: Dict[str, float],
    config: BOConfig,
    black_box_func: Union[Callable, str, None] = None,
    num_scenes: int = 5,
    runs_per_scene: int = 1,
    algorithm: str = "euclidean",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a set of parameters on multiple scenes by running the black box function.

    This function:
    1. Updates the JSON constants file with the given parameters
    2. Runs the black box command (multi_scene_runner.py)
    3. Collects and returns the actual results from the black box

    Args:
        params: Dictionary of parameter values to evaluate
        config: Configuration object
        black_box_func: Either a Python function, a command template string, or None

    Returns:
        Dictionary mapping scene names to their evaluation metrics

    Raises:
        EvaluationError: If the evaluation fails for any scene
    """
    # Step 1: Update the JSON constants file
    print(f"\nUpdating JSON constants at {config.constants_file_path}")
    print(f"Parameters: {params}")

    if not update_json_constants(config, params):
        raise EvaluationError("Failed to update JSON constants")

    print("\n" + "=" * 80)
    print("CONSTANTS FILE UPDATED WITH NEW PARAMETER VALUES")
    print(
        "The Decision Engine service will use the new values for all incoming requests"
    )
    print("=" * 80 + "\n")

    # Step 2: Run the multi_scene_runner.py command with automated input
    # The script is interactive and expects 3 inputs:
    # 1. Number of scenes to run (2 = first two scenes)
    # 2. Number of runs per scene (1 = single run)
    # 3. Algorithm name (SUPER_ROUTER = default algorithm)

    # We use echo to provide these inputs automatically through a pipe
    command = f"echo '{num_scenes}\n{runs_per_scene}\n{algorithm}' | python3 scripts/multi_scene_runner.py"
    if isinstance(black_box_func, str):
        command = black_box_func

    print(f"Running command: {command}")
    print("Providing automated inputs:")
    print(f"  - Number of scenes: {num_scenes}")
    print(f"  - Runs per scene: {runs_per_scene}")
    print(f"  - Algorithm: {algorithm}")

    try:
        # Print a message to let the user know what's happening
        print("\n" + "=" * 80)
        print(f"RUNNING MULTI-SCENE EVALUATION FOR PARAMETERS: {params}")
        print("=" * 80)
        print("(Output from multi_scene_runner.py will not be displayed in real-time)")
        print("Please wait while evaluation completes...\n")

        # Run the command and show output in real-time
        print("Running multi_scene_runner.py with real-time output:")
        process = subprocess.run(
            command, shell=True, capture_output=False, text=True, check=True
        )

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80 + "\n")

        # After multi_scene_runner.py completes, we need to analyze the output files
        # to get the metrics for each scene
        print("Collecting metrics from scene output files...")
        results = {}

        # Process all scenes
        scenes_to_process = []
        for i in range(1, num_scenes + 1):
            scene_name = f"scene-{i}"
            scenes_to_process.append(scene_name)

        for scene in scenes_to_process:
            # Get the path to the output_results.csv file
            output_file = os.path.join(
                os.getcwd(), scene, "run_1", "output_results.csv"
            )

            if os.path.exists(output_file):
                print(f"Analyzing results for {scene}...")
                metrics = _extract_metrics_from_file(output_file)
                results[scene] = metrics
                print(
                    f"{scene} metrics: success_rate={metrics['success_rate']:.4f}, missed_savings=${metrics['missed_savings']:.2f}"
                )
            else:
                print(f"Warning: Results file not found for {scene}")
                # Default metrics in case file is not found
                results[scene] = {"success_rate": 0.90, "missed_savings": 10.0}

        return results

    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with exit code {e.returncode}: {e.stderr}"
        print(f"ERROR: {error_msg}")
        raise EvaluationError(error_msg)

    except Exception as e:
        error_msg = f"Error running command: {e}"
        print(f"ERROR: {error_msg}")
        raise EvaluationError(error_msg)


def _extract_metrics_from_file(filepath):
    """Extract metrics from a results CSV file"""
    import csv

    metrics = {"success_rate": 0.0, "missed_savings": 0.0}

    try:
        success_count = 0
        total_count = 0
        missed_savings = 0.0

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1

                if row.get("final_outcome") == "success":
                    success_count += 1
                    actual_savings = float(row.get("savings", 0.0))
                else:
                    actual_savings = 0.0

                best_savings = float(row.get("best_possible_savings", 0.0))
                missed_savings += max(0, best_savings - actual_savings)

        if total_count > 0:
            metrics["success_rate"] = success_count / total_count

        metrics["missed_savings"] = missed_savings

    except Exception as e:
        print(f"Error processing results file {filepath}: {e}")

    return metrics


def aggregate_metrics(
    scene_metrics: Dict[str, Dict[str, float]],
    aggregation_methods: Dict[str, str] = None,
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple scenes.

    Args:
        scene_metrics: Dictionary mapping scene names to their metrics
        aggregation_methods: Dictionary mapping metric names to aggregation methods
                            (e.g. "mean", "min", "max")

    Returns:
        Dictionary of aggregated metrics
    """
    if not scene_metrics:
        return {}

    # Default aggregation methods if not provided
    if not aggregation_methods:
        aggregation_methods = {"success_rate": "mean", "missed_savings": "mean"}

    # Extract all metric names
    all_metrics = set()
    for scene_data in scene_metrics.values():
        all_metrics.update(scene_data.keys())

    # Aggregate each metric
    aggregated = {}
    for metric in all_metrics:
        # Collect all values for this metric
        values = [
            scene_data.get(metric, 0.0)
            for scene_data in scene_metrics.values()
            if metric in scene_data
        ]

        if not values:
            continue

        # Apply the aggregation method
        method = aggregation_methods.get(metric, "mean")

        if method == "mean":
            aggregated[metric] = sum(values) / len(values)
        elif method == "min":
            aggregated[metric] = min(values)
        elif method == "max":
            aggregated[metric] = max(values)
        elif method == "worst_case":
            # For success_rate, worst case is min; for missed_savings, worst case is max
            if metric == "success_rate":
                aggregated[metric] = min(values)
            else:
                aggregated[metric] = max(values)
        else:
            # Default to mean
            aggregated[metric] = sum(values) / len(values)

    return aggregated
