# Bayesian Optimization System

A YAML-driven Bayesian Optimization (BO) system for tuning the parameters in the sr_cost_routing_constants section of the JSON configuration file.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Configure your optimization parameters in `config/config.yaml`
2. Run the optimization:

```bash
python scripts/run_optimization.py -c config/config.yaml
```

The system will:
1. Load the configuration from the YAML file
2. Update the JSON constants file with different parameter values
3. Run `python3 scripts/multi_scene_runner.py` to evaluate each parameter set
4. Use Bayesian Optimization to suggest better parameters
5. Save the results to CSV and optionally to JSON

## Configuration

The YAML configuration file specifies:
- The JSON file path and section to update
- The parameters to optimize (threshold, spread, alpha, beta)
- The precision for each parameter's display and storage
- The number of trials to run
- The objectives to optimize (success rate and missed savings)

## Additional Options

```bash
# Save the best parameters to a JSON file
python scripts/run_optimization.py -c config/config.yaml --save-best

# Specify a custom output path for JSON results
python scripts/run_optimization.py -c config/config.yaml -o results/optimization_results.json

# Provide a custom black box command (instead of the default multi_scene_runner.py)
python scripts/run_optimization.py -c config/config.yaml -b "python3 scripts/custom_runner.py"
