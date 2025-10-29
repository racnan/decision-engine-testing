# Hyperparameter Tuning System Implementation Plan

## Overview
This document provides a complete implementation plan for deploying the Bayesian Optimization hyperparameter tuning system in your delta-based algorithm branch.

## System Architecture

### Core Algorithm: Bayesian Optimization with Gaussian Processes
- **Surrogate Model**: Gaussian Process with Matérn kernel (ν=2.5)
- **Acquisition Function**: Expected Improvement (EI)
- **Multi-Objective**: Success Rate (maximize) + Missed Savings (minimize)
- **Combined Objective**: `success_rate - (missed_savings / 5.0)`

### Two Algorithm Configurations
1. **Euclidean Algorithm**: 4 parameters (sr_threshold, spread_threshold, alpha, beta)
2. **Delta-Based Algorithm**: 1 parameter (success_rate_delta)

## Implementation Steps

### Phase 1: Folder Structure Setup

#### 1.1 Create Complete Folder Structure
```bash
mkdir -p hyperparameter-tuning/{config,src,scripts,results}
```

#### 1.2 Copy All Files from Current Branch
Copy the entire `hyperparameter-tuning/` folder to your new branch:

```
hyperparameter-tuning/
├── README.md
├── config/
│   ├── config.yaml                    # Euclidean algorithm config
│   └── success_rate_delta.yaml        # Delta-based algorithm config
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration management
│   ├── evaluation.py                  # Parameter evaluation logic
│   ├── json_updater.py                # JSON constants file updates
│   └── optimizer.py                   # Core Bayesian Optimization
├── scripts/
│   ├── run_optimization.py            # Main execution script
│   └── save_best_as_json.py           # Results export
└── results/
    └── best_params.json               # Best parameters output
```

### Phase 2: Dependencies Installation

#### 2.1 Update requirements.txt
Ensure your `requirements.txt` includes:
```txt
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
pandas>=1.3.0
pyyaml>=6.0
redis>=4.0.0
```

#### 2.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### Phase 3: Configuration Updates

#### 3.1 Update File Paths in config.yaml
Edit `hyperparameter-tuning/config/config.yaml`:
```yaml
experiment:
  name: "euclidean"
  output_csv: "optimization_results.csv"
  
  constants_file:
    path: "/path/to/your/sr_cost_routing_constants.json"  # UPDATE THIS PATH
    section: "sr_cost_routing_constants"
```

#### 3.2 Update File Paths in success_rate_delta.yaml
Edit `hyperparameter-tuning/config/success_rate_delta.yaml`:
```yaml
experiment:
  name: "delta_based"
  output_csv: "success_rate_delta_results.csv"
  
  constants_file:
    path: "/path/to/your/super_router_constants.json"  # UPDATE THIS PATH
    section: "sr_cost_routing_constants"
```

#### 3.3 Verify JSON Constants Files
Ensure both target JSON files exist and have the correct structure:
- `sr_cost_routing_constants.json` (for euclidean algorithm)
- `super_router_constants.json` (for delta-based algorithm)

### Phase 4: Integration Verification

#### 4.1 Verify Required Scripts
Ensure these scripts exist in your main `scripts/` folder:
- `multi_scene_runner.py`
- `results.py`
- `run_simulations.py`

#### 4.2 Verify Scene Structure
Ensure you have scene folders with:
- `scene-1/` through `scene-20/`
- Each scene contains `schema.yaml` and `transactions.csv`
- At least `scene-1/` should have `output_results.csv` for testing

#### 4.3 Verify Redis Connection
Ensure Redis is running and accessible:
```bash
redis-cli ping
# Should return: PONG
```

### Phase 5: Testing Implementation

#### 5.1 Test Delta-Based Algorithm (Recommended First)
```bash
cd hyperparameter-tuning
python scripts/run_optimization.py -c config/success_rate_delta.yaml --save-best
```

#### 5.2 Test Euclidean Algorithm
```bash
python scripts/run_optimization.py -c config/config.yaml --save-best
```

#### 5.3 Verify Outputs
Check for these outputs:
- `results/best_params.json`
- `results/success_rate_delta_results.csv` or `results/optimization_results.csv`

### Phase 6: Customization (Optional)

#### 6.1 Adjust Parameter Ranges
Edit the `range` sections in config files if needed:
```yaml
parameters:
  - name: "success_rate_delta"
    type: "float"
    precision: 2
    range:
      low: 0.01    # Adjust as needed
      high: 0.50   # Adjust as needed
```

#### 6.2 Adjust Trial Counts
Modify `trials` section in config files:
```yaml
trials:
  startup: 5      # Initial random trials
  total: 30       # Total trials (including startup)
  parallel_jobs: 2
```

#### 6.3 Adjust Objectives
Modify `objectives` section if needed:
```yaml
objectives:
  success_rate: "maximize"
  missed_savings: "minimize"
  
  aggregation:
    success_rate: "mean"
    missed_savings: "mean"
```

## Key Implementation Details

### Core Components

#### 1. Bayesian Optimizer (`src/optimizer.py`)
- **Gaussian Process**: Matérn kernel with ν=2.5
- **Normalization**: All parameters normalized to [0,1] range
- **Acquisition**: Expected Improvement with 10 restarts
- **Multi-objective**: Weighted combination of success rate and missed savings

#### 2. Evaluation System (`src/evaluation.py`)
- **Parameter Update**: Modifies JSON constants files
- **Multi-Scene Testing**: Runs `multi_scene_runner.py` with automated inputs
- **Metrics Collection**: Success rate and missed savings from scene results
- **Aggregation**: Mean across all scenes

#### 3. Configuration Management (`src/config.py`)
- **YAML Parsing**: Loads configuration from YAML files
- **Parameter Validation**: Validates parameter ranges and types
- **Precision Handling**: Manages decimal precision for different parameters

#### 4. JSON Updater (`src/json_updater.py`)
- **File Modification**: Updates target JSON constants files
- **Section Targeting**: Modifies specific sections in JSON
- **Backup Safety**: Creates backups before modifications

### Algorithm Behavior

#### 1. Initialization Phase (5 trials)
- Evaluates baseline parameters first
- Random sampling for initial GP training data
- Establishes performance baseline

#### 2. Optimization Loop (25 trials)
- GP model learns from previous evaluations
- EI function suggests next promising parameters
- Each evaluation runs multi-scene testing
- Continuously updates best parameters found

#### 3. Results Management
- Real-time saving of best parameters to JSON
- CSV export of all trial data
- Baseline vs optimized comparison
- Progress tracking and reporting

## Troubleshooting Guide

### Common Issues

#### 1. Redis Connection Errors
```bash
# Start Redis if not running
redis-server

# Check connection
redis-cli ping
```

#### 2. Missing Scene Files
```bash
# Verify scene structure
ls scene-1/
# Should show: schema.yaml, transactions.csv, output_results.csv (for scene-1)
```

#### 3. JSON Constants File Not Found
```bash
# Verify file paths in config files
cat hyperparameter-tuning/config/config.yaml | grep "path:"
cat hyperparameter-tuning/config/success_rate_delta.yaml | grep "path:"
```

#### 4. Python Module Import Errors
```bash
# Verify installation
pip list | grep -E "(numpy|scikit-learn|scipy|pandas|pyyaml|redis)"

# Reinstall if needed
pip install --upgrade numpy scikit-learn scipy pandas pyyaml redis
```

### Debug Mode

#### Enable Verbose Logging
Add to your command:
```bash
python scripts/run_optimization.py -c config/success_rate_delta.yaml --save-best 2>&1 | tee optimization.log
```

#### Test Single Scene
Modify `evaluation.py` to test with fewer scenes:
```python
# In evaluate_parameters_on_scenes function
num_scenes = 2  # Instead of 20
```

## Performance Optimization

### 1. Parallel Processing
- Increase `parallel_jobs` in config if you have multiple CPU cores
- Ensure Redis can handle concurrent connections

### 2. Scene Selection
- Start with 2-5 scenes for faster testing
- Gradually increase to full 20 scenes for production

### 3. Trial Optimization
- Reduce `total` trials for quick testing (15-20)
- Use 30+ trials for production optimization

## Expected Results

### Delta-Based Algorithm
- **Parameter**: `success_rate_delta` (0.01-0.50)
- **Target**: `super_router_constants.json`
- **Output**: Optimized success rate delta value

### Euclidean Algorithm
- **Parameters**: sr_threshold, spread_threshold, alpha, beta
- **Target**: `sr_cost_routing_constants.json`
- **Output**: Optimized 4-parameter combination

### Success Metrics
- **Success Rate Improvement**: % increase over baseline
- **Missed Savings Reduction**: $ decrease over baseline
- **Combined Score**: Weighted optimization objective

## Next Steps

1. **Copy the entire hyperparameter-tuning folder** to your new branch
2. **Update file paths** in both config files
3. **Install dependencies** from requirements.txt
4. **Test with delta-based algorithm** first (simpler, single parameter)
5. **Verify outputs** and integration with your existing system
6. **Run euclidean algorithm** for comprehensive optimization
7. **Monitor results** and adjust parameters as needed

This system will automatically find optimal parameters for both your euclidean and delta-based routing algorithms using sophisticated Bayesian Optimization techniques!
