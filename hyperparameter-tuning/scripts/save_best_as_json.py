#!/usr/bin/env python3
"""
Script to save best parameters from optimization results to JSON constants file.

This script reads optimization results and updates the target JSON constants file
with the best parameters found during optimization.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from json_updater import JSONUpdater


def load_optimization_results(results_path: str) -> Dict[str, Any]:
    """
    Load optimization results from JSON file.
    
    Args:
        results_path: Path to the optimization results JSON file
        
    Returns:
        Dictionary containing optimization results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_best_parameters(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract best parameters from optimization results.
    
    Args:
        results: Optimization results dictionary
        
    Returns:
        Dictionary of best parameters
    """
    # Check if best parameters are directly available
    if 'best' in results and 'parameters' in results['best']:
        return results['best']['parameters']
    
    # If not, look through optimization history for the best score
    if 'optimization_history' in results:
        best_trial = None
        best_score = -float('inf')
        
        for trial in results['optimization_history']:
            if trial.get('score', 0) > best_score:
                best_score = trial['score']
                best_trial = trial
        
        if best_trial and 'parameters' in best_trial:
            return best_trial['parameters']
    
    raise ValueError("Could not find best parameters in results")


def save_best_parameters(results_path: str, output_path: Optional[str] = None, 
                        backup: bool = True) -> None:
    """
    Save best parameters from optimization results to JSON constants file.
    
    Args:
        results_path: Path to the optimization results JSON file
        output_path: Output path for best parameters JSON (optional)
        backup: Whether to create backup of original constants file
    """
    # Load optimization results
    print(f"Loading optimization results from: {results_path}")
    results = load_optimization_results(results_path)
    
    # Extract best parameters
    print("Extracting best parameters...")
    best_params = extract_best_parameters(results)
    
    # Get constants file info from results
    if 'experiment' in results and 'config_file' in results:
        config_file = results['config_file']
        print(f"Using config file: {config_file}")
        
        # Load config to get constants file info
        try:
            from config import HyperparameterConfig
            config = HyperparameterConfig(config_file)
            
            constants_file = config.experiment.constants_file['path']
            constants_section = config.experiment.constants_file['section']
            
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using constants file info from results if available...")
            
            if 'constants_file' in results:
                constants_info = results['constants_file']
                constants_file = constants_info['path']
                constants_section = constants_info['section']
            else:
                raise ValueError("Could not determine constants file path and section")
    else:
        raise ValueError("Results file does not contain experiment information")
    
    print(f"Target constants file: {constants_file}")
    print(f"Target section: {constants_section}")
    
    # Initialize JSON updater
    updater = JSONUpdater(constants_file, constants_section)
    
    # Validate section structure
    if not updater.validate_section_structure():
        print(f"Warning: Section structure validation failed for {constants_file}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Update constants file with best parameters
    print(f"\nUpdating constants file with best parameters:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    
    try:
        updater.update_parameters(best_params, create_backup=backup)
        print(f"\nSuccessfully updated {constants_file}")
        
        if backup:
            print("Backup of original file created")
        
    except Exception as e:
        print(f"Error updating constants file: {e}")
        return
    
    # Save best parameters to separate JSON file if requested
    if output_path:
        output_data = {
            'experiment': results.get('experiment', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'source_results': results_path,
            'constants_file': constants_file,
            'constants_section': constants_section,
            'best_parameters': best_params,
            'best_score': results.get('best', {}).get('score', 'unknown'),
            'best_iteration': results.get('best', {}).get('iteration', 'unknown')
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Best parameters saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Experiment: {results.get('experiment', 'unknown')}")
    print(f"Best Score: {results.get('best', {}).get('score', 'unknown')}")
    print(f"Best Iteration: {results.get('best', {}).get('iteration', 'unknown')}")
    print(f"Constants File Updated: {constants_file}")
    print(f"Parameters Updated: {len(best_params)}")
    
    if backup:
        # List recent backups
        backups = updater.list_backups()
        if backups:
            print(f"Recent Backups: {len(backups)} files")
            print(f"Latest Backup: {os.path.basename(backups[0])}")


def list_available_results(results_dir: str = "results") -> None:
    """
    List available optimization results files.
    
    Args:
        results_dir: Directory containing results files
    """
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print(f"Available results files in {results_dir}:")
    
    results_files = []
    for file in os.listdir(results_dir):
        if file.endswith('_results.json'):
            results_files.append(file)
    
    if not results_files:
        print("No results files found")
        return
    
    results_files.sort()
    
    for i, file in enumerate(results_files, 1):
        file_path = os.path.join(results_dir, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            experiment = data.get('experiment', 'unknown')
            best_score = data.get('best', {}).get('score', 'unknown')
            timestamp = data.get('timestamp', 'unknown')
            
            print(f"{i:2d}. {file}")
            print(f"     Experiment: {experiment}")
            print(f"     Best Score: {best_score}")
            print(f"     Timestamp: {timestamp}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. {file} (Error reading: {e})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Save best parameters from optimization results to JSON constants file'
    )
    parser.add_argument('results_path', nargs='?',
                       help='Path to optimization results JSON file')
    parser.add_argument('-o', '--output',
                       help='Output path for best parameters JSON file')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup of original constants file')
    parser.add_argument('--list', action='store_true',
                       help='List available results files')
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing results files (for --list)')
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_available_results(args.results_dir)
        return
    
    # Validate results path
    if not args.results_path:
        print("Error: Results path is required")
        print("Use --list to see available results files")
        sys.exit(1)
    
    if not os.path.exists(args.results_path):
        print(f"Error: Results file not found: {args.results_path}")
        print("Use --list to see available results files")
        sys.exit(1)
    
    try:
        # Save best parameters
        save_best_parameters(
            results_path=args.results_path,
            output_path=args.output,
            backup=not args.no_backup
        )
        
        print(f"\nBest parameters successfully saved!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
