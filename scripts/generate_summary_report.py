#!/usr/bin/env python3
"""
Summary Report Generator for Decision Engine Testing

This script generates a consolidated summary report across all scenes, algorithms, and runs.
It uses upsert logic to preserve existing data while updating specific scene/algorithm combinations.

Usage:
    python3 scripts/generate_summary_report.py --scenes scene-1,scene-2 --algorithm SUPER_ROUTER --runs 3
    python3 scripts/generate_summary_report.py --all  # Process all existing data
"""

import csv
import os
import sys
import argparse
from collections import defaultdict

def analyze_run_results(results_file):
    """
    Analyze results from a single run using the same logic as results.py
    Returns: dict with total_transactions, successful_transactions, achieved_savings, best_possible_savings
    """
    stats = {
        'total_transactions': 0,
        'successful_transactions': 0,
        'failed_transactions': 0,
        'achieved_savings': 0.0,
        'best_possible_savings': 0.0,
        'optimal_choices': 0
    }
    
    try:
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats['total_transactions'] += 1
                
                if row.get('final_outcome') == 'success':
                    stats['successful_transactions'] += 1
                    try:
                        stats['achieved_savings'] += float(row.get('savings', 0.0))
                    except (ValueError, TypeError):
                        pass
                else:
                    stats['failed_transactions'] += 1
                
                try:
                    stats['best_possible_savings'] += float(row.get('best_possible_savings', 0.0))
                except (ValueError, TypeError):
                    pass
                
                # Check if optimal choice was made
                is_optimal = (row.get('chosen_processor') == row.get('best_possible_processor') and 
                             row.get('chosen_network') == row.get('best_possible_network'))
                if is_optimal and row.get('chosen_processor'):
                    stats['optimal_choices'] += 1
    
    except Exception as e:
        print(f"Warning: Could not analyze {results_file}: {e}")
    
    return stats

def map_scene_to_schema(scene_name):
    """Map scene folder names to schema names"""
    mapping = {
        'scene-1': 'Schema 1',
        'scene-2': 'Schema 2', 
        'scene-3': 'Schema 3',
        'scene-4': 'Schema 4',
        'scene-5': 'Schema 5'
    }
    return mapping.get(scene_name, scene_name)

def discover_all_runs():
    """Discover all existing scene/run combinations"""
    runs_data = []
    
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('scene-'):
            scene_folder = item
            scene_path = os.path.join('.', scene_folder)
            
            # Look for run folders
            for run_item in os.listdir(scene_path):
                run_path = os.path.join(scene_path, run_item)
                if os.path.isdir(run_path) and run_item.startswith('run_'):
                    results_file = os.path.join(run_path, 'output_results.csv')
                    if os.path.exists(results_file):
                        try:
                            run_number = int(run_item.split('_')[1])
                            runs_data.append({
                                'scene': scene_folder,
                                'run_number': run_number,
                                'results_file': results_file
                            })
                        except (ValueError, IndexError):
                            continue
    
    return runs_data

def collect_scene_data(scenes, algorithm, num_runs=None):
    """
    Collect data for specific scenes and algorithm
    If num_runs is specified, only process that many runs (for current execution)
    If num_runs is None, process all available runs (for --all mode)
    """
    scene_data = {}
    
    for scene in scenes:
        schema_name = map_scene_to_schema(scene)
        scene_data[schema_name] = []
        
        if num_runs:
            # Process specific number of runs (current execution)
            run_numbers = range(1, num_runs + 1)
        else:
            # Process all available runs (--all mode)
            run_numbers = []
            scene_path = os.path.join('.', scene)
            if os.path.exists(scene_path):
                for item in os.listdir(scene_path):
                    if os.path.isdir(os.path.join(scene_path, item)) and item.startswith('run_'):
                        try:
                            run_num = int(item.split('_')[1])
                            run_numbers.append(run_num)
                        except (ValueError, IndexError):
                            continue
            run_numbers = sorted(run_numbers)
        
        for run_num in run_numbers:
            results_file = os.path.join('.', scene, f'run_{run_num}', 'output_results.csv')
            if os.path.exists(results_file):
                stats = analyze_run_results(results_file)
                
                # Calculate success rate
                if stats['total_transactions'] > 0:
                    success_rate = (stats['successful_transactions'] / stats['total_transactions']) * 100
                else:
                    success_rate = 0.0
                
                run_data = {
                    'schema': schema_name,
                    'algorithm': algorithm,
                    'run': f'Run {run_num}',
                    'success_rate': round(success_rate, 2),
                    'savings': round(stats['achieved_savings'], 2),
                    'best_possible_savings': round(stats['best_possible_savings'], 2)
                }
                
                scene_data[schema_name].append(run_data)
    
    return scene_data

def calculate_averages(scene_data):
    """Calculate average rows for each schema/algorithm combination"""
    averages = []
    
    for schema_name, runs in scene_data.items():
        if not runs:
            continue
            
        # Group by algorithm
        by_algorithm = defaultdict(list)
        for run in runs:
            by_algorithm[run['algorithm']].append(run)
        
        for algorithm, algorithm_runs in by_algorithm.items():
            if len(algorithm_runs) > 1:  # Only add average if multiple runs
                avg_sr = sum(r['success_rate'] for r in algorithm_runs) / len(algorithm_runs)
                avg_savings = sum(r['savings'] for r in algorithm_runs) / len(algorithm_runs)
                avg_best_possible = sum(r['best_possible_savings'] for r in algorithm_runs) / len(algorithm_runs)
                
                averages.append({
                    'schema': schema_name,
                    'algorithm': algorithm,
                    'run': 'Avg.',
                    'success_rate': round(avg_sr, 8),  # Higher precision for averages
                    'savings': round(avg_savings, 2),
                    'best_possible_savings': round(avg_best_possible, 2)
                })
    
    return averages

def load_existing_summary():
    """Load existing summary report if it exists"""
    summary_file = 'summary_report.csv'
    existing_data = []
    
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_data.append({
                        'schema': row['Schema'],
                        'algorithm': row['Algorithm'], 
                        'run': row['Run'],
                        'success_rate': float(row['SR']),
                        'savings': float(row['Savings']),
                        'best_possible_savings': float(row['Best Possible Savings'])
                    })
        except Exception as e:
            print(f"Warning: Could not load existing summary: {e}")
    
    return existing_data

def upsert_summary_data(new_data, existing_data, target_schemas, target_algorithm):
    """
    Upsert logic: Remove existing entries for target schema/algorithm combinations,
    then add new data while preserving other entries
    """
    # Filter out existing entries for target schema/algorithm combinations
    filtered_data = []
    for entry in existing_data:
        if entry['schema'] in target_schemas and entry['algorithm'] == target_algorithm:
            continue  # Remove existing entries for this schema/algorithm
        filtered_data.append(entry)
    
    # Add new data
    all_new_rows = []
    for schema_name, runs in new_data.items():
        all_new_rows.extend(runs)
    
    # Add averages
    averages = calculate_averages(new_data)
    all_new_rows.extend(averages)
    
    # Combine filtered existing data with new data
    final_data = filtered_data + all_new_rows
    
    # Sort by schema, algorithm, then run (with "Avg." at the end for each group)
    def sort_key(item):
        schema_order = item['schema']
        algorithm_order = item['algorithm']
        run_order = item['run']
        
        # Put "Avg." at the end for each group
        if run_order == 'Avg.':
            run_sort = 999
        else:
            try:
                run_sort = int(run_order.split()[-1])
            except:
                run_sort = 0
                
        return (schema_order, algorithm_order, run_sort)
    
    final_data.sort(key=sort_key)
    return final_data

def save_summary_report(data):
    """Save the summary report to CSV"""
    summary_file = 'summary_report.csv'
    
    fieldnames = ['Schema', 'Algorithm', 'Run', 'SR', 'Savings', 'Best Possible Savings']
    
    try:
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in data:
                writer.writerow({
                    'Schema': row['schema'],
                    'Algorithm': row['algorithm'],
                    'Run': row['run'],
                    'SR': row['success_rate'],
                    'Savings': row['savings'],
                    'Best Possible Savings': row['best_possible_savings']
                })
        
        print(f"✓ Summary report saved to: {summary_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error saving summary report: {e}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate consolidated summary report')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scenes', type=str,
                      help='Comma-separated list of scenes (e.g., scene-1,scene-2)')
    group.add_argument('--all', action='store_true',
                      help='Process all existing scenes and algorithms')
    
    parser.add_argument('--algorithm', type=str,
                       help='Algorithm name (required when using --scenes)')
    parser.add_argument('--runs', type=int,
                       help='Number of runs per scene (if not specified, processes all available runs)')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    if args.scenes and not args.algorithm:
        print("Error: --algorithm is required when using --scenes")
        sys.exit(1)
    
    print("="*60)
    print("           SUMMARY REPORT GENERATOR")
    print("="*60)
    
    # Load existing summary data
    existing_data = load_existing_summary()
    print(f"Loaded {len(existing_data)} existing entries")
    
    if args.all:
        # Process all existing data - rebuild entire summary
        print("Processing all existing scenes and runs...")
        all_runs = discover_all_runs()
        
        # Group by scene and algorithm
        by_scene_algo = defaultdict(lambda: defaultdict(list))
        for run_info in all_runs:
            # Try to infer algorithm from existing data or use "Unknown"
            schema = map_scene_to_schema(run_info['scene'])
            # For --all mode, we need to infer algorithm somehow
            # This is a limitation - in practice, algorithm should be stored with each run
            algorithm = "Unknown"  # Default fallback
            
            stats = analyze_run_results(run_info['results_file'])
            if stats['total_transactions'] > 0:
                success_rate = (stats['successful_transactions'] / stats['total_transactions']) * 100
            else:
                success_rate = 0.0
            
            run_data = {
                'schema': schema,
                'algorithm': algorithm,
                'run': f'Run {run_info["run_number"]}',
                'success_rate': round(success_rate, 2),
                'savings': round(stats['achieved_savings'], 2),
                'best_possible_savings': round(stats['best_possible_savings'], 2)
            }
            
            by_scene_algo[schema][algorithm].append(run_data)
        
        # Convert to expected format
        scene_data = {}
        for schema, algorithms in by_scene_algo.items():
            scene_data[schema] = []
            for algorithm, runs in algorithms.items():
                scene_data[schema].extend(runs)
        
        # For --all mode, replace everything
        final_data = []
        for schema_name, runs in scene_data.items():
            final_data.extend(runs)
        
        averages = calculate_averages(scene_data)
        final_data.extend(averages)
        
    else:
        # Process specific scenes and algorithm
        scenes = [s.strip() for s in args.scenes.split(',')]
        algorithm = args.algorithm
        runs = args.runs
        
        print(f"Processing scenes: {', '.join(scenes)}")
        print(f"Algorithm: {algorithm}")
        print(f"Runs: {runs if runs else 'all available'}")
        
        # Collect new data
        scene_data = collect_scene_data(scenes, algorithm, runs)
        target_schemas = [map_scene_to_schema(scene) for scene in scenes]
        
        # Upsert with existing data
        final_data = upsert_summary_data(scene_data, existing_data, target_schemas, algorithm)
    
    # Save the summary report
    if save_summary_report(final_data):
        print(f"✓ Summary report updated with {len(final_data)} total entries")
    else:
        print("✗ Failed to save summary report")
        sys.exit(1)

if __name__ == "__main__":
    main()
