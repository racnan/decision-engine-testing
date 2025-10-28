#!/usr/bin/env python3
"""
Multi-Scene Decision Engine Testing Automation

This script automates the entire testing workflow across multiple scenes:
1. Validates scene folders and transaction files
2. Collects user inputs for runs and algorithm
3. Manages Redis state between runs
4. Executes simulations for each scene/run combination
5. Generates comprehensive cross-scene reports

Usage:
    python3 scripts/multi_scene_runner.py
"""

import os
import sys
import subprocess
import shutil
import glob
from datetime import datetime
import redis
import csv
from collections import defaultdict
import json

# --- Configuration ---
REDIS_CONFIG = {
    'host': '127.0.0.1',
    'port': 6379,
    'socket_connect_timeout': 30,
    'socket_timeout': 10,
    'retry_on_timeout': True,
    'decode_responses': True
}

# Load management configuration
DELAY_BETWEEN_RUNS = 10    # 10 seconds between runs within a scene
DELAY_BETWEEN_SCENES = 30  # 30 seconds between different scenes

def setup_redis_connection():
    """Initialize Redis connection with service config"""
    try:
        client = redis.Redis(**REDIS_CONFIG)
        # Test connection
        client.ping()
        print("✓ Redis connection established")
        return client
    except redis.ConnectionError as e:
        print(f"✗ FATAL: Cannot connect to Redis at {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
        print("  - Is Redis running?")
        print("  - Is decision engine service running?")
        print("  - Check network connectivity")
        print(f"  Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ FATAL: Redis setup error: {e}")
        sys.exit(1)

def flush_redis_and_verify(redis_client, scene_name, run_number):
    """Flush Redis and verify success"""
    try:
        print(f"  → Flushing Redis for {scene_name} Run {run_number}...", end=" ")
        result = redis_client.flushall()
        if result:
            print("✓")
            return True
        else:
            print("✗")
            print(f"✗ FATAL: Redis flush failed for {scene_name} Run {run_number}")
            print("  - Previous run results may be contaminated")
            print("  - Stopping all simulations to ensure data integrity")
            sys.exit(1)
    except Exception as e:
        print("✗")
        print(f"✗ FATAL: Redis flush error for {scene_name} Run {run_number}: {e}")
        sys.exit(1)

def discover_scenes():
    """Discover all scene folders in current directory"""
    scene_folders = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('scene-'):
            try:
                # Extract scene number for sorting
                scene_num = int(item.split('-')[1])
                scene_folders.append((scene_num, item))
            except (ValueError, IndexError):
                continue
    
    # Sort by scene number
    scene_folders.sort(key=lambda x: x[0])
    return [folder for _, folder in scene_folders]

def validate_scene(scene_folder):
    """Validate that a scene has required files"""
    required_files = ['schema.yaml', 'transactions.csv']
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(scene_folder, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    return missing_files

def parse_scene_selection(input_str, available_scenes):
    """Parse user input for scene selection"""
    input_str = input_str.strip().lower()
    
    # Handle 'all' keyword
    if input_str == 'all':
        return available_scenes
    
    selected_scenes = []
    
    # Split by comma for multiple selections
    parts = [part.strip() for part in input_str.split(',')]
    
    for part in parts:
        if '-' in part and len(part.split('-')) == 2:
            # Handle range notation (e.g., "1-3")
            try:
                start, end = part.split('-')
                start_num = int(start)
                end_num = int(end)
                
                if start_num > end_num:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                
                for i in range(start_num, end_num + 1):
                    scene_name = f"scene-{i}"
                    if scene_name in available_scenes and scene_name not in selected_scenes:
                        selected_scenes.append(scene_name)
                        
            except ValueError as e:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Handle individual scene numbers
            try:
                scene_num = int(part)
                scene_name = f"scene-{scene_num}"
                if scene_name in available_scenes:
                    if scene_name not in selected_scenes:
                        selected_scenes.append(scene_name)
                else:
                    raise ValueError(f"Scene {scene_name} not found")
            except ValueError:
                raise ValueError(f"Invalid scene number: {part}")
    
    # Sort scenes by number for consistent ordering
    scene_numbers = []
    for scene in selected_scenes:
        try:
            num = int(scene.split('-')[1])
            scene_numbers.append((num, scene))
        except (ValueError, IndexError):
            scene_numbers.append((999, scene))  # Fallback for unusual names
    
    scene_numbers.sort()
    return [scene for _, scene in scene_numbers]

def get_user_inputs(available_scenes):
    """Collect user inputs for simulation parameters"""
    print(f"\nAvailable scenes: {', '.join(available_scenes)}")
    
    # Get scene selection with flexible input
    while True:
        print("\nScene Selection Options:")
        print("  • Specific scenes: 2,3 (comma-separated)")
        print("  • Range: 1-3 (inclusive range)")
        print("  • All scenes: all")
        print("  • Single scene: 2")
        
        try:
            scene_input = input("Enter scenes to run: ").strip()
            if not scene_input:
                print("Please enter a valid selection")
                continue
                
            selected_scenes = parse_scene_selection(scene_input, available_scenes)
            
            if not selected_scenes:
                print("No valid scenes selected. Please try again.")
                continue
                
            print(f"Selected scenes: {', '.join(selected_scenes)}")
            break
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try again with a valid format")
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Please try again")
    
    # Validate selected scenes
    print(f"\nValidating scenes...")
    invalid_scenes = []
    for scene in selected_scenes:
        missing_files = validate_scene(scene)
        if missing_files:
            print(f"✗ {scene}: Missing {', '.join(missing_files)}")
            invalid_scenes.append(scene)
        else:
            print(f"✓ {scene}: Valid transactions.csv found")
    
    if invalid_scenes:
        print(f"\n✗ FATAL: Invalid scenes found: {', '.join(invalid_scenes)}")
        print("Please ensure all scenes have schema.yaml and transactions.csv files")
        sys.exit(1)
    
    # Get number of runs per scene
    while True:
        try:
            num_runs = int(input("\nEnter number of runs for each scene: "))
            if num_runs >= 1:
                break
            else:
                print("Please enter a number >= 1")
        except ValueError:
            print("Please enter a valid number")
    
    # Get algorithm name
    algorithm = input("Enter the algorithm name: ").strip()
    if not algorithm:
        algorithm = "SUPER_ROUTER"  # Default
        print(f"Using default algorithm: {algorithm}")
    
    return selected_scenes, num_runs, algorithm

def create_run_directory(scene_folder, run_number):
    """Create and return path to run directory"""
    run_dir = os.path.join(scene_folder, f"run_{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def execute_simulation(scene_folder, run_dir, algorithm):
    """Execute simulation for a specific scene and run"""
    try:
        # Get absolute paths
        project_root = os.path.abspath('.')
        script_path = os.path.join(project_root, 'scripts', 'run_simulations.py')
        
        # Use virtual environment python if available, otherwise system python
        venv_python = os.path.join(project_root, '.venv', 'bin', 'python')
        python_exec = venv_python if os.path.exists(venv_python) else 'python3'
        
        # Execute simulation script with scene-specific parameters
        cmd = [
            python_exec, script_path,
            '--scene-path', scene_folder,
            '--output-dir', run_dir,
            '--algorithm', algorithm
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr or result.stdout
            
    except subprocess.TimeoutExpired:
        return False, "Simulation timed out (no timeout set - this should not happen)"
    except Exception as e:
        return False, str(e)

def generate_run_reports(scene_folder, run_dir):
    """Generate reports for a specific run"""
    try:
        project_root = os.path.abspath('.')
        script_path = os.path.join(project_root, 'scripts', 'results.py')
        
        # Use virtual environment python if available, otherwise system python
        venv_python = os.path.join(project_root, '.venv', 'bin', 'python')
        python_exec = venv_python if os.path.exists(venv_python) else 'python3'
        
        # Execute results script with run-specific parameters
        cmd = [
            python_exec, script_path,
            '--input-file', os.path.join(run_dir, 'output_results.csv'),
            '--output-dir', run_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
        
    except Exception as e:
        print(f"Warning: Failed to generate reports for {run_dir}: {e}")
        return False

def collect_run_results(scene_folder, num_runs):
    """Collect and aggregate results from all runs in a scene"""
    scene_results = {
        'scene_name': scene_folder,
        'total_runs': num_runs,
        'successful_runs': 0,
        'failed_runs': 0,
        'aggregate_stats': {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'total_savings': 0.0,
            'best_possible_savings': 0.0
        },
        'run_details': []
    }
    
    for run_num in range(1, num_runs + 1):
        run_dir = os.path.join(scene_folder, f"run_{run_num}")
        results_file = os.path.join(run_dir, 'output_results.csv')
        
        if os.path.exists(results_file):
            scene_results['successful_runs'] += 1
            run_stats = analyze_run_results(results_file)
            scene_results['run_details'].append({
                'run_number': run_num,
                'status': 'success',
                'stats': run_stats
            })
            
            # Aggregate stats
            agg = scene_results['aggregate_stats']
            agg['total_transactions'] += run_stats.get('total_transactions', 0)
            agg['successful_transactions'] += run_stats.get('successful_transactions', 0)
            agg['failed_transactions'] += run_stats.get('failed_transactions', 0)
            agg['total_savings'] += run_stats.get('achieved_savings', 0.0)
            agg['best_possible_savings'] += run_stats.get('best_possible_savings', 0.0)
        else:
            scene_results['failed_runs'] += 1
            scene_results['run_details'].append({
                'run_number': run_num,
                'status': 'failed',
                'stats': {}
            })
    
    return scene_results

def analyze_run_results(results_file):
    """Analyze results from a single run"""
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

def print_cross_scene_summary(all_scene_results, algorithm):
    """Print comprehensive summary across all scenes"""
    print("\n" + "="*80)
    print("                    MULTI-SCENE SIMULATION SUMMARY")
    print("="*80)
    print(f"Algorithm: {algorithm}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Scenes: {len(all_scene_results)}")
    
    # Overall statistics
    total_runs = sum(scene['total_runs'] for scene in all_scene_results)
    successful_runs = sum(scene['successful_runs'] for scene in all_scene_results)
    total_transactions = sum(scene['aggregate_stats']['total_transactions'] for scene in all_scene_results)
    total_successful_transactions = sum(scene['aggregate_stats']['successful_transactions'] for scene in all_scene_results)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Runs Executed: {successful_runs}/{total_runs}")
    print(f"  Total Transactions: {total_transactions:,}")
    print(f"  Successful Transactions: {total_successful_transactions:,}")
    if total_transactions > 0:
        overall_success_rate = (total_successful_transactions / total_transactions) * 100
        print(f"  Overall Success Rate: {overall_success_rate:.2f}%")
    
    # Per-scene breakdown
    print(f"\nPer-Scene Performance:")
    print("-" * 80)
    
    for scene_result in all_scene_results:
        scene_name = scene_result['scene_name']
        stats = scene_result['aggregate_stats']
        successful_runs = scene_result['successful_runs']
        total_runs = scene_result['total_runs']
        
        print(f"\n{scene_name.upper()}:")
        print(f"  Runs: {successful_runs}/{total_runs}")
        
        if stats['total_transactions'] > 0:
            scene_success_rate = (stats['successful_transactions'] / stats['total_transactions']) * 100
            print(f"  Transactions: {stats['successful_transactions']:,}/{stats['total_transactions']:,} ({scene_success_rate:.2f}%)")
            print(f"  Savings: ${stats['total_savings']:.2f} / ${stats['best_possible_savings']:.2f}")
            
            if stats['best_possible_savings'] > 0:
                efficiency = (stats['total_savings'] / stats['best_possible_savings']) * 100
                print(f"  Efficiency: {efficiency:.2f}%")
    
    # Best performing scene
    if all_scene_results:
        best_scene = max(all_scene_results, 
                        key=lambda x: (x['aggregate_stats']['successful_transactions'] / 
                                     max(x['aggregate_stats']['total_transactions'], 1)))
        best_scene_stats = best_scene['aggregate_stats']
        if best_scene_stats['total_transactions'] > 0:
            best_success_rate = (best_scene_stats['successful_transactions'] / 
                               best_scene_stats['total_transactions']) * 100
            print(f"\nBest Performing Scene: {best_scene['scene_name']} ({best_success_rate:.2f}% success rate)")
    
    print("\n" + "="*80)
    print("                              END OF SUMMARY")
    print("="*80)

def main():
    """Main execution function"""
    print("="*60)
    print("           MULTI-SCENE DECISION ENGINE TESTER")
    print("="*60)
    
    # Check current directory
    if not os.path.exists('scripts'):
        print("✗ FATAL: Please run this script from the project root directory")
        print("  Current directory should contain 'scripts/' folder")
        sys.exit(1)
    
    # Discover and validate scenes
    available_scenes = discover_scenes()
    if not available_scenes:
        print("✗ FATAL: No scene folders found")
        print("  Expected scene folders: scene-1, scene-2, etc.")
        sys.exit(1)
    
    # Setup Redis connection
    redis_client = setup_redis_connection()
    
    # Get user inputs
    selected_scenes, num_runs, algorithm = get_user_inputs(available_scenes)
    
    print(f"\n{'='*60}")
    print("STARTING SIMULATIONS")
    print(f"{'='*60}")
    print(f"Scenes: {', '.join(selected_scenes)}")
    print(f"Runs per scene: {num_runs}")
    print(f"Algorithm: {algorithm}")
    print(f"Total runs: {len(selected_scenes) * num_runs}")
    
    # Execute simulations
    all_scene_results = []
    
    for scene_idx, scene_folder in enumerate(selected_scenes, 1):
        print(f"\n[Scene {scene_idx}/{len(selected_scenes)}] {scene_folder.upper()}")
        print("-" * 40)
        
        for run_num in range(1, num_runs + 1):
            print(f"  Run {run_num}/{num_runs}:")
            
            # Flush Redis
            flush_redis_and_verify(redis_client, scene_folder, run_num)
            
            # Create run directory
            run_dir = create_run_directory(scene_folder, run_num)
            print(f"  → Created directory: {run_dir}")
            
            # Execute simulation
            print(f"  → Running simulation...", end=" ")
            success, error = execute_simulation(scene_folder, run_dir, algorithm)
            
            if success:
                print("✓")
                print(f"  → Generating reports...", end=" ")
                generate_run_reports(scene_folder, run_dir)
                print("✓")
                print(f"  → Run {run_num} COMPLETE")
                
                # Add delay between runs (except after the last run)
                if run_num < num_runs:
                    print(f"  → Adding {DELAY_BETWEEN_RUNS}s delay before next run...", end=" ")
                    import time
                    time.sleep(DELAY_BETWEEN_RUNS)
                    print("✓")
            else:
                print("✗")
                print(f"  → Error: {error}")
                print(f"  → Run {run_num} FAILED")
                print("✗ FATAL: Terminating all simulations due to run failure")
                print("  Please fix the issue and restart the simulation")
                sys.exit(1)
        
        # Collect scene results
        scene_results = collect_run_results(scene_folder, num_runs)
        all_scene_results.append(scene_results)
        
        scene_success = scene_results['successful_runs']
        print(f"  Scene Summary: {scene_success}/{num_runs} runs successful")
        
        # Add delay between scenes (except after the last scene)
        if scene_idx < len(selected_scenes):
            print(f"\n🔄 Adding {DELAY_BETWEEN_SCENES}s delay before next scene to allow service recovery...", end=" ")
            import time
            time.sleep(DELAY_BETWEEN_SCENES)
            print("✓")
    
    # Print final summary
    print_cross_scene_summary(all_scene_results, algorithm)
    
    print(f"\n✓ All simulations completed!")
    print(f"✓ Results stored in respective run folders")
    print(f"✓ Individual reports generated for each run")
    
    # Generate consolidated summary report
    generate_consolidated_summary(selected_scenes, algorithm, num_runs)

def generate_consolidated_summary(scenes, algorithm, num_runs):
    """Generate consolidated summary report using the summary report generator"""
    try:
        print(f"\n{'='*60}")
        print("GENERATING CONSOLIDATED SUMMARY REPORT")
        print(f"{'='*60}")
        
        # Get absolute paths
        project_root = os.path.abspath('.')
        script_path = os.path.join(project_root, 'scripts', 'generate_summary_report.py')
        
        # Use virtual environment python if available, otherwise system python
        venv_python = os.path.join(project_root, '.venv', 'bin', 'python')
        python_exec = venv_python if os.path.exists(venv_python) else 'python3'
        
        # Prepare scenes parameter
        scenes_param = ','.join(scenes)
        
        # Execute summary report generator
        cmd = [
            python_exec, script_path,
            '--scenes', scenes_param,
            '--algorithm', algorithm,
            '--runs', str(num_runs)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Consolidated summary report generated successfully!")
            print("✓ Check summary_report.csv for cross-run analysis")
            # Print any output from the summary generator
            if result.stdout.strip():
                print("\nSummary Generator Output:")
                print(result.stdout.strip())
        else:
            print("✗ Warning: Summary report generation failed")
            if result.stderr:
                print(f"Error details: {result.stderr}")
                
    except subprocess.TimeoutExpired:
        print("✗ Warning: Summary report generation timed out")
    except Exception as e:
        print(f"✗ Warning: Could not generate summary report: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ FATAL: Unexpected error: {e}")
        sys.exit(1)
