import csv
import os
from collections import Counter, defaultdict

def analyze_results():
    """
    Reads the simulation results from output_results.csv, analyzes them,
    and prints a summary report including savings and optimization analysis.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_path = os.path.join(project_root, "scene-1", "output_results.csv")

    if not os.path.exists(results_path):
        print(f"ERROR: Results file not found at {results_path}")
        print("INFO: Please run the run_simulations.py script first to generate results.")
        return

    # --- Initialize Metrics ---
    total_transactions = 0
    successful_transactions = 0
    
    total_achieved_savings = 0.0
    total_best_possible_savings = 0.0
    optimal_choices = 0

    # Use defaultdict to simplify initialization
    processor_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})
    network_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})

    print(f"INFO: Analyzing results from {results_path}...")

    try:
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_transactions += 1
                is_success = row.get('final_outcome') == 'success'
                
                if is_success:
                    successful_transactions += 1
                
                # --- Processor & Network Analysis ---
                chosen_processor = row.get('chosen_processor')
                if chosen_processor and chosen_processor != 'None':
                    processor_stats[chosen_processor]['total'] += 1
                    if is_success:
                        processor_stats[chosen_processor]['success'] += 1
                    else:
                        processor_stats[chosen_processor]['fail'] += 1

                chosen_network = row.get('chosen_network')
                if chosen_network and chosen_network != 'None':
                    network_stats[chosen_network]['total'] += 1
                    if is_success:
                        network_stats[chosen_network]['success'] += 1
                    else:
                        network_stats[chosen_network]['fail'] += 1

                # --- Savings & Optimization Analysis ---
                try:
                    total_achieved_savings += float(row.get('savings', 0.0))
                    total_best_possible_savings += float(row.get('best_possible_savings', 0.0))
                except (ValueError, TypeError):
                    pass # Ignore rows with invalid savings data

                is_optimal = (row.get('chosen_processor') == row.get('best_possible_processor') and 
                              row.get('chosen_network') == row.get('best_possible_network'))
                
                if is_optimal and row.get('chosen_processor'):
                    optimal_choices += 1

    except (IOError, KeyError) as e:
        print(f"FATAL: An error occurred while reading the results file: {e}")
        return

    # --- Print the Final Report ---
    print("\n--- Simulation Analysis Report ---")
    print("-" * 34)

    if total_transactions == 0:
        print("No transactions found in the results file.")
        return

    # --- Overall Performance ---
    overall_success_rate = (successful_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    print(f"\nOverall Performance:")
    print(f"  - Total Transactions Simulated: {total_transactions}")
    print(f"  - Successful Transactions:      {successful_transactions}")
    print(f"  - Failed Transactions:          {total_transactions - successful_transactions}")
    print(f"  - Overall Success Rate:         {overall_success_rate:.2f}%")

    # --- Processor Choice Distribution ---
    print(f"\nProcessor Choice Distribution:")
    if not processor_stats:
        print("  - No processors were chosen during the simulation.")
    else:
        # Sort processors by the total number of times they were chosen
        sorted_processors = sorted(processor_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        total_choices = sum(p['total'] for p in processor_stats.values())
        
        for processor, stats in sorted_processors:
            percentage = (stats['total'] / total_choices) * 100 if total_choices > 0 else 0
            print(f"  - {processor:<10}: {stats['total']} times ({percentage:.2f}%)")
            print(f"    - Success: {stats['success']}")
            print(f"    - Fail:    {stats['fail']}")

    # --- Network Choice Distribution ---
    print(f"\nNetwork Choice Distribution:")
    if not network_stats:
        print("  - No networks were chosen during the simulation.")
    else:
        sorted_networks = sorted(network_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        total_network_choices = sum(n['total'] for n in network_stats.values())

        for network, stats in sorted_networks:
            percentage = (stats['total'] / total_network_choices) * 100 if total_network_choices > 0 else 0
            print(f"  - {network:<10}: {stats['total']} times ({percentage:.2f}%)")
            print(f"    - Success: {stats['success']}")
            print(f"    - Fail:    {stats['fail']}")

    # --- Savings & Optimization Analysis ---
    print(f"\nSavings and Optimization Analysis:")
    missed_savings = total_best_possible_savings - total_achieved_savings
    
    print(f"  - Transactions with Best Option Selected:   {optimal_choices}")
    print(f"  - Transactions with Sub-Optimal Selected: {total_transactions - optimal_choices}")
    print(f"  - Total Potential Savings Missed:         ${missed_savings:,.2f}")
            
    print("\n--- End of Report ---")


if __name__ == "__main__":
    analyze_results()
