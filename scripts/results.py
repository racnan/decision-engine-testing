import csv
import os
from collections import Counter

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

    processor_choices = Counter()
    network_choices = Counter()
    processor_performance = {}

    print(f"INFO: Analyzing results from {results_path}...")

    try:
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_transactions += 1
                
                # --- Overall & Processor Performance ---
                if row.get('final_outcome') == 'success':
                    successful_transactions += 1
                
                chosen_processor = row.get('chosen_processor')
                if chosen_processor and chosen_processor != 'None':
                    processor_choices[chosen_processor] += 1
                    if chosen_processor not in processor_performance:
                        processor_performance[chosen_processor] = {'success': 0, 'total': 0}
                    processor_performance[chosen_processor]['total'] += 1
                    if row.get('final_outcome') == 'success':
                        processor_performance[chosen_processor]['success'] += 1

                # --- Network & Savings Analysis ---
                chosen_network = row.get('chosen_network')
                if chosen_network and chosen_network != 'None':
                    network_choices[chosen_network] += 1

                try:
                    total_achieved_savings += float(row.get('savings', 0.0))
                    total_best_possible_savings += float(row.get('best_possible_savings', 0.0))
                except (ValueError, TypeError):
                    pass # Ignore rows with invalid savings data

                # --- Optimization Analysis ---
                if (row.get('chosen_processor') == row.get('best_possible_processor') and 
                    row.get('chosen_network') == row.get('best_possible_network')):
                    if row.get('chosen_processor'): # Ensure it wasn't a case where no choice was possible
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

    # --- Choice Distributions ---
    print(f"\nProcessor Choice Distribution:")
    if not processor_choices:
        print("  - No processors were chosen during the simulation.")
    else:
        for processor, count in processor_choices.most_common():
            percentage = (count / sum(processor_choices.values())) * 100
            print(f"  - {processor:<10}: {count} times ({percentage:.2f}%)")

    print(f"\nNetwork Choice Distribution:")
    if not network_choices:
        print("  - No networks were chosen during the simulation.")
    else:
        for network, count in network_choices.most_common():
            percentage = (count / sum(network_choices.values())) * 100
            print(f"  - {network:<10}: {count} times ({percentage:.2f}%)")

    # --- Savings & Optimization Analysis ---
    print(f"\nSavings and Optimization Analysis:")
    avg_achieved_savings = (total_achieved_savings / total_transactions) * 100 if total_transactions > 0 else 0
    avg_best_possible_savings = (total_best_possible_savings / total_transactions) * 100 if total_transactions > 0 else 0
    performance_score = (total_achieved_savings / total_best_possible_savings) * 100 if total_best_possible_savings > 0 else 0
    optimal_choice_perc = (optimal_choices / total_transactions) * 100 if total_transactions > 0 else 0

    print(f"  - Average Achieved Savings:    {avg_achieved_savings:.2f}%")
    print(f"  - Average Best Possible Savings: {avg_best_possible_savings:.2f}%")
    print(f"  - Routing Performance Score:     {performance_score:.2f}%")
    print(f"  - Optimal Choices Made:        {optimal_choices}/{total_transactions} ({optimal_choice_perc:.2f}%)")
            
    print("\n--- End of Report ---")


if __name__ == "__main__":
    analyze_results()
