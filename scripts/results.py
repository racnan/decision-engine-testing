import csv
import os
from collections import Counter

def analyze_results():
    """
    Reads the simulation results from output_results.csv, analyzes them,
    and prints a summary report.
    """
    # Define the path to the results file.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_path = os.path.join(project_root, "scene-1", "output_results.csv")

    if not os.path.exists(results_path):
        print(f"ERROR: Results file not found at {results_path}")
        print("INFO: Please run the run_simulations.py script first to generate results.")
        return

    # Initialize variables to store our metrics.
    total_transactions = 0
    successful_transactions = 0
    
    # Use a Counter to easily count how many times each processor was chosen.
    processor_choices = Counter()
    
    # Use a dictionary to store success/total counts for each processor.
    # e.g., {'Stripe': {'success': 100, 'total': 120}}
    processor_performance = {}

    print(f"INFO: Analyzing results from {results_path}...")

    try:
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_transactions += 1
                
                # --- Overall Performance ---
                if row['final_outcome'] == 'success':
                    successful_transactions += 1
                
                # --- Processor-specific analysis ---
                chosen_processor = row['chosen_processor']
                if chosen_processor and chosen_processor != 'None':
                    # Increment the count for the chosen processor.
                    processor_choices[chosen_processor] += 1
                    
                    # Initialize the dictionary for the processor if it's the first time we see it.
                    if chosen_processor not in processor_performance:
                        processor_performance[chosen_processor] = {'success': 0, 'total': 0}
                    
                    # Increment the total attempts for this processor.
                    processor_performance[chosen_processor]['total'] += 1
                    
                    # If the outcome was a success, increment its success count.
                    if row['final_outcome'] == 'success':
                        processor_performance[chosen_processor]['success'] += 1

    except (IOError, KeyError) as e:
        print(f"FATAL: An error occurred while reading the results file: {e}")
        return

    # --- Print the final report ---
    print("\n--- Simulation Analysis Report ---")
    print("-" * 32)

    if total_transactions == 0:
        print("No transactions found in the results file.")
        return

    # Overall summary
    overall_success_rate = (successful_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    print(f"\nOverall Performance:")
    print(f"  - Total Transactions Simulated: {total_transactions}")
    print(f"  - Successful Transactions:      {successful_transactions}")
    print(f"  - Failed Transactions:          {total_transactions - successful_transactions}")
    print(f"  - Overall Success Rate:         {overall_success_rate:.2f}%")

    # Processor choice distribution
    print(f"\nProcessor Choice Distribution:")
    if not processor_choices:
        print("  - No processors were chosen during the simulation.")
    else:
        # .most_common() gives us a sorted list of (processor, count) tuples.
        for processor, count in processor_choices.most_common():
            percentage = (count / total_transactions) * 100
            print(f"  - {processor:<10}: {count} times ({percentage:.2f}%)")

    # Per-processor performance
    print(f"\nPerformance by Chosen Processor:")
    if not processor_performance:
        print("  - No data available.")
    else:
        for processor, data in sorted(processor_performance.items()):
            success_rate = (data['success'] / data['total']) * 100 if data['total'] > 0 else 0
            print(f"  - {processor}:")
            print(f"    - Chosen {data['total']} times.")
            print(f"    - Succeeded {data['success']} times.")
            print(f"    - Success Rate: {success_rate:.2f}%")
            
    print("\n--- End of Report ---")


if __name__ == "__main__":
    analyze_results()
