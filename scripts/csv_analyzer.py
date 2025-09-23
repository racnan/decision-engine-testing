import csv
import os
from collections import Counter
import ast

def analyze_generated_csv():
    """
    Reads the generated transactions.csv file and analyzes the distribution
    of the generated data to verify the generator's logic.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(project_root, "scene-1", "transactions.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: Generated CSV file not found at {csv_path}")
        print("INFO: Please run the csv_generator.py script first.")
        return

    total_rows = 0
    payment_type_counts = Counter()
    card_network_counts = Counter()
    card_type_counts = Counter()
    wallet_provider_counts = Counter()
    processor_success_counts = {}

    print(f"INFO: Analyzing generated data from {csv_path}...")

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            processor_names = [h.replace('_outcome', '') for h in reader.fieldnames if h.endswith('_outcome')]
            for name in processor_names:
                processor_success_counts[name] = 0
            
            if processor_names:
                print(f"INFO: Found processor outcome columns for: {processor_names}")

            for row in reader:
                total_rows += 1
                
                # Analyze payment method distribution
                pmt = row.get('payment_method_type')
                if pmt:
                    try:
                        pmt_val = ast.literal_eval(pmt)[0]
                        payment_type_counts[pmt_val] += 1
                    except (ValueError, SyntaxError):
                        payment_type_counts['Invalid Format'] += 1
                        continue

                    payment_method = row.get('payment_method')
                    if not payment_method:
                        continue

                    try:
                        method_list = ast.literal_eval(payment_method)
                    except (ValueError, SyntaxError):
                        continue

                    if pmt_val == 'CARD':
                        if len(method_list) == 1:
                            card_type_counts['Single Network'] += 1
                            card_network_counts[method_list[0]] += 1
                        elif len(method_list) > 1:
                            card_type_counts['Combination'] += 1
                            for network in method_list:
                                card_network_counts[network] += 1
                    elif pmt_val == 'WALLET':
                        if method_list:
                            wallet_provider_counts[method_list[0]] += 1
                
                # Analyze processor outcomes
                for processor_name in processor_names:
                    if row.get(f"{processor_name}_outcome") == 'success':
                        processor_success_counts[processor_name] += 1

    except (IOError, KeyError) as e:
        print(f"FATAL: An error occurred while reading the CSV file: {e}")
        return

    # --- Print the final report ---
    print("\n--- CSV Generation Analysis Report ---")
    print("-" * 36)

    if total_rows == 0:
        print("No transactions found in the CSV file.")
        return

    print(f"Total Transactions Generated: {total_rows}\n")

    print("Payment Method Type Distribution:")
    for pmt, count in payment_type_counts.items():
        percentage = (count / total_rows) * 100
        print(f"  - {pmt:<10}: {count} rows ({percentage:.2f}%)")

    if payment_type_counts['CARD'] > 0:
        print("\nCARD Transaction Analysis:")
        card_total = payment_type_counts['CARD']
        print("  Card Types:")
        for card_type, count in card_type_counts.items():
            percentage = (count / card_total) * 100
            print(f"    - {card_type:<18}: {count} rows ({percentage:.2f}%)")
        print("\n  Card Network Distribution:")
        for network, count in card_network_counts.items():
            print(f"    - {network:<18}: Appears in {count} rows")

    if payment_type_counts['WALLET'] > 0:
        print("\nWALLET Transaction Analysis:")
        for provider, count in wallet_provider_counts.items():
            percentage = (count / payment_type_counts['WALLET']) * 100
            print(f"  - {provider:<10}: {count} rows ({percentage:.2f}%)")

    print("\nTheoretical Processor Success Rates (from CSV):")
    if not processor_success_counts:
        print("  - No processor outcome columns found in CSV.")
    else:
        for processor, success_count in sorted(processor_success_counts.items()):
            success_rate = (success_count / total_rows) * 100 if total_rows > 0 else 0
            print(f"  - {processor:<10}: {success_rate:.2f}% ({success_count}/{total_rows} successes)")
            
    print("\n--- End of Report ---")

if __name__ == "__main__":
    analyze_generated_csv()
