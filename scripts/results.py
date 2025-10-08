import csv
import os
import argparse
import sys
from collections import Counter, defaultdict

def analyze_results(input_file=None, output_dir=None):
    """
    Reads the simulation results from output_results.csv, analyzes them,
    and prints a summary report including savings and optimization analysis.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if input_file:
        results_path = input_file
    else:
        results_path = os.path.join(project_root, "scene-1", "output_results.csv")
    
    if output_dir:
        output_directory = output_dir
    else:
        output_directory = os.path.join(project_root, "scene-1")

    if not os.path.exists(results_path):
        print(f"ERROR: Results file not found at {results_path}")
        print("INFO: Please run the run_simulations.py script first to generate results.")
        return

    # --- Initialize Metrics ---
    total_transactions = 0
    successful_transactions = 0
    
    optimal_choices = 0

    # Detailed savings metrics
    achieved_savings_success = 0.0
    best_possible_savings_success = 0.0
    best_possible_savings_fail = 0.0

    # Use defaultdict to simplify initialization
    processor_stats = defaultdict(lambda: {
        'total': 0, 'success': 0, 'fail': 0,
        'networks': defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})
    })
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
                chosen_network = row.get('chosen_network')

                if chosen_processor and chosen_processor != 'None':
                    stats = processor_stats[chosen_processor]
                    stats['total'] += 1
                    if is_success:
                        stats['success'] += 1
                    else:
                        stats['fail'] += 1

                    if chosen_network and chosen_network != 'None':
                        network_stats_ptr = stats['networks'][chosen_network]
                        network_stats_ptr['total'] += 1
                        if is_success:
                            network_stats_ptr['success'] += 1
                        else:
                            network_stats_ptr['fail'] += 1
                
                if chosen_network and chosen_network != 'None':
                    network_stats[chosen_network]['total'] += 1
                    if is_success:
                        network_stats[chosen_network]['success'] += 1
                    else:
                        network_stats[chosen_network]['fail'] += 1

                # --- Savings & Optimization Analysis ---
                try:
                    best_possible = float(row.get('best_possible_savings', 0.0))
                    if is_success:
                        achieved_savings_success += float(row.get('savings', 0.0))
                        best_possible_savings_success += best_possible
                    else:
                        # For failed transactions, the opportunity cost is the best possible savings.
                        best_possible_savings_fail += best_possible
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
    print("\nOverall Performance:")
    overall_success_rate = (successful_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    print(f"    # The total number of transactions that were processed from the input file.")
    print(f"  - Total Transactions Simulated: {total_transactions}")
    print(f"    # The count of transactions that ended with a 'success' outcome.")
    print(f"  - Successful Transactions:      {successful_transactions}")
    print(f"    # The count of transactions that ended with a 'fail' outcome.")
    print(f"  - Failed Transactions:          {total_transactions - successful_transactions}")
    print(f"    # The percentage of successful transactions out of the total.")
    print(f"  - Overall Success Rate:         {overall_success_rate:.2f}%")

    # --- Connector-Network Distribution ---
    print("\nConnector-Network Distribution:")
    print("    # Breaks down processor usage and the networks chosen for each.")
    if not processor_stats:
        print("  - No processors were chosen during the simulation.")
    else:
        sorted_processors = sorted(processor_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        total_choices = sum(p['total'] for p in processor_stats.values())

        for processor, stats in sorted_processors:
            percentage = (stats['total'] / total_choices) * 100 if total_choices > 0 else 0
            print(f"\n  - {processor:<10}: {stats['total']} times ({percentage:.2f}%)")
            print(f"    - Success: {stats['success']}, Fail: {stats['fail']}")

            if stats['networks']:
                print("    - Network Distribution:")
                sorted_networks = sorted(stats['networks'].items(), key=lambda item: item[1]['total'], reverse=True)
                for network, net_stats in sorted_networks:
                    net_percentage = (net_stats['total'] / stats['total']) * 100 if stats['total'] > 0 else 0
                    print(f"      - {network:<10}: {net_stats['total']} times ({net_percentage:.2f}%)")
                    print(f"        - Success: {net_stats['success']}, Fail: {net_stats['fail']}")

    # --- Network Choice Distribution ---
    print("\nNetwork Choice Distribution:")
    print("    # Breaks down how many times each network was chosen and its performance.")
    if not network_stats:
        print("  - No networks were chosen during the simulation.")
    else:
        sorted_networks = sorted(network_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        total_network_choices = sum(n['total'] for n in network_stats.values())

        for network, stats in sorted_networks:
            percentage = (stats['total'] / total_network_choices) * 100 if total_network_choices > 0 else 0
            print(f"    # The number and percentage of times this network was selected.")
            print(f"  - {network:<10}: {stats['total']} times ({percentage:.2f}%)")
            print(f"      # How many of the selections resulted in a success.")
            print(f"    - Success: {stats['success']}")
            print(f"      # How many of the selections resulted in a failure.")
            print(f"    - Fail:    {stats['fail']}")

    # --- Savings & Optimization Analysis ---
    print("\nSavings and Optimization Analysis:")
    print("    # Evaluates the performance of the routing logic against a perfect outcome.")
    
    print(f"\n  - Transactions with Best Option Selected: {optimal_choices}")

    # --- Analysis for Successful Transactions ---
    print("\n  --- Analysis for Successful Transactions ---")
    missed_savings_success = best_possible_savings_success - achieved_savings_success
    print(f"    - Total Savings Achieved:           ${achieved_savings_success:,.2f}")
    print(f"    - Best Possible Savings:            ${best_possible_savings_success:,.2f}")
    print(f"    - Savings Missed:                   ${missed_savings_success:,.2f}")

    # --- Analysis for Failed Transactions ---
    print("\n  --- Analysis for Failed Transactions ---")
    print(f"    - Savings Missed (Opp. Cost):      ${best_possible_savings_fail:,.2f}")
    print(f"      # This is the total savings that could have been realized if these transactions had succeeded via their optimal route.")
            
    # --- Overall Summary ---
    total_missed_savings = missed_savings_success + best_possible_savings_fail
    print("\n  --- Overall Summary ---")
    print(f"    - Total Potential Savings Missed (for Successful and Failed transactions):     ${total_missed_savings:,.2f}")
            
    print("\n--- End of Report ---")


# === NEW FUNCTIONS FOR ASCII TABLE GENERATION ===

def generate_ascii_table(headers, rows, title="", max_width=80):
    """
    Generate ASCII table with borders and proper alignment.
    """
    if not rows:
        return f"\n{title}\nNo data available.\n"
    
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width_col = len(str(header))
        for row in rows:
            if i < len(row):
                max_width_col = max(max_width_col, len(str(row[i])))
        col_widths.append(max_width_col + 2)  # Add padding
    
    # Create table
    table_lines = []
    
    # Title
    if title:
        title_line = f"│{title.center(sum(col_widths) + len(headers) - 1)}│"
        table_lines.append("┌" + "─" * (sum(col_widths) + len(headers) - 1) + "┐")
        table_lines.append(title_line)
        table_lines.append("├" + "┬".join("─" * w for w in col_widths) + "┤")
    else:
        table_lines.append("┌" + "┬".join("─" * w for w in col_widths) + "┐")
    
    # Headers
    header_line = "│"
    for i, header in enumerate(headers):
        header_line += f" {str(header).ljust(col_widths[i] - 1)}│"
    table_lines.append(header_line)
    table_lines.append("├" + "┼".join("─" * w for w in col_widths) + "┤")
    
    # Data rows
    for row in rows:
        row_line = "│"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                # Right-align numbers, left-align text
                cell_str = str(cell)
                if cell_str.replace('.', '').replace('%', '').replace('$', '').replace(',', '').replace('-', '').isdigit():
                    formatted_cell = cell_str.rjust(col_widths[i] - 1)
                else:
                    formatted_cell = cell_str.ljust(col_widths[i] - 1)
                row_line += f" {formatted_cell}│"
        table_lines.append(row_line)
    
    # Bottom border
    table_lines.append("└" + "┴".join("─" * w for w in col_widths) + "┘")
    
    return "\n" + "\n".join(table_lines) + "\n"

def create_csv_performance_report(input_file=None, output_dir=None):
    """
    Create separate CSV performance reports for each category in a dedicated folder.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if input_file:
        results_path = input_file
    else:
        results_path = os.path.join(project_root, "scene-1", "output_results.csv")
    
    if output_dir:
        reports_folder = os.path.join(output_dir, "detailed_performance_analysis")
    else:
        reports_folder = os.path.join(project_root, "scene-1", "detailed_performance_analysis")

    if not os.path.exists(results_path):
        print(f"ERROR: Cannot create CSV reports - results file not found at {results_path}")
        return

    # Create the reports folder if it doesn't exist
    try:
        os.makedirs(reports_folder, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create reports folder: {e}")
        return

    # Re-analyze data for CSV generation
    total_transactions = 0
    successful_transactions = 0
    optimal_choices = 0
    
    processor_stats = defaultdict(lambda: {
        'total': 0, 'success': 0, 'fail': 0,
        'networks': defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})
    })
    network_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})

    try:
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_transactions += 1
                is_success = row.get('final_outcome') == 'success'
                
                if is_success:
                    successful_transactions += 1
                
                chosen_processor = row.get('chosen_processor')
                chosen_network = row.get('chosen_network')

                if chosen_processor and chosen_processor != 'None':
                    stats = processor_stats[chosen_processor]
                    stats['total'] += 1
                    if is_success:
                        stats['success'] += 1
                    else:
                        stats['fail'] += 1

                    if chosen_network and chosen_network != 'None':
                        network_stats_ptr = stats['networks'][chosen_network]
                        network_stats_ptr['total'] += 1
                        if is_success:
                            network_stats_ptr['success'] += 1
                        else:
                            network_stats_ptr['fail'] += 1
                
                if chosen_network and chosen_network != 'None':
                    network_stats[chosen_network]['total'] += 1
                    if is_success:
                        network_stats[chosen_network]['success'] += 1
                    else:
                        network_stats[chosen_network]['fail'] += 1

                is_optimal = (row.get('chosen_processor') == row.get('best_possible_processor') and 
                              row.get('chosen_network') == row.get('best_possible_network'))
                
                if is_optimal and row.get('chosen_processor'):
                    optimal_choices += 1

    except (IOError, KeyError) as e:
        print(f"ERROR: Could not read results file for CSV report: {e}")
        return

    # Generate separate CSV files for each category
    
    # 1. Overall Performance CSV
    overall_success_rate = (successful_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    failed_transactions = total_transactions - successful_transactions
    
    overall_data = [
        ["Metric", "Value", "Percentage", "Success_Rate"],
        ["Total Transactions", total_transactions, "100.00%", f"{overall_success_rate:.2f}%"],
        ["Successful Transactions", successful_transactions, f"{(successful_transactions/total_transactions)*100:.2f}%", "100.00%"],
        ["Failed Transactions", failed_transactions, f"{(failed_transactions/total_transactions)*100:.2f}%", "0.00%"],
        ["Best Option Selected", optimal_choices, f"{(optimal_choices/total_transactions)*100:.2f}%", "100.00%"]
    ]
    
    try:
        overall_csv_path = os.path.join(reports_folder, "overall_performance.csv")
        with open(overall_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(overall_data)
        print(f"Overall performance CSV saved to: {overall_csv_path}")
    except IOError as e:
        print(f"ERROR: Could not write overall performance CSV: {e}")

    # 2. Processor Performance CSV
    if processor_stats:
        processor_data = [["Processor", "Selections", "Percentage", "Success", "Failure", "Success_Rate"]]
        total_choices = sum(p['total'] for p in processor_stats.values())
        sorted_processors = sorted(processor_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        
        for processor, stats in sorted_processors:
            percentage = (stats['total'] / total_choices) * 100 if total_choices > 0 else 0
            success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            processor_data.append([
                processor, stats['total'], f"{percentage:.2f}%", 
                stats['success'], stats['fail'], f"{success_rate:.2f}%"
            ])
        
        try:
            processor_csv_path = os.path.join(reports_folder, "processor_performance.csv")
            with open(processor_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(processor_data)
            print(f"Processor performance CSV saved to: {processor_csv_path}")
        except IOError as e:
            print(f"ERROR: Could not write processor performance CSV: {e}")

    # 3. Network Performance CSV
    if network_stats:
        network_data = [["Network", "Selections", "Percentage", "Success", "Failure", "Success_Rate"]]
        total_network_choices = sum(n['total'] for n in network_stats.values())
        sorted_networks = sorted(network_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        
        for network, stats in sorted_networks:
            percentage = (stats['total'] / total_network_choices) * 100 if total_network_choices > 0 else 0
            success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            network_data.append([
                network, stats['total'], f"{percentage:.2f}%", 
                stats['success'], stats['fail'], f"{success_rate:.2f}%"
            ])
        
        try:
            network_csv_path = os.path.join(reports_folder, "network_performance.csv")
            with open(network_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(network_data)
            print(f"Network performance CSV saved to: {network_csv_path}")
        except IOError as e:
            print(f"ERROR: Could not write network performance CSV: {e}")

    # 4. Processor-Network Breakdown CSV
    breakdown_data = [["Processor", "Network", "Selections", "Percentage_of_Processor", "Success", "Failure", "Success_Rate"]]
    
    for processor, stats in sorted(processor_stats.items(), key=lambda item: item[1]['total'], reverse=True):
        if stats['networks']:
            sorted_networks = sorted(stats['networks'].items(), key=lambda item: item[1]['total'], reverse=True)
            for network, net_stats in sorted_networks:
                net_percentage = (net_stats['total'] / stats['total']) * 100 if stats['total'] > 0 else 0
                success_rate = (net_stats['success'] / net_stats['total']) * 100 if net_stats['total'] > 0 else 0
                breakdown_data.append([
                    processor, network, net_stats['total'], f"{net_percentage:.2f}%",
                    net_stats['success'], net_stats['fail'], f"{success_rate:.2f}%"
                ])
    
    if len(breakdown_data) > 1:  # More than just headers
        try:
            breakdown_csv_path = os.path.join(reports_folder, "processor_network_breakdown.csv")
            with open(breakdown_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(breakdown_data)
            print(f"Processor-Network breakdown CSV saved to: {breakdown_csv_path}")
        except IOError as e:
            print(f"ERROR: Could not write processor-network breakdown CSV: {e}")

    print(f"\nAll CSV reports saved to folder: {reports_folder}")

def create_detailed_performance_report(input_file=None, output_dir=None):
    """
    Create detailed ASCII table report file by re-analyzing the results data.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if input_file:
        results_path = input_file
    else:
        results_path = os.path.join(project_root, "scene-1", "output_results.csv")
    
    if output_dir:
        report_path = os.path.join(output_dir, "detailed_performance_report.txt")
    else:
        report_path = os.path.join(project_root, "scene-1", "detailed_performance_report.txt")

    if not os.path.exists(results_path):
        print(f"ERROR: Cannot create detailed report - results file not found at {results_path}")
        return

    # Re-analyze data for table generation
    total_transactions = 0
    successful_transactions = 0
    optimal_choices = 0
    
    processor_stats = defaultdict(lambda: {
        'total': 0, 'success': 0, 'fail': 0,
        'networks': defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})
    })
    network_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})

    try:
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_transactions += 1
                is_success = row.get('final_outcome') == 'success'
                
                if is_success:
                    successful_transactions += 1
                
                chosen_processor = row.get('chosen_processor')
                chosen_network = row.get('chosen_network')

                if chosen_processor and chosen_processor != 'None':
                    stats = processor_stats[chosen_processor]
                    stats['total'] += 1
                    if is_success:
                        stats['success'] += 1
                    else:
                        stats['fail'] += 1

                    if chosen_network and chosen_network != 'None':
                        network_stats_ptr = stats['networks'][chosen_network]
                        network_stats_ptr['total'] += 1
                        if is_success:
                            network_stats_ptr['success'] += 1
                        else:
                            network_stats_ptr['fail'] += 1
                
                if chosen_network and chosen_network != 'None':
                    network_stats[chosen_network]['total'] += 1
                    if is_success:
                        network_stats[chosen_network]['success'] += 1
                    else:
                        network_stats[chosen_network]['fail'] += 1

                is_optimal = (row.get('chosen_processor') == row.get('best_possible_processor') and 
                              row.get('chosen_network') == row.get('best_possible_network'))
                
                if is_optimal and row.get('chosen_processor'):
                    optimal_choices += 1

    except (IOError, KeyError) as e:
        print(f"ERROR: Could not read results file for detailed report: {e}")
        return

    # Generate report content
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = []
    report_content.append("=" * 80)
    report_content.append("                    DECISION ENGINE PERFORMANCE REPORT")
    report_content.append("=" * 80)
    report_content.append(f"Generated: {timestamp}")
    report_content.append(f"Simulation: scene-1/output_results.csv")
    report_content.append(f"Total Transactions: {total_transactions}")
    report_content.append("")

    # Overall Performance Table
    overall_success_rate = (successful_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    failed_transactions = total_transactions - successful_transactions
    
    overall_headers = ["Metric", "Value", "Percentage", "Success Rate"]
    overall_rows = [
        ["Total Transactions", total_transactions, "100.00%", f"{overall_success_rate:.2f}%"],
        ["Successful Transactions", successful_transactions, f"{(successful_transactions/total_transactions)*100:.2f}%", "100.00%"],
        ["Failed Transactions", failed_transactions, f"{(failed_transactions/total_transactions)*100:.2f}%", "0.00%"],
        ["Best Option Selected", optimal_choices, f"{(optimal_choices/total_transactions)*100:.2f}%", "100.00%"]
    ]
    
    report_content.append(generate_ascii_table(overall_headers, overall_rows, "OVERALL PERFORMANCE"))

    # Processor Performance Table
    if processor_stats:
        processor_headers = ["Processor", "Selections", "Percentage", "Success", "Failure", "Success Rate"]
        processor_rows = []
        total_choices = sum(p['total'] for p in processor_stats.values())
        
        sorted_processors = sorted(processor_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        for processor, stats in sorted_processors:
            percentage = (stats['total'] / total_choices) * 100 if total_choices > 0 else 0
            success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            processor_rows.append([
                processor, stats['total'], f"{percentage:.2f}%", 
                stats['success'], stats['fail'], f"{success_rate:.2f}%"
            ])
        
        report_content.append(generate_ascii_table(processor_headers, processor_rows, "PROCESSOR PERFORMANCE"))

    # Network Performance Table
    if network_stats:
        network_headers = ["Network", "Selections", "Percentage", "Success", "Failure", "Success Rate"]
        network_rows = []
        total_network_choices = sum(n['total'] for n in network_stats.values())
        
        sorted_networks = sorted(network_stats.items(), key=lambda item: item[1]['total'], reverse=True)
        for network, stats in sorted_networks:
            percentage = (stats['total'] / total_network_choices) * 100 if total_network_choices > 0 else 0
            success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            network_rows.append([
                network, stats['total'], f"{percentage:.2f}%", 
                stats['success'], stats['fail'], f"{success_rate:.2f}%"
            ])
        
        report_content.append(generate_ascii_table(network_headers, network_rows, "NETWORK PERFORMANCE"))

    # Processor-Network Breakdown Table
    breakdown_headers = ["Combination", "Selections", "% of Proc.", "Success", "Failure", "Success Rate"]
    breakdown_rows = []
    
    for processor, stats in sorted(processor_stats.items(), key=lambda item: item[1]['total'], reverse=True):
        if stats['networks']:
            sorted_networks = sorted(stats['networks'].items(), key=lambda item: item[1]['total'], reverse=True)
            for network, net_stats in sorted_networks:
                net_percentage = (net_stats['total'] / stats['total']) * 100 if stats['total'] > 0 else 0
                success_rate = (net_stats['success'] / net_stats['total']) * 100 if net_stats['total'] > 0 else 0
                # Truncate long names for better table formatting
                combo_name = f"{processor[:6]}-{network[:4]}"
                breakdown_rows.append([
                    combo_name, net_stats['total'], f"{net_percentage:.2f}%",
                    net_stats['success'], net_stats['fail'], f"{success_rate:.2f}%"
                ])
    
    if breakdown_rows:
        report_content.append(generate_ascii_table(breakdown_headers, breakdown_rows, "PROCESSOR-NETWORK BREAKDOWN"))

    # Footer
    report_content.append("=" * 80)
    report_content.append("                              END OF REPORT")
    report_content.append("=" * 80)

    # Write to file
    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_content))
        print(f"\nDetailed report saved to: {report_path}")
    except IOError as e:
        print(f"ERROR: Could not write detailed report file: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate results analysis and reports')
    parser.add_argument('--input-file', type=str,
                       help='Path to input results CSV file')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for reports')
    return parser.parse_args()

def main():
    """Main function with argument parsing"""
    args = parse_arguments()
    
    print("INFO: Starting results analysis...")
    analyze_results(args.input_file, args.output_dir)
    print("INFO: Creating detailed performance report...")
    create_detailed_performance_report(args.input_file, args.output_dir)
    print("INFO: Creating CSV performance reports...")
    create_csv_performance_report(args.input_file, args.output_dir)
    print("INFO: Results analysis complete.")

if __name__ == "__main__":
    main()
