#!/usr/bin/env python3
"""
Performance Visualization Script for Decision Engine Testing

This script generates visualizations comparing algorithm performance across different schemas:
1. Success Rate (SR) across algorithms
2. Savings across algorithms

Usage:
    python3 scripts/plot_performance.py
    python3 scripts/plot_performance.py --input summary_report.csv --output-dir charts/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
import argparse
from pathlib import Path

# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_process_data(csv_file):
    """
    Load summary report CSV and process it for visualization.
    Returns processed DataFrame with average data only.
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data from {csv_file}")
        print(f"  Total rows: {len(df)}")
        
        # Filter for average rows only (consolidated algorithm performance)
        avg_df = df[df['Run'] == 'Avg.'].copy()
        print(f"  Average rows: {len(avg_df)}")
        
        if avg_df.empty:
            print("✗ No 'Avg.' rows found in the data")
            return None
        
        # Get unique algorithms and schemas
        algorithms = sorted(avg_df['Algorithm'].unique())
        schemas = sorted(avg_df['Schema'].unique())
        
        print(f"  Algorithms: {', '.join(algorithms)}")
        print(f"  Schemas: {', '.join(schemas)}")
        
        return avg_df, algorithms, schemas
        
    except FileNotFoundError:
        print(f"✗ Error: File '{csv_file}' not found")
        return None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def create_sr_chart(df, algorithms, schemas, output_dir):
    """
    Create Success Rate comparison chart across algorithms and schemas.
    """
    plt.figure(figsize=(12, 8))
    
    # Create color palette for algorithms
    colors = plt.cm.Set1(range(len(algorithms)))
    algorithm_colors = dict(zip(algorithms, colors))
    
    # Plot data for each algorithm
    for algorithm in algorithms:
        algo_data = df[df['Algorithm'] == algorithm]
        
        # Prepare data for plotting
        schema_values = []
        sr_values = []
        
        for schema in schemas:
            schema_data = algo_data[algo_data['Schema'] == schema]
            if not schema_data.empty:
                schema_values.append(schema)
                sr_values.append(float(schema_data['SR'].iloc[0]))
            else:
                # Handle missing data - skip this point
                continue
        
        if schema_values:  # Only plot if we have data
            plt.plot(schema_values, sr_values, 
                    marker='o', linewidth=2.5, markersize=8,
                    label=algorithm, color=algorithm_colors[algorithm])
    
    # Customize the chart
    plt.title('Success Rate Across Different Algorithms', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Schema', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    
    # Set y-axis to show 0-100 range for success rate
    plt.ylim(0, 105)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Customize legend
    plt.legend(title='Algorithm', title_fontsize=12, fontsize=11, 
              loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(schemas) > 5 else 0)
    
    # Add subtle background color
    plt.gca().set_facecolor('#fafafa')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, 'sr_across_algorithms.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Success Rate chart saved: {output_file}")
    
    plt.close()

def create_savings_chart(df, algorithms, schemas, output_dir):
    """
    Create Savings comparison chart across algorithms and schemas.
    """
    plt.figure(figsize=(12, 8))
    
    # Create color palette for algorithms (same as SR chart for consistency)
    colors = plt.cm.Set1(range(len(algorithms)))
    algorithm_colors = dict(zip(algorithms, colors))
    
    # Plot data for each algorithm
    for algorithm in algorithms:
        algo_data = df[df['Algorithm'] == algorithm]
        
        # Prepare data for plotting
        schema_values = []
        savings_values = []
        
        for schema in schemas:
            schema_data = algo_data[algo_data['Schema'] == schema]
            if not schema_data.empty:
                schema_values.append(schema)
                savings_values.append(float(schema_data['Savings'].iloc[0]))
            else:
                # Handle missing data - skip this point
                continue
        
        if schema_values:  # Only plot if we have data
            plt.plot(schema_values, savings_values, 
                    marker='s', linewidth=2.5, markersize=8,
                    label=algorithm, color=algorithm_colors[algorithm])
    
    # Customize the chart
    plt.title('Savings Across Different Algorithms', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Schema', fontsize=14, fontweight='bold')
    plt.ylabel('Savings ($)', fontsize=14, fontweight='bold')
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Customize legend
    plt.legend(title='Algorithm', title_fontsize=12, fontsize=11, 
              loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(schemas) > 5 else 0)
    
    # Add subtle background color
    plt.gca().set_facecolor('#fafafa')
    
    # Format y-axis to show currency properly
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, 'savings_across_algorithms.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Savings chart saved: {output_file}")
    
    plt.close()

def create_combined_chart(df, algorithms, schemas, output_dir):
    """
    Create a combined chart showing both SR and Savings in subplots.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Create color palette for algorithms
    colors = plt.cm.Set1(range(len(algorithms)))
    algorithm_colors = dict(zip(algorithms, colors))
    
    # Plot Success Rate (top subplot)
    for algorithm in algorithms:
        algo_data = df[df['Algorithm'] == algorithm]
        
        schema_values = []
        sr_values = []
        
        for schema in schemas:
            schema_data = algo_data[algo_data['Schema'] == schema]
            if not schema_data.empty:
                schema_values.append(schema)
                sr_values.append(float(schema_data['SR'].iloc[0]))
        
        if schema_values:
            ax1.plot(schema_values, sr_values, 
                    marker='o', linewidth=2.5, markersize=8,
                    label=algorithm, color=algorithm_colors[algorithm])
    
    ax1.set_title('Success Rate Across Different Algorithms', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Algorithm', fontsize=10, loc='upper right')
    ax1.set_facecolor('#fafafa')
    
    # Plot Savings (bottom subplot)
    for algorithm in algorithms:
        algo_data = df[df['Algorithm'] == algorithm]
        
        schema_values = []
        savings_values = []
        
        for schema in schemas:
            schema_data = algo_data[algo_data['Schema'] == schema]
            if not schema_data.empty:
                schema_values.append(schema)
                savings_values.append(float(schema_data['Savings'].iloc[0]))
        
        if schema_values:
            ax2.plot(schema_values, savings_values, 
                    marker='s', linewidth=2.5, markersize=8,
                    label=algorithm, color=algorithm_colors[algorithm])
    
    ax2.set_title('Savings Across Different Algorithms', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Schema', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Savings ($)', fontsize=12, fontweight='bold')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Algorithm', fontsize=10, loc='upper right')
    ax2.set_facecolor('#fafafa')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    # Save the combined chart
    output_file = os.path.join(output_dir, 'combined_performance_charts.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Combined chart saved: {output_file}")
    
    plt.close()

def print_data_summary(df, algorithms, schemas):
    """
    Print a summary of the data for verification.
    """
    print(f"\n{'='*60}")
    print("                    DATA SUMMARY")
    print(f"{'='*60}")
    
    for algorithm in algorithms:
        algo_data = df[df['Algorithm'] == algorithm]
        print(f"\n{algorithm}:")
        for schema in schemas:
            schema_data = algo_data[algo_data['Schema'] == schema]
            if not schema_data.empty:
                sr = schema_data['SR'].iloc[0]
                savings = schema_data['Savings'].iloc[0]
                print(f"  {schema}: SR={sr}%, Savings=${savings}")
            else:
                print(f"  {schema}: No data")

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate performance visualization charts from summary report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/plot_performance.py
  python3 scripts/plot_performance.py --input custom_report.csv
  python3 scripts/plot_performance.py --output-dir visualizations/
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='summary_report.csv',
        help='Input CSV file path (default: summary_report.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for chart files (default: current directory)'
    )
    
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Generate combined chart in addition to individual charts'
    )
    
    return parser.parse_args()

def main():
    """
    Main function to orchestrate the visualization generation.
    """
    args = parse_arguments()
    
    print("="*60)
    print("           PERFORMANCE VISUALIZATION GENERATOR")
    print("="*60)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"✗ Error: Input file '{args.input}' not found")
        print("Please run the automation system first to generate summary_report.csv")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    result = load_and_process_data(args.input)
    if result is None:
        sys.exit(1)
    
    df, algorithms, schemas = result
    
    # Print data summary
    print_data_summary(df, algorithms, schemas)
    
    print(f"\n{'='*60}")
    print("GENERATING CHARTS")
    print(f"{'='*60}")
    
    # Generate individual charts
    create_sr_chart(df, algorithms, schemas, args.output_dir)
    create_savings_chart(df, algorithms, schemas, args.output_dir)
    
    # Generate combined chart if requested
    if args.combined:
        create_combined_chart(df, algorithms, schemas, args.output_dir)
    
    print(f"\n{'='*60}")
    print("                 CHARTS GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("Files generated:")
    print("  • sr_across_algorithms.png - Success Rate comparison")
    print("  • savings_across_algorithms.png - Savings comparison")
    
    if args.combined:
        print("  • combined_performance_charts.png - Combined view")

if __name__ == "__main__":
    main()
