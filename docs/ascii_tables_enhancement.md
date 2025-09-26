# Performance Reports Enhancement

## Overview

The `results.py` script has been enhanced to generate both ASCII table reports and CSV exports in addition to the existing console output. This enhancement provides professional, readable formats that can be easily shared, analyzed, and imported into other tools.

## What's New

### Enhanced `results.py` Script

The script now generates three outputs:

1. **Console Output**: The original detailed text analysis (unchanged)
2. **ASCII Table Report**: A new file with professionally formatted tables
3. **CSV Report**: A structured CSV file for data analysis and visualization

### Generated Files

#### ASCII Table Report
- **Filename**: `detailed_performance_report.txt`
- **Location**: Same directory as the simulation results (e.g., `scene-1/`)
- **Format**: ASCII tables with Unicode box-drawing characters

#### CSV Reports
- **Location**: `detailed_performance_analysis/` folder within the simulation directory
- **Format**: Separate CSV files for each performance category
- **Files Generated**:
  - `overall_performance.csv` - Summary metrics and totals
  - `processor_performance.csv` - Performance by processor
  - `network_performance.csv` - Performance by network
  - `processor_network_breakdown.csv` - Detailed processor-network combinations

## Usage

Run the script exactly as before:

```bash
python3 scripts/results.py
```

**Output:**
- Console displays the original detailed analysis
- File `detailed_performance_report.txt` is automatically generated
- File `performance_report.csv` is automatically generated
- Success messages show both file locations

## Report Structure

The ASCII table report includes:

### 1. Overall Performance Table
- Total transactions processed
- Success/failure breakdown
- Overall success rates
- Best option selection metrics

### 2. Processor Performance Table
- Selection frequency for each processor
- Success/failure counts
- Success rates by processor

### 3. Network Performance Table
- Network usage statistics
- Performance metrics by network
- Success rates by network type

### 4. Processor-Network Breakdown Table
- Detailed combination analysis
- Performance of specific processor-network pairs
- Percentage breakdown within each processor

## Features

### Professional Formatting
- Unicode box-drawing characters for clean borders
- Proper column alignment (numbers right-aligned, text left-aligned)
- Consistent spacing and padding
- Centered table titles

### Automatic Sizing
- Tables automatically adjust to content width
- Proper handling of varying data lengths
- Truncated names for better formatting when needed

### Comprehensive Data
- All performance metrics included
- Calculated percentages and rates
- Timestamp and source file information

## File Example

```
================================================================================
                    DECISION ENGINE PERFORMANCE REPORT
================================================================================
Generated: 2025-09-25 23:27:41
Simulation: scene-1/output_results.csv
Total Transactions: 100

┌─────────────────────────────────────────────────────────────┐
│                     OVERALL PERFORMANCE                     │
├─────────────────────────┬───────┬────────────┬──────────────┤
│ Metric                  │ Value │ Percentage │ Success Rate │
├─────────────────────────┼───────┼────────────┼──────────────┤
│ Total Transactions      │    100│     100.00%│        44.00%│
│ Successful Transactions │     44│      44.00%│       100.00%│
│ Failed Transactions     │     56│      56.00%│         0.00%│
│ Best Option Selected    │     44│      44.00%│       100.00%│
└─────────────────────────┴───────┴────────────┴──────────────┘
```

## CSV Reports Structure

The framework generates separate CSV files for each performance category in the `detailed_performance_analysis/` folder:

### 1. Overall Performance (`overall_performance.csv`)
```csv
Metric,Value,Percentage,Success_Rate
Total Transactions,100,100.00%,44.00%
Successful Transactions,44,44.00%,100.00%
Failed Transactions,56,56.00%,0.00%
Best Option Selected,44,44.00%,100.00%
```

### 2. Processor Performance (`processor_performance.csv`)
```csv
Processor,Selections,Percentage,Success,Failure,Success_Rate
Stripe,51,54.26%,26,25,50.98%
Adyen,43,45.74%,18,25,41.86%
```

### 3. Network Performance (`network_performance.csv`)
```csv
Network,Selections,Percentage,Success,Failure,Success_Rate
MASTERCARD,35,37.23%,13,22,37.14%
VISA,35,37.23%,16,19,45.71%
STAR,21,22.34%,13,8,61.90%
ACCEL,3,3.19%,2,1,66.67%
```

### 4. Processor-Network Breakdown (`processor_network_breakdown.csv`)
```csv
Processor,Network,Selections,Percentage_of_Processor,Success,Failure,Success_Rate
Stripe,MASTERCARD,22,43.14%,9,13,40.91%
Stripe,VISA,18,35.29%,9,9,50.00%
Stripe,STAR,8,15.69%,6,2,75.00%
Adyen,VISA,17,39.53%,7,10,41.18%
```

### CSV Benefits
- **Organized by Category**: Each CSV focuses on specific analysis type
- **Excel/Spreadsheet Compatible**: Easy to open in Excel, Google Sheets, etc.
- **Data Analysis Ready**: Perfect for pivot tables, charts, and statistical analysis
- **Programmatic Access**: Easy to import into Python, R, or other analysis tools
- **Focused Analysis**: Each file contains relevant columns for its specific data type
- **Visualization**: Create targeted charts and graphs from focused datasets

## Benefits

1. **Easy Sharing**: Both text and CSV files can be shared via email, messaging, or documentation
2. **Version Control**: Can be committed to Git for historical tracking
3. **Universal Viewing**: ASCII tables open in any text editor, CSV opens in spreadsheet applications
4. **Professional Appearance**: Clean, structured formats for reports and analysis
5. **Data Analysis**: CSV format enables advanced analysis and visualization
6. **Backward Compatible**: Original console output remains unchanged

## Implementation Details

### Non-Invasive Enhancement
- No existing code was modified
- New functions added at the end of `results.py`
- Single line addition to main function
- Zero risk to existing functionality

### Code Structure
- `generate_ascii_table()`: Core table generation function
- `create_detailed_performance_report()`: Report orchestration function
- Automatic data re-analysis for table generation
- Error handling for file operations

## Future Enhancements

Potential future improvements could include:
- Multiple output formats (HTML, Markdown, CSV)
- Configurable table styles
- Color coding for terminal output
- Custom report templates
- Batch processing for multiple scenarios
