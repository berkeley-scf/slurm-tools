#!/usr/bin/env python3
"""
Analyze CPU or GPU usage for Slurm partitions from sacct job data.

This script provides comprehensive analysis of cluster resource utilization including:
- CPU utilization by partition and node (when --mode cpu is specified)
- GPU utilization by node (when --mode gpu is specified)

The script queries job data directly from Slurm using the sacct command for the specified date range.

Usage:
    python analyze_usage.py -S START_DATE -E END_DATE --mode {cpu,gpu} [-p partition] [--save-jobs] [-v]
    
Arguments:
    -S, --start-date: Start date for sacct query in YYYY-MM-DD format (required)
    -E, --end-date: End date for sacct query in YYYY-MM-DD format (required)
    --mode: Analysis mode - either "cpu" or "gpu" (required)
    -p, --partition: Partition name to analyze (optional, defaults to all partitions)
    --save-jobs: Save detailed job data to CSV file (optional, can be large)
    -v, --verbose: Enable verbose output (optional, shows loading and saving messages)

CPU Mode:
    Analyzes CPU utilization by partition and node:
    - Calculates CPU-hours used vs. available capacity
    - Provides utilization percentages by partition and node
    - Accounts for hyperthreading when available in node-capacity.txt
                   
GPU Mode:
    Analyzes GPU utilization by node:
    - Parses AllocTRES column for GPU allocation (e.g., "gres/gpu=2")
    - Reports GPU-hours used and utilization percentage per node
    - Shows GPU type and capacity from node-capacity.txt
                   
Examples:
    python analyze_usage.py -S 2025-01-01 -E 2025-08-31 --mode cpu              # CPU analysis for all partitions
    python analyze_usage.py -S 2025-01-01 -E 2025-12-31 -p epurdom --mode gpu  # GPU analysis for epurdom partition
    python analyze_usage.py -S 2024-01-01 -E 2025-12-31 -p gpu --mode gpu --save-jobs  # GPU analysis with detailed job data
"""

import pandas as pd
import math
import sys
import argparse
import subprocess
import io
from datetime import datetime, timedelta

def parse_gpu_allocation(alloc_tres):
    """Parse GPU allocation from AllocTRES field."""
    import re
    if pd.isna(alloc_tres) or alloc_tres == '':
        return 0
    
    # Look for patterns like "gres/gpu=4" or "gres/gpu:tesla=2"
    match = re.search(r'gres/gpu[^=]*=(\d+)', str(alloc_tres))
    if match:
        return int(match.group(1))
    return 0

def load_capacity_data(verbose=False):
    """Load node capacity data from node-capacity.txt"""
    try:
        capacity_df = pd.read_csv('/var/nitedump/node-capacity.txt', sep='|')
        if verbose:
            print(f"Loaded capacity data for {len(capacity_df)} nodes")
        return capacity_df
    except FileNotFoundError:
        if verbose:
            print("Warning: node-capacity.txt file not found. Creating empty capacity DataFrame.")
        return pd.DataFrame(columns=['node', 'partition', 'cpus', 'gpus', 'hyperthreading', 'gpu_type'])
    except Exception as e:
        if verbose:
            print(f"Warning: Error reading node-capacity.txt: {e}. Creating empty capacity DataFrame.")
        return pd.DataFrame(columns=['node', 'partition', 'cpus', 'gpus', 'hyperthreading', 'gpu_type'])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze CPU and GPU usage for Slurm partitions using sacct',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -S 2025-01-01 -E 2025-08-31 --mode cpu                    # CPU analysis for all partitions
  %(prog)s -S 2025-01-01 -E 2025-12-31 -p epurdom --mode gpu        # GPU analysis for epurdom partition
  %(prog)s -S 2024-01-01 -E 2025-12-31 -p gpu --mode gpu --save-jobs # GPU analysis with detailed job data
        """
    )
    
    parser.add_argument(
        '-p', '--partition',
        type=str,
        help='Partition name to analyze (default: all partitions)'
    )
    
    parser.add_argument(
        '-S', '--start-date',
        type=str,
        required=True,
        help='Start date for sacct query in YYYY-MM-DD format (e.g., 2025-01-01)'
    )
    
    parser.add_argument(
        '-E', '--end-date',
        type=str,
        required=True,
        help='End date for sacct query in YYYY-MM-DD format (e.g., 2025-08-31)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['cpu', 'gpu'],
        required=True,
        help='Analysis mode: "cpu" for CPU utilization analysis or "gpu" for GPU utilization analysis'
    )
    
    parser.add_argument(
        '--save-jobs',
        action='store_true',
        help='Save detailed job data to CSV file (can be large, default: False)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (default: False)'
    )
    
    return parser.parse_args()

def query_sacct_data(start_date, end_date, verbose=False):
    """Query job data from sacct for the specified date range."""
    # Construct the sacct command
    sacct_cmd = [
        'sacct', '-a', 
        '-S', start_date, 
        '-E', end_date,
        '--format=JobID,User,Partition,QoS,Account,NodeList,Start,Elapsed,ElapsedRaw,AllocCPUS,AllocTRES',
        '--parsable2', 
        '--allocations'
    ]
    
    try:
        # Run sacct command
        if verbose:
            print("Running sacct command...")
        sacct_process = subprocess.run(sacct_cmd, capture_output=True, text=True, check=True)
        
        # Filter out "None assigned" lines
        lines = sacct_process.stdout.strip().split('\n')
        filtered_lines = [line for line in lines if "None assigned" not in line]
        
        if len(filtered_lines) <= 1:  # Only header or no data
            if verbose:
                print("No job data found for the specified date range.")
            return pd.DataFrame()
        
        # Convert to pandas DataFrame
        csv_data = '\n'.join(filtered_lines)
        df = pd.read_csv(io.StringIO(csv_data), sep='|')
        
        if verbose:
            print(f"Successfully loaded {len(df):,} jobs from sacct")
        return df
        
    except subprocess.CalledProcessError as e:
        print(f"Error running sacct command: {e}")
        print(f"Command output: {e.stderr}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing sacct output: {e}")
        return pd.DataFrame()

def calculate_cpu_utilization(df, capacity_df, start_date, end_date, partition):
    """Calculate CPU utilization by partition."""

    # Store original df for detailed analysis
    original_df = df.copy()
    
    # Filter for CPU partitions if specified
    if partition:
        df = df[df['Partition'] == partition].copy()  # Explicit copy to avoid SettingWithCopyWarning
        original_df = original_df[original_df['Partition'] == partition].copy()
    
    # Filter capacity for CPU partitions (those with CPUs > 0)
    capacity_cpu = capacity_df[capacity_df['cpus'] > 0]
    
    # Merge with capacity data to get hyperthreading information
    df = df.merge(capacity_cpu[['node', 'hyperthreading']], 
                  left_on='NodeList', right_on='node', how='left')
    original_df = original_df.merge(capacity_cpu[['node', 'hyperthreading']], 
                                    left_on='NodeList', right_on='node', how='left')
    
    # Check if NodeList contains multiple nodes (has comma or range notation)
    df['single_node'] = ~(df['NodeList'].str.contains(',') | df['NodeList'].str.contains('-'))
    original_df['single_node'] = ~(original_df['NodeList'].str.contains(',') | original_df['NodeList'].str.contains('-'))
    
    # Adjust AllocCPUS for hyperthreading: round up to nearest even integer for single nodes with hyperthreading
    df['adjusted_cpus'] = df['AllocCPUS'].where(
        ~(df['single_node'] & (df['hyperthreading'] == 1)),
        df['AllocCPUS']+1
    )
    original_df['adjusted_cpus'] = original_df['AllocCPUS'].where(
        ~(original_df['single_node'] & (original_df['hyperthreading'] == 1)),
        original_df['AllocCPUS']+1 
    )

    # Calculate CPU-seconds for each job (using ElapsedRaw which is in seconds)
    df['cpu_seconds'] = df['adjusted_cpus'] * df['ElapsedRaw']
    original_df['cpu_seconds'] = original_df['adjusted_cpus'] * original_df['ElapsedRaw']
    original_df['cpu_hours'] = original_df['cpu_seconds'] / 3600
    original_df['elapsed_hours'] = original_df['ElapsedRaw'] / 3600
    
    
    # Group by partition and node
    grouped = df.groupby(['Partition', 'NodeList']).agg({
        'cpu_seconds': 'sum',  # Total CPU-seconds allocated
        'JobID': 'count'  # Number of jobs (using JobID instead of JobIDRaw)
    }).reset_index()
    
    # Rename the count column to be clearer
    grouped = grouped.rename(columns={'JobID': 'JobCount'})
    
    # Calculate total capacity separately from job data
    if partition:
        # Only include nodes from the specified partition
        capacity_partition = capacity_df[(capacity_df['partition'] == partition) & (capacity_df['cpus'] > 0)]
    else:
        # Include all partitions that appear in the job data
        partitions_in_data = df['Partition'].unique()
        capacity_partition = capacity_df[capacity_df['partition'].isin(partitions_in_data) & (capacity_df['cpus'] > 0)]
    
    # Calculate total time period in seconds
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    total_seconds = (end_dt - start_dt).total_seconds() + 24 * 3600  # Add 1 day to include both start and end dates
    
    # Add capacity information to the grouped data for per-node analysis
    grouped = grouped.merge(capacity_partition[['node', 'partition', 'cpus']], 
                           left_on='NodeList', right_on='node', how='left')
    
    # Calculate utilization for nodes that had jobs
    grouped['total_capacity_cpu_seconds'] = grouped['cpus'] * total_seconds
    grouped['cpu_utilization'] = grouped['cpu_seconds'] / grouped['total_capacity_cpu_seconds']
    
    # Return both grouped results and original job data for detailed analysis
    return grouped, original_df

def calculate_gpu_utilization(df, capacity_df, start_date, end_date, partition):
    """Calculate GPU utilization by node."""
    # Store original df for detailed analysis
    original_df = df.copy()

    # Filter for GPU partitions if specified
    if partition:
        df = df[df['Partition'] == partition].copy()  # Explicit copy to avoid SettingWithCopyWarning
        original_df = original_df[original_df['Partition'] == partition].copy()
        
    # Parse GPU allocation from AllocTRES
    df['gpu_count'] = df['AllocTRES'].apply(parse_gpu_allocation)
    original_df['gpu_count'] = original_df['AllocTRES'].apply(parse_gpu_allocation)
    
    # Calculate GPU-seconds for each job
    df['gpu_seconds'] = df['gpu_count'] * df['ElapsedRaw']
    original_df['gpu_seconds'] = original_df['gpu_count'] * original_df['ElapsedRaw']
    original_df['gpu_hours'] = original_df['gpu_seconds'] / 3600
    original_df['elapsed_hours'] = original_df['ElapsedRaw'] / 3600
    
    # Filter for jobs that actually used GPUs
    df_gpu = df[df['gpu_count'] > 0]
    original_gpu_jobs = original_df[original_df['gpu_count'] > 0]
    
    # Group by node
    grouped = df_gpu.groupby('NodeList').agg({
        'gpu_seconds': 'sum',
        'JobID': 'count'  # Number of jobs (using JobID instead of JobIDRaw)
    }).reset_index()
    
    # Rename the count column to be clearer
    grouped = grouped.rename(columns={'JobID': 'JobCount'})
    
    # Add capacity information for per-node analysis of nodes that had GPU jobs
    if partition:
        capacity_partition = capacity_df[(capacity_df['partition'] == partition) & (capacity_df['gpus'] > 0)]
    else:
        partitions_in_data = df_gpu['Partition'].unique() if len(df_gpu) > 0 else []
        capacity_partition = capacity_df[capacity_df['partition'].isin(partitions_in_data) & (capacity_df['gpus'] > 0)]
    
    # Merge with GPU capacity for nodes that had jobs
    grouped = grouped.merge(capacity_partition[['node', 'partition', 'gpus']], 
                           left_on='NodeList', right_on='node', how='left')
    
    # Calculate total time period in seconds
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    total_seconds = (end_dt - start_dt).total_seconds() + 24 * 3600  # Add 1 day to include both start and end dates
    
    # Calculate utilization for nodes that had GPU jobs
    grouped['total_capacity_gpu_seconds'] = grouped['gpus'] * total_seconds
    grouped['gpu_utilization'] = grouped['gpu_seconds'] / grouped['total_capacity_gpu_seconds']
    
    # Return both grouped results and original GPU job data for detailed analysis
    return grouped, original_gpu_jobs

def print_cpu_results(results, job_data, capacity_df, partition, start_date, end_date, verbose=False):
    """Generate CPU utilization results as a string."""
    output = []
    output.append(f"\nCPU Analysis for {partition if partition else 'all partitions'} from {start_date} to {end_date}")
    output.append("=" * 80)
    
    # Calculate analysis period in days and total seconds
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    analysis_period_days = (end_dt - start_dt).days + 1
    total_seconds = (end_dt - start_dt).total_seconds() + 24 * 3600

    if total_seconds < 28*24*3600:
        output.append(f"\nWarning: Time interval is less than 28 days. Calculations are based only on\njobs starting in the time interval and not the length of time the jobs ran in\nthe interval, so the approximate utilization estimates may be inaccurate for\nshort time intervals.")
    
    # Summary by partition
    partition_summary = results.groupby('Partition').agg({
        'cpu_seconds': 'sum',
        'JobCount': 'sum'
    }).reset_index()
    
    # Calculate total capacity from capacity file for accurate utilization
    if partition:
        # Calculate capacity for the specific partition
        partition_capacity = capacity_df[(capacity_df['partition'] == partition) & (capacity_df['cpus'] > 0)]
        total_cpus_available = partition_capacity['cpus'].sum()
        total_cpu_seconds_available = total_cpus_available * total_seconds
        
        partition_summary['total_capacity_cpu_seconds'] = total_cpu_seconds_available
    else:
        # Calculate capacity for each partition separately
        partitions_in_data = results['Partition'].unique()
        capacity_by_partition = capacity_df[capacity_df['partition'].isin(partitions_in_data) & (capacity_df['cpus'] > 0)].groupby('partition')['cpus'].sum()
        
        partition_summary['total_capacity_cpu_seconds'] = partition_summary['Partition'].map(
            lambda p: capacity_by_partition.get(p, 0) * total_seconds
        )
    
    partition_summary['cpu_utilization'] = partition_summary['cpu_seconds'] / partition_summary['total_capacity_cpu_seconds']
    partition_summary['cpu_hours_used'] = partition_summary['cpu_seconds'] / 3600
    partition_summary['cpu_hours_available'] = partition_summary['total_capacity_cpu_seconds'] / 3600
    
    output.append("\nCPU Utilization by Partition:")
    for _, row in partition_summary.iterrows():
        output.append(f"\nPartition: {row['Partition']}")
        output.append(f"  Number of jobs: {row['JobCount']:,}")
        output.append(f"  Total CPU-hours used: {row['cpu_hours_used']:,.2f}")
        output.append(f"  Total CPU-hours available: {row['cpu_hours_available']:,.0f}")
        output.append(f"  Utilization percentage: {row['cpu_utilization']*100:.2f}%")
        output.append(f"  Analysis period: {analysis_period_days} days ({start_date} to {end_date})")
    
    # QoS breakdown if QoS data exists
    if 'QOS' in job_data.columns and not job_data['QOS'].isna().all():
        output.append(f"\nUsage Breakdown by QoS:")
        
        qos_analysis = job_data.groupby('QOS').agg({
            'JobID': 'count',
            'cpu_hours': 'sum',
            'elapsed_hours': ['mean', 'median', 'max']
        }).round(2)
        
        qos_analysis.columns = ['JobCount', 'TotalCPUHours', 'AvgDuration_Hours', 'MedianDuration_Hours', 'MaxDuration_Hours']
        
        # Calculate total capacity for utilization percentage
        total_cpu_hours_available = partition_summary['cpu_hours_available'].sum()
        qos_analysis['CPUUtilizationPercent'] = (qos_analysis['TotalCPUHours'] / total_cpu_hours_available) * 100

        for qos in qos_analysis.index:
            qos_data = qos_analysis.loc[qos]
            output.append(f"\n  QoS: {qos}")
            output.append(f"    Number of jobs: {qos_data['JobCount']:,}")
            output.append(f"    Total CPU-hours: {qos_data['TotalCPUHours']:,.2f}")
            output.append(f"    CPU utilization of total capacity: {qos_data['CPUUtilizationPercent']:.2f}%")
            output.append(f"    Average job duration: {qos_data['AvgDuration_Hours']:.2f} hours")
            output.append(f"    Median job duration: {qos_data['MedianDuration_Hours']:.2f} hours")
            output.append(f"    Longest job duration: {qos_data['MaxDuration_Hours']:.2f} hours")
        
        # Save QoS summary to CSV
        qos_summary_file = f"cpu_{partition if partition else 'all'}_{start_date}_{end_date}_qos_summary.csv"
        qos_analysis.to_csv(qos_summary_file)
        if verbose:
            output.append(f"\nQoS summary saved to: {qos_summary_file}")
    
    # Detailed node results - show CPU-hours per node
    output.append(f"\nCPU-Hours by Node (sorted by usage):")
    node_results = results.copy()
    node_results['cpu_hours_used'] = node_results['cpu_seconds'] / 3600
    sorted_nodes = node_results.sort_values('cpu_hours_used', ascending=False)
    node_table = sorted_nodes[['Partition', 'NodeList', 'cpu_hours_used', 'JobCount']].to_string(
        index=False, 
        float_format='%.2f',
        formatters={'cpu_hours_used': '{:,.2f}'.format, 'JobCount': '{:,}'.format}
    )
    output.append(node_table)
    
    # Job duration statistics
    if len(job_data) > 0:
        avg_duration = job_data['elapsed_hours'].mean()
        median_duration = job_data['elapsed_hours'].median()
        max_duration = job_data['elapsed_hours'].max()
        
        output.append(f"\nJob Duration Statistics:")
        output.append(f"  Average job duration: {avg_duration:.2f} hours")
        output.append(f"  Median job duration: {median_duration:.2f} hours")
        output.append(f"  Longest job duration: {max_duration:.2f} hours")
        
        # CPU allocation distribution
        output.append(f"\nDistribution of Jobs by Number of CPUs:")
        cpu_dist = job_data['AllocCPUS'].value_counts().sort_index()
        for cpus, count in cpu_dist.items():
            output.append(f"  {cpus:3d} CPUs: {count:,} jobs")
    
    return '\n'.join(output)

def print_gpu_results(results, job_data, capacity_df, partition, start_date, end_date, verbose=False):
    """Generate GPU utilization results as a string."""
    output = []
    output.append(f"\nGPU Analysis for {partition if partition else 'all partitions'} from {start_date} to {end_date}")
    output.append("=" * 80)
    
    # Calculate analysis period in days and total seconds
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    analysis_period_days = (end_dt - start_dt).days + 1
    total_seconds = (end_dt - start_dt).total_seconds() + 24 * 3600
    
    if total_seconds < 28*24*3600:
        output.append(f"\nWarning: Time interval is less than 28 days. Calculations are based only on\njobs starting in the time interval and not the length of time the jobs ran in\nthe interval, so the approximate utilization estimates may be inaccurate for\nshort time intervals.")

    # Calculate total GPU capacity from capacity file
    if partition:
        # Calculate capacity for the specific partition
        partition_capacity = capacity_df[(capacity_df['partition'] == partition) & (capacity_df['gpus'] > 0)]
    else:
        # Include all partitions that appear in the GPU job data
        partitions_in_data = job_data['Partition'].unique() if len(job_data) > 0 else []
        partition_capacity = capacity_df[capacity_df['partition'].isin(partitions_in_data) & (capacity_df['gpus'] > 0)]
    
    total_gpus_available = partition_capacity['gpus'].sum()
    total_gpu_hours_available = total_gpus_available * analysis_period_days * 24
    total_gpu_hours_used = job_data['gpu_hours'].sum() if len(job_data) > 0 else 0
    overall_utilization = (total_gpu_hours_used / total_gpu_hours_available) * 100 if total_gpu_hours_available > 0 else 0
    
    output.append(f"\nOverall GPU Summary:")
    output.append(f"  Total GPU jobs: {len(job_data):,}")
    output.append(f"  Total GPU-hours used: {total_gpu_hours_used:,.2f}")
    output.append(f"  Total GPU-hours available: {total_gpu_hours_available:,.0f}")
    output.append(f"  Overall GPU utilization: {overall_utilization:.2f}%")
    output.append(f"  Total GPUs available: {total_gpus_available}")
    output.append(f"  Analysis period: {analysis_period_days} days ({start_date} to {end_date})")
        
    # QoS breakdown if QoS data exists
    if len(job_data) > 0 and 'QOS' in job_data.columns and not job_data['QOS'].isna().all():
        output.append(f"\nUsage Breakdown by QoS:")
        
        qos_analysis = job_data.groupby('QOS').agg({
            'JobID': 'count',
            'gpu_hours': 'sum',
            'elapsed_hours': ['mean', 'median', 'max']
        }).round(2)
        
        qos_analysis.columns = ['JobCount', 'TotalGPUHours', 'AvgDuration_Hours', 'MedianDuration_Hours', 'MaxDuration_Hours']
        
        # Calculate utilization percentage for each QoS
        qos_analysis['GPUUtilizationPercent'] = (qos_analysis['TotalGPUHours'] / total_gpu_hours_available) * 100
        
        for qos in qos_analysis.index:
            qos_data = qos_analysis.loc[qos]
            output.append(f"\n  QoS: {qos}")
            output.append(f"    Number of jobs: {qos_data['JobCount']:,}")
            output.append(f"    Total GPU-hours: {qos_data['TotalGPUHours']:,.2f}")
            output.append(f"    GPU utilization of total capacity: {qos_data['GPUUtilizationPercent']:.2f}%")
            output.append(f"    Average job duration: {qos_data['AvgDuration_Hours']:.2f} hours")
            output.append(f"    Median job duration: {qos_data['MedianDuration_Hours']:.2f} hours")
            output.append(f"    Longest job duration: {qos_data['MaxDuration_Hours']:.2f} hours")
        
        # Save QoS summary to CSV
        qos_summary_file = f"gpu_{partition if partition else 'all'}_{start_date}_{end_date}_qos_summary.csv"
        qos_analysis.to_csv(qos_summary_file)
        if verbose:
            output.append(f"\nQoS summary saved to: {qos_summary_file}")
    
    # Detailed node results - show GPU-hours per node
    if len(results) > 0:
        output.append(f"\nGPU-Hours by Node (sorted by usage):")
        node_results = results.copy()
        node_results['gpu_hours_used'] = node_results['gpu_seconds'] / 3600
        sorted_nodes = node_results.sort_values('gpu_hours_used', ascending=False)
        node_table = sorted_nodes[['NodeList', 'gpu_hours_used', 'gpu_utilization', 'JobCount']].to_string(
            index=False, 
            float_format='%.2f',
            formatters={'gpu_hours_used': '{:,.2f}'.format, 'JobCount': '{:,}'.format, 'gpu_utilization': '{:.4f}'.format}
        )
        output.append(node_table)
    
    # Job duration statistics
    if len(job_data) > 0:
        avg_duration = job_data['elapsed_hours'].mean()
        median_duration = job_data['elapsed_hours'].median()
        max_duration = job_data['elapsed_hours'].max()
        
        output.append(f"\nGPU Job Duration Statistics:")
        output.append(f"  Average job duration: {avg_duration:.2f} hours")
        output.append(f"  Median job duration: {median_duration:.2f} hours")
        output.append(f"  Longest job duration: {max_duration:.2f} hours")
        
        # GPU allocation distribution
        output.append(f"\nDistribution of Jobs by Number of GPUs:")
        gpu_dist = job_data['gpu_count'].value_counts().sort_index()
        for gpus, count in gpu_dist.items():
            output.append(f"  {gpus:3d} GPUs: {count:,} jobs")

    return '\n'.join(output)

def run(args):   
    # Query job data from sacct
    if args['verbose']:
        print(f"Querying job data from {args['start_date']} to {args['end_date']}...")
    df = query_sacct_data(args['start_date'], args['end_date'], args['verbose'])
    
    # Load capacity data
    capacity_df = load_capacity_data(args['verbose'])
    
    # Perform analysis based on mode
    if args['mode'] == 'cpu':
        results, job_data = calculate_cpu_utilization(df, capacity_df, args['start_date'], args['end_date'], args['partition'])
        report = print_cpu_results(results, job_data, capacity_df, args['partition'], args['start_date'], args['end_date'], args['verbose'])
         
        # Save CPU results
        output_file = f"cpu_{args['partition'] if args['partition'] else 'all'}_{args['start_date']}_{args['end_date']}_analysis.csv"
        results.to_csv(output_file, index=False)
        if args['verbose']:
            print(f"\nCPU analysis results saved to {output_file}")
        
        # Save detailed job data if requested
        if args['save_jobs']:
            job_output_file = f"cpu_{args['partition'] if args['partition'] else 'all'}_{args['start_date']}_{args['end_date']}_jobs.csv"
            job_data.to_csv(job_output_file, index=False)
            if args['verbose']:
                print(f"Detailed job data saved to {job_output_file}")
        
    elif args['mode'] == 'gpu':
        results, job_data = calculate_gpu_utilization(df, capacity_df, args['start_date'], args['end_date'], args['partition'])
        report = print_gpu_results(results, job_data, capacity_df, args['partition'], args['start_date'], args['end_date'], args['verbose'])

        
        # Save GPU results
        output_file = f"gpu_{args['partition'] if args['partition'] else 'all'}_{args['start_date']}_{args['end_date']}_analysis.csv"
        results.to_csv(output_file, index=False)
        if args['verbose']:
            print(f"\nGPU analysis results saved to {output_file}")
        
        # Save detailed job data if requested
        if args['save_jobs']:
            job_output_file = f"gpu_{args['partition'] if args['partition'] else 'all'}_{args['start_date']}_{args['end_date']}_jobs.csv"
            job_data.to_csv(job_output_file, index=False)
            if args['verbose']:
                print(f"Detailed GPU job data saved to {job_output_file}")

    return report

def main():
    args = parse_arguments()
    args_dict = {'partition': args.partition, 'save_jobs': args.save_jobs, 'verbose': args.verbose, 'start_date': args.start_date, 'end_date': args.end_date, 'mode': args.mode}
    report = run(args_dict)
    print(report)

if __name__ == "__main__":
    main()
