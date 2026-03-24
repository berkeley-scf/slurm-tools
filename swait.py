#!/usr/bin/env python3
"""
Analyze Slurm job queue wait times by QoS and resource allocation.

This script queries sacct for job data and reports wait time percentiles
(25th, 50th, 75th, 90th, 95th, 99th) broken out by QoS and number of
GPUs (for GPU partitions) or CPUs (for non-GPU partitions).

Usage:
    python swait.py -S START_DATE -E END_DATE -p PARTITION [--nodes NODE1,NODE2,...] [--gpu TYPE1,TYPE2,...] [--time_units] [-v]

Arguments:
    -S, --start-date: Start date in YYYY-MM-DD format (required)
    -E, --end-date:   End date in YYYY-MM-DD format (required)
    -p, --partition:  Partition name (required)
    --nodes:          Comma-separated list of nodes to filter on (optional)
    --gpu:            Comma-separated list of GPU types to filter on (optional)
    --time_units:     Report wait times in seconds/minutes/hours/days (default: whole minutes)
    -v, --verbose:    Enable verbose output

Examples:
    python swait.py -S 2025-01-01 -E 2025-06-30 -p low
    python swait.py -S 2025-01-01 -E 2025-06-30 -p jsteinhardt --nodes smaug,balrog
    python swait.py -S 2025-01-01 -E 2025-06-30 -p gpu --gpu A100
    python swait.py -S 2025-01-01 -E 2025-06-30 -p gpu --gpu A5000,A4000
    python swait.py -S 2025-01-01 -E 2025-06-30 -p low --time_units
    python swait.py -S 2025-01-01 -E 2025-06-30 -p lambda -v
"""

import pandas as pd
import re
import sys
import argparse
import subprocess
import io
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze Slurm job queue wait times by QoS and resource allocation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-S', '--start-date', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('-E', '--end-date', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('-p', '--partition', type=str, required=True,
                        help='Partition name to analyze')
    parser.add_argument('--nodes', type=str, default=None,
                        help='Comma-separated list of nodes to filter on')
    parser.add_argument('--gpu', type=str, default=None,
                        help='Comma-separated list of GPU types to filter on (e.g., A100,A5000)')
    parser.add_argument('--time_units', '--time-units', dest='time_units', default=False, action='store_true',
                        help='Report wait times in seconds/minutes/hours/days (default: whole minutes)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def load_capacity_data(verbose=False):
    """Load node capacity data from node-capacity.txt."""
    try:
        capacity_df = pd.read_csv('/var/nitedump/node-capacity.txt', sep='|')
        if verbose:
            print(f"Loaded capacity data for {len(capacity_df)} nodes")
        return capacity_df
    except (FileNotFoundError, Exception) as e:
        if verbose:
            print(f"Warning: Could not read node-capacity.txt: {e}")
        return pd.DataFrame(columns=['node', 'partition', 'cpus', 'gpus', 'hyperthreading', 'gpu_type'])


def partition_has_gpus(capacity_df, partition):
    """Return True if any node in the partition has GPUs > 0."""
    part_nodes = capacity_df[capacity_df['partition'] == partition]
    if len(part_nodes) == 0:
        return False
    return (part_nodes['gpus'] > 0).any()


def parse_gpu_allocation(alloc_tres):
    """Parse GPU count from AllocTRES field."""
    if pd.isna(alloc_tres) or alloc_tres == '':
        return 0
    match = re.search(r'gres/gpu[^=]*=(\d+)', str(alloc_tres))
    if match:
        return int(match.group(1))
    return 0


def query_sacct_data(start_date, end_date, partition, verbose=False):
    """Query job data from sacct for the specified date range and partition."""
    sacct_cmd = [
        'sacct', '-a',
        '-S', start_date,
        '-E', end_date,
        '-r', partition,
        '--format=JobID,User,Partition,QoS,Account,NodeList,Submit,Start,Elapsed,ElapsedRaw,AllocCPUS,AllocTRES,State',
        '--parsable2',
        '--allocations',
    ]

    try:
        if verbose:
            print("Running sacct command...")
        result = subprocess.run(sacct_cmd, capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        filtered_lines = [line for line in lines if 'None assigned' not in line]

        if len(filtered_lines) <= 1:
            if verbose:
                print("No job data found for the specified date range.")
            return pd.DataFrame()

        csv_data = '\n'.join(filtered_lines)
        df = pd.read_csv(io.StringIO(csv_data), sep='|')

        if verbose:
            print(f"Loaded {len(df):,} jobs from sacct")
        return df

    except subprocess.CalledProcessError as e:
        print(f"Error running sacct command: {e}", file=sys.stderr)
        if e.stderr:
            print(f"stderr: {e.stderr}", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing sacct output: {e}", file=sys.stderr)
        return pd.DataFrame()


def compute_wait_times(df, partition, nodes, gpu_types, capacity_df, verbose=False):
    """Compute wait times and return a DataFrame with QoS and resource columns."""
    if verbose:
        print(f"Jobs in partition '{partition}': {len(df):,}")

    if len(df) == 0:
        return pd.DataFrame()

    # Filter by nodes if specified
    if nodes:
        node_set = {n.strip() for n in nodes}
        df = df[df['NodeList'].isin(node_set)].copy()
        if verbose:
            print(f"Jobs after node filter: {len(df):,}")

    # Filter by GPU type(s) using node-capacity.txt data (not AllocTRES)
    if gpu_types:
        gpu_type_set = {g.strip().upper() for g in gpu_types if g.strip()}
        part_capacity = capacity_df[capacity_df['partition'] == partition].copy()
        part_capacity['gpu_type'] = part_capacity['gpu_type'].fillna('').astype(str).str.upper()

        gpu_nodes = set(part_capacity[part_capacity['gpu_type'].isin(gpu_type_set)]['node'])
        df = df[df['NodeList'].isin(gpu_nodes)].copy()

        if verbose:
            matched_gpu_types = sorted(set(part_capacity[part_capacity['gpu_type'].isin(gpu_type_set)]['gpu_type']))
            print(f"GPU type filter requested: {', '.join(sorted(gpu_type_set))}")
            print(f"GPU types found in partition '{partition}': {', '.join(matched_gpu_types) if matched_gpu_types else 'none'}")
            print(f"Jobs after GPU type filter: {len(df):,}")

    if len(df) == 0:
        return pd.DataFrame()

    # Drop jobs that never started (sacct outputs literal string "None")
    df = df[~(df['Start'].isna() | (df['Start'] == 'None'))].copy()

    # Parse Submit and Start as datetimes, drop rows where parsing fails
    df['Submit_dt'] = pd.to_datetime(df['Submit'], errors='coerce')
    df['Start_dt'] = pd.to_datetime(df['Start'], errors='coerce')
    df = df.dropna(subset=['Submit_dt', 'Start_dt'])

    # Compute wait time in minutes
    df['wait_seconds'] = (df['Start_dt'] - df['Submit_dt']).dt.total_seconds()
    # Drop negative waits (shouldn't happen, but be safe)
    df = df[df['wait_seconds'] >= 0].copy()
    df['wait_minutes'] = df['wait_seconds'] / 60.0

    if verbose:
        print(f"Jobs with valid wait times: {len(df):,}")

    # Determine resource column based on whether partition has GPUs
    has_gpus = partition_has_gpus(capacity_df, partition)
    if has_gpus:
        df['resource_count'] = df['AllocTRES'].apply(parse_gpu_allocation)
        resource_label = 'GPUs'
    else:
        cpu_bins = [0, 1, 2, 4, 8, 16, 32, 64, 128, float('inf')]
        cpu_labels = ['1', '2', '3-4', '5-8', '9-16', '17-32', '33-64', '65-128', '>128']
        df['resource_count'] = pd.cut(df['AllocCPUS'], bins=cpu_bins, labels=cpu_labels, right=True)
        resource_label = 'CPUs'

    return df, resource_label


def format_duration(minutes, time_units=False):
    """Format a duration in minutes for reporting."""
    if not time_units:
        return f"{minutes:.0f}"
    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    elif minutes < 60:
        return f"{minutes:.0f}m"
    elif minutes < 1440:
        hours = minutes / 60
        return f"{hours:.0f}h"
    else:
        days = minutes / 1440
        return f"{days:.0f}d"


def wait_color(minutes):
    """Return a rich color name based on wait time in minutes."""
    if minutes < 10:
        return 'green'
    elif minutes < 120:
        return 'yellow'
    else:
        return 'red'


def print_report(df, resource_label, partition, start_date, end_date, nodes, gpu_types, time_units):
    """Print wait time percentiles grouped by QoS and resource count."""
    console = Console()

    console.print(f"\nWait Time and Preemption Analysis for partition '{partition}' from {start_date} to {end_date}")
    if nodes:
        console.print(f"Filtered to nodes: {', '.join(nodes)}")
    if gpu_types:
        console.print(f"Filtered to GPU types: {', '.join(gpu_types)}")

    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    pct_labels = ['p25', 'p50', 'p75', 'p90', 'p95', 'p99']

    # CPU bins are categoricals with a defined order; GPU counts are plain ints.
    # observed=True suppresses the FutureWarning for categorical groupby.
    grouped = df.groupby(['QOS', 'resource_count'], observed=True)

    cpu_bin_order = ['1', '2', '3-4', '5-8', '9-16', '17-32', '33-64', '65-128', '>128']
    is_cpu = resource_label == 'CPUs'

    def sort_key(item):
        qos, res = item[0]
        if is_cpu:
            order = cpu_bin_order.index(str(res)) if str(res) in cpu_bin_order else 999
        else:
            order = int(res)
        return (qos, order)

    # Build rows storing raw minute values for coloring
    rows = []
    for (qos, res_count), group in sorted(grouped, key=sort_key):
        waits = group['wait_minutes']
        pcts = waits.quantile(percentiles).values
        rows.append({
            'QoS': qos,
            resource_label: str(res_count),
            'n_jobs': len(group),
            'n_preempted': (group['State'] == 'PREEMPTED').sum(),
            'pct_values': pcts,
        })

    if not rows:
        console.print("\nNo jobs found matching criteria.")
        return

    col_width = max(8, len(resource_label))

    if time_units:
        console.print("\nNumber of preempted jobs and wait time percentiles (formatted as seconds/minutes/hours/days):")
    else:
        console.print("\nNumber of preempted jobs and wait time percentiles in minutes:")

    table = Table(show_header=True, header_style="bold")
    table.add_column("QoS", justify="left", min_width=20)
    table.add_column(resource_label, justify="right", min_width=col_width)
    table.add_column("n_jobs", justify="right", min_width=8)
    table.add_column("n_preempt", justify="right", min_width=9)
    for lbl in pct_labels:
        table.add_column(lbl, justify="right", min_width=8)

    prev_qos = None
    for row in rows:
        if prev_qos is not None and row['QoS'] != prev_qos:
            table.add_section()
        pct_cells = [
            Text(format_duration(val, time_units=time_units), style=wait_color(val))
            for val in row['pct_values']
        ]
        ratio = row['n_preempted'] / row['n_jobs'] if row['n_jobs'] > 0 else 0
        if ratio < 0.01:
            preempt_color = 'green'
        elif ratio < 0.1:
            preempt_color = 'yellow'
        else:
            preempt_color = 'red'
        table.add_row(
            row['QoS'],
            row[resource_label],
            f"{row['n_jobs']:,}",
            Text(f"{row['n_preempted']:,}", style=preempt_color),
            *pct_cells,
        )
        prev_qos = row['QoS']

    console.print(table)
    console.print(f"Total jobs analyzed: {len(df):,}")


def main():
    args = parse_arguments()

    nodes = None
    if args.nodes:
        nodes = [n.strip() for n in args.nodes.split(',')]

    gpu_types = None
    if args.gpu:
        gpu_types = [g.strip() for g in args.gpu.split(',')]

    capacity_df = load_capacity_data(args.verbose)

    if args.verbose:
        print(f"Querying job data from {args.start_date} to {args.end_date}...")
    df = query_sacct_data(args.start_date, args.end_date, args.partition, args.verbose)

    if df.empty:
        print("No job data returned.", file=sys.stderr)
        sys.exit(1)

    result = compute_wait_times(df, args.partition, nodes, gpu_types, capacity_df, args.verbose)
    if isinstance(result, pd.DataFrame) and result.empty:
        print(f"No jobs found for partition '{args.partition}'.", file=sys.stderr)
        sys.exit(1)

    wait_df, resource_label = result

    if wait_df.empty:
        print(f"No jobs with valid wait times found.", file=sys.stderr)
        sys.exit(1)

    print_report(wait_df, resource_label, args.partition, args.start_date, args.end_date, nodes, gpu_types, args.time_units)


if __name__ == "__main__":
    main()
