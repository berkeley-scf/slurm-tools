#!/usr/bin/env python3
"""
List available nodes in a Slurm partition.

Queries sinfo for nodes and prints those whose reported state is exactly
idle, mix, or alloc as a space-separated list, expanding any compressed
Slurm node notation (e.g., n[001-005,008]).

Usage:
    python nodelist.py -p <partition>

Arguments:
    -p, --partition: Slurm partition name (required)

Output:
    Space-separated list of node names in idle, mix, or alloc states.

Examples:
    python nodelist.py -p gpu
    python nodelist.py -p low
"""

import sys
import argparse
import subprocess


def expand_nodelist(nodelist):
    """Expand a compressed Slurm nodelist (e.g., n[001-005]) to individual node names."""
    if not nodelist or nodelist == '(null)':
        return []
    result = subprocess.run(
        ['scontrol', 'show', 'hostnames', nodelist],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error expanding nodelist '{nodelist}': {result.stderr.strip()}", file=sys.stderr)
        return []
    return [n for n in result.stdout.strip().split('\n') if n]


def main():
    parser = argparse.ArgumentParser(
        description='List available nodes in a Slurm partition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-p', '--partition', type=str, required=True,
                        help='Slurm partition name')
    args = parser.parse_args()

    result = subprocess.run(
        ['sinfo', '-p', args.partition, '-o', '%t %N', '--noheader'],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error running sinfo: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)

    wanted_states = {'idle', 'mix', 'alloc'}
    all_nodes = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        state, nodelist = parts
        if state in wanted_states:
            all_nodes.extend(expand_nodelist(nodelist))

    print(' '.join(sorted(all_nodes)))


if __name__ == '__main__':
    main()
