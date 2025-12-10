"""Utility to compute degree distribution from edge list or NetworkX graph.

Usage:
    # From Python with NetworkX graph (in-memory, preferred):
    from utils.degree_distribution import compute_degree_distribution
    result = compute_degree_distribution(G)

    # From Python with edge list file:
    result = compute_degree_distribution("dataset/ba/ba001/edge.csv.gz")

    # CLI:
    python -m utils.degree_distribution dataset/ba/ba001/edge.csv.gz
"""

import gzip
import json
import os
from collections import Counter

import numpy as np


def compute_degree_distribution(graph_or_path, output_dir: str = None) -> dict:
    """
    Compute degree distribution from NetworkX graph or edge list file.

    Args:
        graph_or_path: NetworkX graph object OR path to edge.csv.gz file
        output_dir: Directory to save degree_distribution.json (optional)
                   If graph_or_path is a file path, defaults to its directory

    Returns:
        dict with:
        - 'distribution': dict mapping degree (int) -> count (int)
        - 'stats': dict with min, max, mean, std, median
        - 'degrees': list of degree for each node (sorted by node id)
    """
    # Check if input is a NetworkX graph or file path
    if hasattr(graph_or_path, 'degree'):
        # It's a NetworkX graph - use in-memory computation
        degrees = [d for _, d in sorted(graph_or_path.degree())]
    else:
        # It's a file path - load from disk
        edge_list_path = graph_or_path
        degree_count = Counter()

        open_fn = gzip.open if edge_list_path.endswith('.gz') else open
        with open_fn(edge_list_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                src, dst = int(parts[0]), int(parts[1])
                degree_count[src] += 1
                degree_count[dst] += 1

        if not degree_count:
            return {'distribution': {}, 'stats': {}, 'degrees': []}

        max_node = max(degree_count.keys())
        degrees = [degree_count.get(i, 0) for i in range(max_node + 1)]

        # Default output_dir to file's directory
        if output_dir is None:
            output_dir = os.path.dirname(edge_list_path)

    # Compute distribution (degree -> count)
    distribution = Counter(degrees)
    distribution = {int(k): int(v) for k, v in sorted(distribution.items())}

    # Compute stats
    degrees_array = np.array(degrees)
    stats = {
        'min': int(np.min(degrees_array)),
        'max': int(np.max(degrees_array)),
        'mean': float(np.mean(degrees_array)),
        'std': float(np.std(degrees_array)),
        'median': float(np.median(degrees_array)),
        'n_nodes': len(degrees),
    }

    result = {
        'distribution': distribution,
        'stats': stats,
        'degrees': degrees,
    }

    # Save to JSON if output_dir is provided
    if output_dir:
        output_path = os.path.join(output_dir, 'degree_distribution.json')
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"      Saved degree distribution to {output_path}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute degree distribution from edge list")
    parser.add_argument("edge_list_path", type=str, help="Path to edge.csv.gz file")
    parser.add_argument("--no-save", action="store_true", help="Don't save to JSON file")
    args = parser.parse_args()

    result = compute_degree_distribution(args.edge_list_path, save=not args.no_save)

    print(f"Stats: {result['stats']}")
    print(f"Degree distribution (top 10):")
    for degree, count in list(result['distribution'].items())[:10]:
        print(f"  degree {degree}: {count} nodes")


if __name__ == "__main__":
    main()
