"""Utility to estimate power law exponent from degree distribution.

Requires: pip install tailestim

Usage:
    # From Python with NetworkX graph:
    from utils.power_law_estimator import estimate_power_law
    result = estimate_power_law(G)

    # From Python with degree list:
    result = estimate_power_law(degrees)

    # From Python with edge list file:
    result = estimate_power_law("dataset/ba/ba001/edge.csv.gz")

    # CLI:
    python -m utils.power_law_estimator dataset/ba/ba001/edge.csv.gz
"""

import gzip
import json
import os
from collections import Counter

import numpy as np
from tailestim import HillEstimator


def estimate_power_law(graph_or_path_or_degrees, output_dir: str = None) -> dict:
    """
    Estimate power law exponent from NetworkX graph, degree list, or edge list file.

    Args:
        graph_or_path_or_degrees: NetworkX graph, list of degrees, or path to edge.csv.gz
        output_dir: Directory to save power_law.json (optional)

    Returns:
        dict with:
        - 'gamma': power law exponent
        - 'k_min': minimum degree threshold
        - 'n_tail': number of samples in the tail
        - 'stats': additional statistics
    """
    # Get degrees from input
    if hasattr(graph_or_path_or_degrees, 'degree'):
        # NetworkX graph
        degrees = [d for _, d in graph_or_path_or_degrees.degree()]
    elif isinstance(graph_or_path_or_degrees, str):
        # File path - load from disk
        edge_list_path = graph_or_path_or_degrees
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
            return {'gamma': None, 'k_min': None, 'n_tail': 0, 'stats': {}}

        max_node = max(degree_count.keys())
        degrees = [degree_count.get(i, 0) for i in range(max_node + 1)]

        # Default output_dir to file's directory
        if output_dir is None:
            output_dir = os.path.dirname(edge_list_path)
    else:
        # Assume it's a list of degrees
        degrees = list(graph_or_path_or_degrees)

    # Filter out zero degrees for power law estimation
    degrees = [d for d in degrees if d > 0]

    if len(degrees) < 10:
        return {'gamma': None, 'k_min': None, 'n_tail': len(degrees), 'stats': {}}

    # Fit Hill estimator
    estimator = HillEstimator()
    estimator.fit(degrees)
    result = estimator.get_result()

    output = {
        'gamma': float(result.gamma_) if result.gamma_ is not None else None,
        'k_min': float(result.k_min_) if hasattr(result, 'k_min_') and result.k_min_ is not None else None,
        'n_tail': int(result.n_tail_) if hasattr(result, 'n_tail_') else len(degrees),
        'stats': {
            'n_nodes': len(degrees),
            'mean_degree': float(np.mean(degrees)),
            'max_degree': int(np.max(degrees)),
        }
    }

    # Save to JSON if output_dir is provided
    if output_dir:
        output_path = os.path.join(output_dir, 'power_law.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"      Saved power law estimate to {output_path}")

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Estimate power law exponent from edge list")
    parser.add_argument("edge_list_path", type=str, help="Path to edge.csv.gz file")
    parser.add_argument("--no-save", action="store_true", help="Don't save to JSON file")
    args = parser.parse_args()

    output_dir = None if args.no_save else os.path.dirname(args.edge_list_path)
    result = estimate_power_law(args.edge_list_path, output_dir=output_dir)

    print(f"Power law exponent (gamma): {result['gamma']}")
    print(f"Minimum degree threshold (k_min): {result['k_min']}")
    print(f"Tail size: {result['n_tail']}")
    print(f"Stats: {result['stats']}")


if __name__ == "__main__":
    main()
