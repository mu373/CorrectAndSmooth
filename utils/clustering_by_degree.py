"""
Compute average clustering coefficient per degree C(k) for an undirected graph.

Usage:
    python -m utils.clustering_by_degree dataset/ws/ws001/edge.csv.gz
    python -m utils.clustering_by_degree dataset/ws/ws001/edge.csv.gz --no-save  # just prints
"""

import argparse
import gzip
import json
import os

import networkx as nx
import numpy as np


def _load_graph(graph_or_path):
    """Return a NetworkX graph from a path or passthrough if already a graph."""
    if hasattr(graph_or_path, "degree"):
        return graph_or_path

    path = graph_or_path
    G = nx.Graph()
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, dst = line.split(",")
            G.add_edge(int(src), int(dst))
    return G


def compute_clustering_by_degree(graph_or_path, output_dir: str = None) -> dict:
    """
    Compute average clustering coefficient per degree C(k).

    Args:
        graph_or_path: NetworkX graph or path to edge.csv(.gz)
        output_dir: If set, write clustering_by_degree.json there. If None and
                    a path is provided, defaults to the edge file directory.

    Returns:
        dict with:
            - 'clustering_by_degree': {degree: avg_clustering}
            - 'stats': overall stats (avg_clustering, n_nodes, n_edges, max_degree)
    """
    G = _load_graph(graph_or_path)
    clustering = nx.clustering(G)
    degrees = dict(G.degree())

    sums = {}
    counts = {}
    for node, k in degrees.items():
        c = clustering.get(node, 0.0)
        sums[k] = sums.get(k, 0.0) + c
        counts[k] = counts.get(k, 0) + 1

    ck = {int(k): float(sums[k] / counts[k]) for k in sorted(sums.keys()) if counts[k] > 0}
    avg_clust = float(np.mean(list(clustering.values()))) if clustering else 0.0
    result = {
        "clustering_by_degree": ck,
        "stats": {
            "avg_clustering": avg_clust,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "max_degree": max(degrees.values()) if degrees else 0,
        },
    }

    if output_dir is None and not hasattr(graph_or_path, "degree"):
        output_dir = os.path.dirname(graph_or_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "clustering_by_degree.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved clustering-by-degree to {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute C(k) (clustering by degree) from edge list")
    parser.add_argument("edge_list_path", type=str, help="Path to edge.csv(.gz) file")
    parser.add_argument("--no-save", action="store_true", help="Do not write clustering_by_degree.json")
    args = parser.parse_args()

    if args.no_save:
        compute_clustering_by_degree(args.edge_list_path, output_dir=None)
    else:
        compute_clustering_by_degree(args.edge_list_path)


if __name__ == "__main__":
    main()
