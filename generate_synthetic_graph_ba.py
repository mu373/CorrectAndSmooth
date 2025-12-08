"""Generate synthetic Barabasi-Albert graphs with node labels and features.

Usage (CLI):
    # Auto-increment name (ba001, ba002, ...)
    python generate_synthetic_graph_ba.py

    # With custom parameters
    python generate_synthetic_graph_ba.py --sigma 2.0 --m 10
    python generate_synthetic_graph_ba.py --labeling degree --n_nodes 5000
    python generate_synthetic_graph_ba.py --name my_test --sigma 0.5

Usage (Python):
    from generate_synthetic_graph_ba import generate_ba_dataset

    # Auto-increment name
    generate_ba_dataset(sigma=0.5)  # -> ba001
    generate_ba_dataset(sigma=1.0)  # -> ba002

    # Explicit name
    generate_ba_dataset("my_test", sigma=0.5)

    # Parameter sweep
    for sigma in [0.5, 1.0, 2.0]:
        generate_ba_dataset(sigma=sigma, m=5)

Output:
    dataset/ba/{name}/
        edge.csv.gz       # source,target pairs
        node-feat.csv.gz  # node features (dim_features columns)
        node-label.csv.gz # node labels
        metadata.json     # all parameters + results
"""

import networkx as nx
import numpy as np
import gzip
import json
import os


def generate_ba_graph(n_nodes, m):
    """Generate Barabasi-Albert graph"""
    G = nx.barabasi_albert_graph(n_nodes, m)
    return G


def label_by_louvain(G):
    """Label nodes using Louvain community detection"""
    communities = nx.community.louvain_communities(G)
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for idx, community in enumerate(communities):
        for node in community:
            labels[node] = idx
    return labels


def label_by_degree(G, top_n_percent=0.1, bottom_n_percent=0.1):
    """Label nodes by degree: top=1, bottom=2, middle=0"""
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    n = len(sorted_nodes)

    top_k = int(n * top_n_percent)
    bottom_k = int(n * bottom_n_percent)

    labels = np.zeros(n, dtype=int)
    for node in sorted_nodes[:top_k]:
        labels[node] = 1
    for node in sorted_nodes[-bottom_k:]:
        labels[node] = 2
    return labels


def generate_features(labels, dim_features, sigma=1.0):
    """Generate features from labels using cluster centroids (GMM)

    For each class c, create centroid mu_c ~ N(0, I)
    For each node i with label y_i: x_i = mu_{y_i} + epsilon, where epsilon ~ N(0, sigma)
    """
    n_nodes = len(labels)
    unique_labels = np.unique(labels)

    # Generate centroids for each class
    centroids = {label: np.random.randn(dim_features) for label in unique_labels}

    # Generate features
    features = np.zeros((n_nodes, dim_features))
    for i, label in enumerate(labels):
        mu = centroids[label]
        epsilon = np.random.randn(dim_features) * sigma
        features[i] = mu + epsilon

    return features


def get_next_name(base_dir="dataset/ba"):
    """Find next available name: ba001, ba002, ..."""
    if not os.path.exists(base_dir):
        return "ba001"
    existing = [
        d for d in os.listdir(base_dir) if d.startswith("ba") and d[2:].isdigit()
    ]
    if not existing:
        return "ba001"
    max_num = max(int(d[2:]) for d in existing)
    return f"ba{max_num + 1:03d}"


def save_to_ogb_format(G, labels, features, output_dir):
    """Save graph in OGB raw format"""
    os.makedirs(output_dir, exist_ok=True)

    # edge.csv.gz
    edges = list(G.edges())
    with gzip.open(os.path.join(output_dir, "edge.csv.gz"), "wt") as f:
        for src, dst in edges:
            f.write(f"{src},{dst}\n")

    # node-feat.csv.gz
    with gzip.open(os.path.join(output_dir, "node-feat.csv.gz"), "wt") as f:
        for row in features:
            f.write(",".join(map(str, row)) + "\n")

    # node-label.csv.gz
    with gzip.open(os.path.join(output_dir, "node-label.csv.gz"), "wt") as f:
        for label in labels:
            f.write(f"{label}\n")

    # # num-node-list.csv.gz
    # with gzip.open(os.path.join(output_dir, "num-node-list.csv.gz"), "wt") as f:
    #     f.write(f"{G.number_of_nodes()}\n")

    # # num-edge-list.csv.gz
    # with gzip.open(os.path.join(output_dir, "num-edge-list.csv.gz"), "wt") as f:
    #     f.write(f"{G.number_of_edges()}\n")


def save_metadata(metadata, output_dir):
    """Save metadata to JSON file"""
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def generate_ba_dataset(
    name=None,
    n_nodes=1000,
    m=5,
    labeling="louvain",
    top_n_percent=0.1,
    bottom_n_percent=0.1,
    dim_features=128,
    sigma=1.0,
    seed=42,
):
    """Generate BA graph dataset and save to dataset/ba/{name}/

    Args:
        name: Dataset name. If None, auto-increment (ba001, ba002, ...)
        n_nodes: Number of nodes
        m: BA graph density (edges per new node)
        labeling: "louvain" or "degree"
        top_n_percent: Top percent for degree labeling
        bottom_n_percent: Bottom percent for degree labeling
        dim_features: Feature dimension
        sigma: Noise level (higher = harder classification)
        seed: Random seed

    Returns:
        output_dir: Path to the data directory
    """
    # Auto-increment name if not provided
    if name is None:
        name = get_next_name()

    np.random.seed(seed)

    # Generate graph
    G = generate_ba_graph(n_nodes, m)

    # Generate labels
    if labeling == "louvain":
        labels = label_by_louvain(G)
    else:
        labels = label_by_degree(G, top_n_percent, bottom_n_percent)

    # Generate features
    features = generate_features(labels, dim_features, sigma)

    # Save to dataset/ba/{name}/
    output_dir = f"dataset/ba/{name}"
    save_to_ogb_format(G, labels, features, output_dir)

    # Save metadata
    metadata = {
        "name": name,
        "n_nodes": n_nodes,
        "m": m,
        "labeling": labeling,
        "top_n_percent": top_n_percent,
        "bottom_n_percent": bottom_n_percent,
        "dim_features": dim_features,
        "sigma": sigma,
        "seed": seed,
        "n_edges": G.number_of_edges(),
        "n_classes": len(np.unique(labels)),
    }
    save_metadata(metadata, output_dir)

    print(
        f"Generated graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    print(f"Labels: {len(np.unique(labels))} classes")
    print(f"Features: {features.shape}")
    print(f"Saved to {output_dir}")

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic BA graph with labels and features"
    )
    parser.add_argument("--n_nodes", type=int, default=1000, help="Number of nodes")
    parser.add_argument(
        "--m", type=int, default=5, help="BA graph density (edges per new node)"
    )
    parser.add_argument(
        "--labeling",
        choices=["louvain", "degree"],
        default="louvain",
        help="Labeling method: louvain (community) or degree (structural)",
    )
    parser.add_argument(
        "--top_n_percent",
        type=float,
        default=0.1,
        help="Top percent for degree labeling",
    )
    parser.add_argument(
        "--bottom_n_percent",
        type=float,
        default=0.1,
        help="Bottom percent for degree labeling",
    )
    parser.add_argument(
        "--dim_features", type=int, default=128, help="Feature dimension"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise level (higher = harder classification)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name (e.g., ba001). Auto-increment if not provided.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_ba_dataset(
        name=args.name,
        n_nodes=args.n_nodes,
        m=args.m,
        labeling=args.labeling,
        top_n_percent=args.top_n_percent,
        bottom_n_percent=args.bottom_n_percent,
        dim_features=args.dim_features,
        sigma=args.sigma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
