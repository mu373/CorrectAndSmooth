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


def label_by_louvain(G, n_classes=None):
    """Label nodes using Louvain community detection

    If n_classes is specified, merge smallest communities until we have n_classes.
    """
    communities = list(nx.community.louvain_communities(G))

    # Merge communities if we have too many
    if n_classes is not None and len(communities) > n_classes:
        # Sort by size (smallest first) and merge smallest into larger ones
        communities = sorted(communities, key=len)
        while len(communities) > n_classes:
            smallest = communities.pop(0)
            # Merge into the next smallest
            communities[0] = communities[0].union(smallest)
            communities = sorted(communities, key=len)

    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for idx, community in enumerate(communities):
        for node in community:
            labels[node] = idx
    return labels


def label_by_degree(G, n_classes=3, top_n_percent=0.1, bottom_n_percent=0.1):
    """Label nodes by degree percentile

    - Class 0: top n% by degree (hubs)
    - Class n_classes-1: bottom n% by degree (spokes)
    - Classes 1 to n_classes-2: randomly assigned to middle nodes
    """
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    n = len(sorted_nodes)

    top_k = int(n * top_n_percent)
    bottom_k = int(n * bottom_n_percent)

    labels = np.zeros(n, dtype=int)

    # Top nodes -> class 0
    for node in sorted_nodes[:top_k]:
        labels[node] = 0

    # Bottom nodes -> class n_classes-1
    for node in sorted_nodes[-bottom_k:]:
        labels[node] = n_classes - 1

    # Middle nodes -> random classes 1 to n_classes-2
    middle_nodes = sorted_nodes[top_k : n - bottom_k]
    if n_classes > 2 and len(middle_nodes) > 0:
        middle_classes = np.random.randint(1, n_classes - 1, size=len(middle_nodes))
        for node, cls in zip(middle_nodes, middle_classes):
            labels[node] = cls

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


def save_to_ogb_format(G, labels, features, output_dir, verbose=True):
    """Save graph in OGB raw format"""
    os.makedirs(output_dir, exist_ok=True)

    # edge.csv.gz
    if verbose:
        print("      Saving edge.csv.gz...")
    edges = list(G.edges())
    with gzip.open(os.path.join(output_dir, "edge.csv.gz"), "wt") as f:
        for src, dst in edges:
            f.write(f"{src},{dst}\n")

    # node-feat.csv.gz
    if verbose:
        print("      Saving node-feat.csv.gz...")
    with gzip.open(os.path.join(output_dir, "node-feat.csv.gz"), "wt") as f:
        for row in features:
            f.write(",".join(map(str, row)) + "\n")

    # node-label.csv.gz
    if verbose:
        print("      Saving node-label.csv.gz...")
    with gzip.open(os.path.join(output_dir, "node-label.csv.gz"), "wt") as f:
        for label in labels:
            f.write(f"{label}\n")


def save_metadata(metadata, output_dir, verbose=True):
    """Save metadata to JSON file"""
    if verbose:
        print("      Saving metadata.json...")
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def generate_ba_dataset(
    name=None,
    n_nodes=1000,
    m=5,
    labeling="louvain",
    n_classes=None,
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
        n_classes: Number of classes. For louvain, merges communities. For degree, default is 3.
        top_n_percent: Top percent for degree labeling (hubs)
        bottom_n_percent: Bottom percent for degree labeling (spokes)
        dim_features: Feature dimension
        sigma: Noise level (higher = harder classification)
        seed: Random seed

    Returns:
        output_dir: Path to the data directory
    """
    # Auto-increment name if not provided
    if name is None:
        name = get_next_name()
    print(f"[1/4] Generating BA graph: n_nodes={n_nodes}, m={m}")

    np.random.seed(seed)

    # Generate graph
    G = generate_ba_graph(n_nodes, m)
    print(f"[2/4] Generating labels: method={labeling}, n_classes={n_classes}")

    # Generate labels
    if labeling == "louvain":
        labels = label_by_louvain(G, n_classes=n_classes)
    else:
        nc = n_classes if n_classes is not None else 3
        labels = label_by_degree(G, n_classes=nc, top_n_percent=top_n_percent, bottom_n_percent=bottom_n_percent)

    print(f"[3/4] Generating features: dim={dim_features}, sigma={sigma}")
    # Generate features
    features = generate_features(labels, dim_features, sigma)

    print(f"[4/4] Saving to {f'dataset/ba/{name}'}...")
    # Save to dataset/ba/{name}/
    output_dir = f"dataset/ba/{name}"
    save_to_ogb_format(G, labels, features, output_dir)

    # Save metadata
    metadata = {
        "name": name,
        "n_nodes": n_nodes,
        "m": m,
        "labeling": labeling,
        "n_classes_requested": n_classes,
        "top_n_percent": top_n_percent,
        "bottom_n_percent": bottom_n_percent,
        "dim_features": dim_features,
        "sigma": sigma,
        "seed": seed,
        "n_edges": G.number_of_edges(),
        "n_classes": len(np.unique(labels)),
    }
    save_metadata(metadata, output_dir)

    print(f"Done! {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(np.unique(labels))} classes, features={features.shape}")

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
        "--n_classes",
        type=int,
        default=None,
        help="Number of classes. For louvain, merges communities. For degree, default is 3.",
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
        n_classes=args.n_classes,
        top_n_percent=args.top_n_percent,
        bottom_n_percent=args.bottom_n_percent,
        dim_features=args.dim_features,
        sigma=args.sigma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
