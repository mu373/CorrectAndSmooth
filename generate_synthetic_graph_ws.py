"""Generate synthetic Watts-Strogatz graphs with ring-based labels and smooth features.

Usage (CLI):
    # Auto-increment name (ws001, ws002, ...)
    python generate_synthetic_graph_ws.py

    # Control rewiring and community pattern
    python generate_synthetic_graph_ws.py --p 0.8 --labeling highfreq --n_segments 40
    python generate_synthetic_graph_ws.py --labeling imbalanced --majority_frac 0.8
    python generate_synthetic_graph_ws.py --labeling equal --n_classes 6 --p 0.1
    python generate_synthetic_graph_ws.py --feature_type random_uniform --dim_features 64

Usage (Python):
    from generate_synthetic_graph_ws import generate_ws_dataset

    # Auto-increment name
    generate_ws_dataset(p=0.1)   # -> ws001
    generate_ws_dataset(p=0.8)   # -> ws002

    # Explicit name
    generate_ws_dataset("my_ws_test", p=0.5, labeling="highfreq", n_segments=50)

Output:
    dataset/ws/{name}/
        edge.csv.gz       # source,target pairs
        node-feat.csv.gz  # node features (dim_features columns)
        node-label.csv.gz # node labels
        metadata.json     # all parameters + results
"""

import gzip
import json
import os

import networkx as nx
import numpy as np


def generate_ws_graph(n_nodes, k, p, seed=None):
    """Generate Watts-Strogatz graph."""
    return nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)


def _segments_from_count(n_nodes, n_segments):
    """Create nearly equal-sized contiguous segments summing to n_nodes."""
    base = n_nodes // n_segments
    remainder = n_nodes % n_segments
    segments = [base + (1 if i < remainder else 0) for i in range(n_segments)]
    total = sum(segments)
    if total < n_nodes:
        segments[-1] += n_nodes - total
    elif total > n_nodes:
        segments[-1] -= total - n_nodes
    return segments


def assign_contiguous_labels(n_nodes, segments, labels):
    """Assign labels to contiguous blocks along the ring."""
    labels_arr = np.zeros(n_nodes, dtype=int)
    idx = 0
    for seg_size, label in zip(segments, labels):
        end = min(idx + seg_size, n_nodes)
        labels_arr[idx:end] = label
        idx = end
        if idx >= n_nodes:
            break
    if idx < n_nodes:
        labels_arr[idx:] = labels[-1]
    return labels_arr


def label_equal_segments(n_nodes, n_classes):
    """Label contiguous equal-sized sectors around the ring."""
    segments = _segments_from_count(n_nodes, n_classes)
    labels = list(range(n_classes))
    return assign_contiguous_labels(n_nodes, segments, labels), segments


def label_highfreq_segments(n_nodes, n_classes, n_segments=None, segment_size=None):
    """Label many small contiguous segments to create frequent label switches."""
    if segment_size is not None and segment_size > 0:
        n_segments = max(1, int(np.ceil(n_nodes / segment_size)))
    if n_segments is None:
        n_segments = max(32, n_classes * 4)
    segments = _segments_from_count(n_nodes, n_segments)
    labels = [i % n_classes for i in range(n_segments)]
    return assign_contiguous_labels(n_nodes, segments, labels), segments


def label_imbalanced(n_nodes, majority_frac=0.8, majority_class=0, minority_class=1):
    """Label ring with one large majority segment followed by minority."""
    majority_size = int(round(n_nodes * majority_frac))
    majority_size = min(max(1, majority_size), n_nodes - 1)
    segments = [majority_size, n_nodes - majority_size]
    labels = [majority_class, minority_class]
    return assign_contiguous_labels(n_nodes, segments, labels), segments


def generate_sinusoidal_features(
    n_nodes, dim_features=128, num_frequencies=None, feature_noise_sigma=0.0
):
    """Smooth positional features using sin/cos pairs along the ring."""
    if num_frequencies is None:
        num_frequencies = max(1, dim_features // 2)
    base_dim = 2 * num_frequencies

    positions = np.arange(n_nodes)
    features = np.zeros((n_nodes, base_dim))
    for j in range(num_frequencies):
        freq = 2 * np.pi * (j + 1) / n_nodes
        features[:, 2 * j] = np.sin(freq * positions)
        features[:, 2 * j + 1] = np.cos(freq * positions)

    if base_dim >= dim_features:
        features = features[:, :dim_features]
    else:
        pad = np.zeros((n_nodes, dim_features - base_dim))
        features = np.concatenate([features, pad], axis=1)

    if feature_noise_sigma > 0:
        features = features + np.random.randn(*features.shape) * feature_noise_sigma
    return features


def generate_random_features(
    n_nodes, dim_features=128, low=0.0, high=1.0, feature_noise_sigma=0.0
):
    """Random features sampled uniformly with optional Gaussian noise."""
    features = np.random.uniform(low, high, size=(n_nodes, dim_features))
    if feature_noise_sigma > 0:
        features = features + np.random.randn(*features.shape) * feature_noise_sigma
    return features


def get_next_name(base_dir="dataset/ws"):
    """Find next available name: ws001, ws002, ..."""
    if not os.path.exists(base_dir):
        return "ws001"
    existing = [
        d for d in os.listdir(base_dir) if d.startswith("ws") and d[2:].isdigit()
    ]
    if not existing:
        return "ws001"
    max_num = max(int(d[2:]) for d in existing)
    return f"ws{max_num + 1:03d}"


def save_to_ogb_format(G, labels, features, output_dir, verbose=True):
    """Save graph in OGB raw format."""
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("      Saving edge.csv.gz...")
    edges = list(G.edges())
    with gzip.open(os.path.join(output_dir, "edge.csv.gz"), "wt") as f:
        for src, dst in edges:
            f.write(f"{src},{dst}\n")

    if verbose:
        print("      Saving node-feat.csv.gz...")
    with gzip.open(os.path.join(output_dir, "node-feat.csv.gz"), "wt") as f:
        for row in features:
            f.write(",".join(map(str, row)) + "\n")

    if verbose:
        print("      Saving node-label.csv.gz...")
    with gzip.open(os.path.join(output_dir, "node-label.csv.gz"), "wt") as f:
        for label in labels:
            f.write(f"{label}\n")


def save_metadata(metadata, output_dir, verbose=True):
    """Save metadata to JSON file."""
    if verbose:
        print("      Saving metadata.json...")
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def generate_ws_dataset(
    name=None,
    n_nodes=1000,
    k=10,
    p=0.1,
    labeling="equal",
    n_classes=3,
    n_segments=None,
    segment_size=None,
    majority_frac=0.8,
    majority_class=0,
    minority_class=1,
    dim_features=128,
    num_frequencies=None,
    feature_type="sinusoidal",
    feature_noise_sigma=0.0,
    seed=42,
):
    """Generate WS graph dataset and save to dataset/ws/{name}/."""
    if name is None:
        name = get_next_name()
    print(f"[1/4] Generating WS graph: n_nodes={n_nodes}, k={k}, p={p}")

    np.random.seed(seed)
    G = generate_ws_graph(n_nodes, k, p, seed=seed)

    print(f"[2/4] Generating labels: pattern={labeling}")
    if labeling == "equal":
        labels, segments = label_equal_segments(n_nodes, n_classes)
    elif labeling == "highfreq":
        labels, segments = label_highfreq_segments(
            n_nodes, n_classes, n_segments=n_segments, segment_size=segment_size
        )
    elif labeling == "imbalanced":
        labels, segments = label_imbalanced(
            n_nodes,
            majority_frac=majority_frac,
            majority_class=majority_class,
            minority_class=minority_class,
        )
    else:
        raise ValueError("labeling must be one of: equal, highfreq, imbalanced")

    print(f"[3/4] Generating features: type={feature_type}, dim={dim_features}, num_freq={num_frequencies}")
    if feature_type == "sinusoidal":
        features = generate_sinusoidal_features(
            n_nodes,
            dim_features=dim_features,
            num_frequencies=num_frequencies,
            feature_noise_sigma=feature_noise_sigma,
        )
        meta_num_freq = num_frequencies if num_frequencies is not None else max(1, dim_features // 2)
    elif feature_type == "random_uniform":
        features = generate_random_features(
            n_nodes,
            dim_features=dim_features,
            feature_noise_sigma=feature_noise_sigma,
        )
        meta_num_freq = None
    else:
        raise ValueError("feature_type must be one of: sinusoidal, random_uniform")

    output_dir = f"dataset/ws/{name}"
    print(f"[4/4] Saving to {output_dir}...")
    save_to_ogb_format(G, labels, features, output_dir)

    print("      Computing graph metrics...")
    avg_clustering = nx.average_clustering(G)
    avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    metadata = {
        "name": name,
        "n_nodes": n_nodes,
        "k": k,
        "p": p,
        "labeling": labeling,
        "n_classes_requested": n_classes,
        "n_classes": len(unique),
        "segments": segments,
        "n_segments": len(segments),
        "segment_size": segment_size,
        "majority_frac": majority_frac,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "dim_features": dim_features,
        "feature_type": feature_type,
        "num_frequencies": meta_num_freq,
        "feature_noise_sigma": feature_noise_sigma,
        "seed": seed,
        "n_edges": G.number_of_edges(),
        "avg_clustering": avg_clustering,
        "avg_degree": avg_degree,
        "label_counts": label_counts,
    }
    save_metadata(metadata, output_dir)

    print(
        f"Done! {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"{len(unique)} classes, features={features.shape}"
    )
    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic WS graph with ring-based labels and sinusoidal features"
    )
    parser.add_argument("--n_nodes", type=int, default=1000, help="Number of nodes")
    parser.add_argument(
        "--k", type=int, default=10, help="Each node is joined with its k nearest neighbors in the ring"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.1,
        help="Rewiring probability (controls small-world randomness)",
    )
    parser.add_argument(
        "--labeling",
        choices=["equal", "highfreq", "imbalanced"],
        default="equal",
        help="Label assignment pattern along the ring",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=3,
        help="Number of classes for equal/highfreq labeling",
    )
    parser.add_argument(
        "--n_segments",
        type=int,
        default=None,
        help="Number of segments for highfreq labeling (overrides default)",
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        default=None,
        help="Target segment size for highfreq labeling (overrides n_segments if set)",
    )
    parser.add_argument(
        "--majority_frac",
        type=float,
        default=0.8,
        help="Fraction of nodes in majority class for imbalanced labeling",
    )
    parser.add_argument(
        "--majority_class", type=int, default=0, help="Label for majority segment"
    )
    parser.add_argument(
        "--minority_class", type=int, default=1, help="Label for minority segment"
    )
    parser.add_argument(
        "--dim_features", type=int, default=128, help="Feature dimension (will truncate/pad sinusoids)"
    )
    parser.add_argument(
        "--num_frequencies",
        type=int,
        default=None,
        help="Number of sinusoidal frequencies (defaults to dim_features//2)",
    )
    parser.add_argument(
        "--feature_noise_sigma",
        type=float,
        default=0.0,
        help="Gaussian noise level added to generated features",
    )
    parser.add_argument(
        "--feature_type",
        choices=["sinusoidal", "random_uniform"],
        default="sinusoidal",
        help="Feature generation type",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name (e.g., ws001). Auto-increment if not provided.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_ws_dataset(
        name=args.name,
        n_nodes=args.n_nodes,
        k=args.k,
        p=args.p,
        labeling=args.labeling,
        n_classes=args.n_classes,
        n_segments=args.n_segments,
        segment_size=args.segment_size,
        majority_frac=args.majority_frac,
        majority_class=args.majority_class,
        minority_class=args.minority_class,
        dim_features=args.dim_features,
        num_frequencies=args.num_frequencies,
        feature_type=args.feature_type,
        feature_noise_sigma=args.feature_noise_sigma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
