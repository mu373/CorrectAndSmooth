#!/usr/bin/env python3
"""Visualize a Watts-Strogatz dataset as a colored ring by node labels.

Example:
    python generate_ws_vis.py --dataname ws001
"""

import argparse
import gzip
import os
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_labels(path: str) -> np.ndarray:
    """Read node-label.csv(.gz) -> array of ints."""
    with gzip.open(path, "rt") as f:
        labels = [int(line.strip()) for line in f if line.strip() != ""]
    return np.asarray(labels, dtype=int)


def read_edges(path: str) -> np.ndarray:
    """Read edge.csv(.gz) -> array of (src, dst)."""
    edges = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src_str, dst_str = line.split(",")
            edges.append((int(src_str), int(dst_str)))
    return np.asarray(edges, dtype=int)


def ring_positions(n_nodes: int) -> np.ndarray:
    """Place nodes evenly on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def build_color_map(labels: Iterable[int], cmap_name: str = "tab20") -> Dict[int, Tuple[float, ...]]:
    """Map each label to a consistent color (wraps if labels > colormap size)."""
    unique_labels = sorted(set(int(l) for l in labels))
    cmap = plt.get_cmap(cmap_name)
    return {label: cmap(label % cmap.N) for label in unique_labels}


def plot_ring(
    labels: np.ndarray,
    edges: np.ndarray,
    output_path: str,
    title: str,
    max_edges: int,
    point_size: float,
) -> None:
    """Render the ring plot and save to disk."""
    n_nodes = len(labels)
    pos = ring_positions(n_nodes)
    label_to_color = build_color_map(labels)
    node_colors = np.asarray([label_to_color[int(lbl)] for lbl in labels])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    if edges.size > 0:
        if max_edges is not None and len(edges) > max_edges:
            rng = np.random.default_rng(0)  # deterministic edge sampling
            idx = rng.choice(len(edges), size=max_edges, replace=False)
            edges_to_draw = edges[idx]
        else:
            edges_to_draw = edges
        for src, dst in edges_to_draw:
            xs = [pos[src, 0], pos[dst, 0]]
            ys = [pos[src, 1], pos[dst, 1]]
            ax.plot(xs, ys, color="lightgray", linewidth=0.3, alpha=0.35, zorder=1)

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=node_colors,
        s=point_size,
        edgecolors="none",
        zorder=2,
    )

    # Legend for up to 20 labels to avoid clutter.
    legend_items = list(label_to_color.items())
    if len(legend_items) <= 20:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                color=color,
                label=str(label),
            )
            for label, color in legend_items
        ]
        ax.legend(handles=handles, title="Label", loc="upper right", fontsize=7, title_fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize WS graph as a ring colored by labels.")
    parser.add_argument("--dataname", required=True, help="Dataset name (e.g., ws001).")
    parser.add_argument(
        "--base_dir",
        default="dataset/ws",
        help="Root directory containing the dataset (default: dataset/ws).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to dataset/ws/{dataname}/vis.png.",
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=4000,
        help="Draw at most this many edges to keep the plot readable (None for all).",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=4.0,
        help="Marker size for nodes.",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(args.base_dir, args.dataname)
    edge_path = os.path.join(dataset_dir, "edge.csv.gz")
    label_path = os.path.join(dataset_dir, "node-label.csv.gz")

    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing edge file: {edge_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label file: {label_path}")

    labels = read_labels(label_path)
    edges = read_edges(edge_path)
    if edges.ndim == 1 and edges.size == 0:
        edges = np.empty((0, 2), dtype=int)

    output_path = args.output or os.path.join(dataset_dir, "vis.png")
    title = f"{args.dataname} (n={len(labels)}, edges={len(edges)}, labels={len(set(labels))})"

    plot_ring(
        labels=labels,
        edges=edges,
        output_path=output_path,
        title=title,
        max_edges=args.max_edges,
        point_size=args.point_size,
    )
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
