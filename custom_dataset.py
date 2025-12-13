"""
Custom dataset loader for non-OGB datasets.

Usage:
    dataset = CustomDataset('ba001')  # loads from dataset/ba/ba001/

Expected directory structure:
    dataset/{prefix}/{dataname}/
    ├── edge.csv(.gz)         # source,target pairs, no header
    ├── node-feat.csv(.gz)    # node features, comma-separated
    ├── node-label.csv(.gz)   # one integer label per line
    └── split/                # OPTIONAL - auto-generated if missing
        ├── train.csv(.gz)
        ├── valid.csv(.gz)
        └── test.csv(.gz)
"""

import os
import re
import gzip
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy import sparse


def get_prefix(dataname):
    """Extract non-numeric prefix from dataname.

    Examples:
        'ba001' -> 'ba'
        'ws_graph_123' -> 'ws_graph_'
    """
    match = re.match(r"^([a-zA-Z_]+)", dataname)
    return match.group(1) if match else dataname


def read_csv(filepath):
    """Read CSV file, supporting both .csv and .csv.gz formats."""
    if os.path.exists(filepath + ".gz"):
        filepath = filepath + ".gz"
    elif not os.path.exists(filepath):
        return None

    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt") as f:
            lines = f.readlines()
    else:
        with open(filepath, "r") as f:
            lines = f.readlines()

    return [line.strip() for line in lines if line.strip()]


class CustomDataset:
    """
    Custom dataset loader mimicking PygNodePropPredDataset interface.

    Args:
        dataname: Dataset name (e.g., 'ba001')
        root: Root directory for datasets (default: 'dataset')
        split_ratio: Tuple of (train, valid, test) ratios if auto-generating splits
        seed: Random seed for reproducible split generation
    """

    def __init__(self, dataname, root="dataset", split_ratio=(0.6, 0.2, 0.2), seed=42):
        self.dataname = dataname
        self.prefix = get_prefix(dataname)
        self.root = root
        self.split_ratio = split_ratio
        self.seed = seed

        # Build path: dataset/{prefix}/{dataname}/
        self.data_dir = os.path.join(root, self.prefix, dataname)

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        # Load data
        self._data = self._load_data()
        self._split_idx = self._load_or_generate_splits()
        self._num_classes = int(self._data.y.max().item()) + 1

    def _load_data(self):
        """Load graph data from CSV files."""
        # Load edges
        edge_lines = read_csv(os.path.join(self.data_dir, "edge.csv"))
        if edge_lines is None:
            raise FileNotFoundError(f"edge.csv not found in {self.data_dir}")

        edges = []
        for line in edge_lines:
            src, dst = line.split(",")
            edges.append([int(src), int(dst)])
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Load node features
        feat_lines = read_csv(os.path.join(self.data_dir, "node-feat.csv"))
        if feat_lines is None:
            raise FileNotFoundError(f"node-feat.csv not found in {self.data_dir}")

        features = []
        for line in feat_lines:
            features.append([float(x) for x in line.split(",")])
        features = torch.tensor(features, dtype=torch.float)

        # Load labels
        label_lines = read_csv(os.path.join(self.data_dir, "node-label.csv"))
        if label_lines is None:
            raise FileNotFoundError(f"node-label.csv not found in {self.data_dir}")

        labels = torch.tensor([int(x) for x in label_lines], dtype=torch.long)
        # OGB format: [N, 1]
        labels = labels.unsqueeze(1)

        num_nodes = features.shape[0]

        # Validate
        assert labels.shape[0] == num_nodes, (
            f"Number of labels ({labels.shape[0]}) != number of nodes ({num_nodes})"
        )

        return Data(x=features, y=labels, edge_index=edges, num_nodes=num_nodes)

    def _load_or_generate_splits(self):
        """Load splits from files or auto-generate if missing."""
        split_dir = os.path.join(self.data_dir, "split")

        train_lines = read_csv(os.path.join(split_dir, "train.csv"))
        valid_lines = read_csv(os.path.join(split_dir, "valid.csv"))
        test_lines = read_csv(os.path.join(split_dir, "test.csv"))

        if (
            train_lines is not None
            and valid_lines is not None
            and test_lines is not None
        ):
            # Load from files
            train_idx = torch.tensor([int(x) for x in train_lines], dtype=torch.long)
            valid_idx = torch.tensor([int(x) for x in valid_lines], dtype=torch.long)
            test_idx = torch.tensor([int(x) for x in test_lines], dtype=torch.long)
            print(f"Loaded splits from {split_dir}")
        else:
            # Auto-generate splits
            train_idx, valid_idx, test_idx = self._generate_splits()
            print(
                f"Auto-generated splits: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
            )

        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def _generate_splits(self):
        """Generate random train/valid/test splits."""
        num_nodes = self._data.num_nodes
        indices = np.arange(num_nodes)

        # Use fixed seed for reproducibility
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)

        train_ratio, valid_ratio, test_ratio = self.split_ratio
        train_end = int(num_nodes * train_ratio)
        valid_end = train_end + int(num_nodes * valid_ratio)

        train_idx = torch.tensor(sorted(indices[:train_end]), dtype=torch.long)
        valid_idx = torch.tensor(sorted(indices[train_end:valid_end]), dtype=torch.long)
        test_idx = torch.tensor(sorted(indices[valid_end:]), dtype=torch.long)

        return train_idx, valid_idx, test_idx

    def __getitem__(self, idx):
        """Return the graph data (mimics PygNodePropPredDataset)."""
        assert idx == 0, "CustomDataset only contains one graph"
        return self._data

    def __len__(self):
        return 1

    def get_idx_split(self):
        """Return train/valid/test split indices."""
        return self._split_idx

    @property
    def num_classes(self):
        """Return number of classes (auto-detected from labels)."""
        return self._num_classes


def preprocess_custom(data, dataname, preprocess_type="spectral", num_propagations=10, p=None, alpha=None, use_cache=True):
    """
    Preprocess embeddings for custom datasets.

    Saves to: embeddings/{prefix}/{dataname}-{preprocess_type}.pt
    Example:  embeddings/ba/ba001-spectral.pt

    Args:
        data: PyG Data object
        dataname: Dataset name (e.g., 'ba001')
        preprocess_type: Type of preprocessing ('spectral', 'diffusion')
        num_propagations: Number of propagation steps for diffusion
        p: Power parameter for diffusion
        alpha: Alpha parameter for diffusion
        use_cache: Whether to use cached embeddings if available

    Returns:
        Embedding tensor
    """
    prefix = get_prefix(dataname)
    embed_dir = os.path.join("embeddings", prefix)
    embed_path = os.path.join(embed_dir, f"{dataname}-{preprocess_type}.pt")

    # Check cache
    if use_cache and os.path.exists(embed_path):
        print(f"Using cache: {embed_path}")
        return torch.load(embed_path)

    print(f"Computing {preprocess_type} embeddings for {dataname}...")
    os.makedirs(embed_dir, exist_ok=True)

    if preprocess_type == "spectral":
        result = _spectral_embedding(data)
    elif preprocess_type == "diffusion":
        result = _diffusion_embedding(data, num_propagations, p, alpha)
    else:
        raise ValueError(f"Unknown preprocess_type: {preprocess_type}")

    torch.save(result, embed_path)
    print(f"Saved embeddings to: {embed_path}")
    return result


def _spectral_embedding(data, k=128):
    """Compute spectral embeddings for a graph."""
    from norm_spec import spectral_embedding

    edge_index = to_undirected(data.edge_index, data.num_nodes)
    N = data.num_nodes
    row, col = edge_index
    adj = sparse.csr_matrix(
        (np.ones(row.shape[0]), (row.numpy(), col.numpy())), shape=(N, N)
    )
    embedding = spectral_embedding(adj, k=k)
    return torch.tensor(embedding).float()


def _diffusion_embedding(data, num_propagations=10, p=None, alpha=None):
    """Compute diffusion embeddings for a graph."""
    from tqdm import tqdm

    if p is None:
        p = 1.0
    if alpha is None:
        alpha = 0.5

    N = data.num_nodes
    edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = edge_index
    adj = sparse.csr_matrix(
        (np.ones(row.shape[0]), (row.numpy(), col.numpy())), shape=(N, N)
    )
    adj = adj + sparse.eye(N)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    adj = sparse.diags(deg_inv_sqrt) @ adj @ sparse.diags(deg_inv_sqrt)
    adj = adj.tocsr()

    x = data.x.numpy()
    x = x ** p
    for _ in tqdm(range(num_propagations)):
        x = x - alpha * (sparse.eye(adj.shape[0]) - adj) @ x
        x = x ** p

    return torch.from_numpy(x).float()
