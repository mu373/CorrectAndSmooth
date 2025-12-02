"""
Node weighting strategies (normalizers) for graph adjacency matrices.

These determine how edges are scaled based on node properties like degree or PageRank.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import torch


def safe_inverse(x: torch.Tensor, power: float = -1.0, eps: float = 1e-10) -> torch.Tensor:
    """Compute x^power safely, handling zeros and infinities."""
    result = (x + eps).pow(power)
    result[result.isinf()] = 0
    return result


@dataclass
class NormalizationResult:
    """Result of computing normalization weights for a graph."""
    left_norm: torch.Tensor      # For C^{-1} A (left multiplication)
    right_norm: torch.Tensor     # For A C^{-1} (right multiplication)
    symmetric_norm: torch.Tensor # For C^{-1/2} A C^{-1/2}
    metadata: Dict[str, Any]     # Raw values for debugging

    def to(self, device: torch.device) -> 'NormalizationResult':
        """Move all tensors to specified device."""
        return NormalizationResult(
            left_norm=self.left_norm.to(device),
            right_norm=self.right_norm.to(device),
            symmetric_norm=self.symmetric_norm.to(device),
            metadata=self.metadata
        )


class BaseNormalizer(ABC):
    """Abstract base class for node weighting strategies."""

    @abstractmethod
    def compute_node_weights(self, adj: torch.sparse.Tensor) -> NormalizationResult:
        """
        Compute normalization weights for all nodes.

        Args:
            adj: Sparse adjacency matrix

        Returns:
            NormalizationResult with left, right, and symmetric normalization vectors
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this normalizer for caching."""
        pass


class DegreeNormalizer(BaseNormalizer):
    """
    Degree-based normalization (default).

    Uses node degree as the weighting factor:
    - Left norm: D^{-1}
    - Right norm: D^{-1}
    - Symmetric norm: D^{-1/2}
    """

    def compute_node_weights(self, adj: torch.sparse.Tensor) -> NormalizationResult:
        # Compute degree for each node
        deg = torch.sparse.sum(adj, dim=1).to_dense().float()

        # Handle isolated nodes by setting their centrality to min(non-zero) * 0.01
        nonzero_mask = deg > 0
        if nonzero_mask.any() and (~nonzero_mask).any():
            min_nonzero = deg[nonzero_mask].min()
            deg[~nonzero_mask] = min_nonzero * 0.01

        deg_inv = safe_inverse(deg, power=-1.0)
        deg_inv_sqrt = safe_inverse(deg, power=-0.5)

        return NormalizationResult(
            left_norm=deg_inv,
            right_norm=deg_inv,
            symmetric_norm=deg_inv_sqrt,
            metadata={'degree': deg}
        )

    @property
    def name(self) -> str:
        return "degree"


class PageRankNormalizer(BaseNormalizer):
    """
    PageRank-based normalization using power iteration.

    Uses PageRank scores as the weighting factor instead of degree.
    This can give better results for graphs with hub nodes.

    Args:
        damping: Damping factor (default 0.85)
        max_iter: Maximum number of power iterations (default 100)
        tol: Convergence tolerance (default 1e-6)
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def compute_node_weights(self, adj: torch.sparse.Tensor) -> NormalizationResult:
        N = adj.shape[0]
        device = adj.device

        # Build row-stochastic matrix P = D^{-1} A
        deg = torch.sparse.sum(adj, dim=1).to_dense().float()
        deg_inv = safe_inverse(deg, power=-1.0)

        # Scale adjacency by inverse degree (row-stochastic)
        indices = adj.indices()
        row, col = indices[0], indices[1]
        values = adj.values() * deg_inv[row]
        P = torch.sparse_coo_tensor(indices, values, adj.shape, device=device).coalesce()

        # Power iteration for PageRank: pr = (1-d)/N + d * P^T @ pr
        pr = torch.ones(N, device=device) / N
        teleport = (1 - self.damping) / N

        for _ in range(self.max_iter):
            pr_old = pr.clone()
            # P^T @ pr = sum of pr[j] * P[j,i] for all j -> i
            # Since P[j,i] = A[j,i] / deg[j], this is random walk step
            pr = teleport + self.damping * torch.sparse.mm(P.t(), pr.unsqueeze(1)).squeeze()

            if (pr - pr_old).abs().max() < self.tol:
                break

        # Handle isolated nodes
        nonzero_mask = pr > 0
        if nonzero_mask.any() and (~nonzero_mask).any():
            min_nonzero = pr[nonzero_mask].min()
            pr[~nonzero_mask] = min_nonzero * 0.01

        pr_inv = safe_inverse(pr, power=-1.0)
        pr_inv_sqrt = safe_inverse(pr, power=-0.5)

        return NormalizationResult(
            left_norm=pr_inv,
            right_norm=pr_inv,
            symmetric_norm=pr_inv_sqrt,
            metadata={'pagerank': pr, 'iterations': _}
        )

    @property
    def name(self) -> str:
        return f"pagerank_d{self.damping}"


# Registry for easy lookup by name
NORMALIZERS = {
    'degree': DegreeNormalizer,
    'pagerank': PageRankNormalizer,
}


def get_normalizer(name: str, **kwargs) -> BaseNormalizer:
    """
    Get a normalizer by name.

    Args:
        name: Name of the normalizer ('degree', 'pagerank')
        **kwargs: Additional arguments for the normalizer

    Returns:
        BaseNormalizer instance
    """
    if name not in NORMALIZERS:
        raise ValueError(f"Unknown normalizer: {name}. Available: {list(NORMALIZERS.keys())}")
    return NORMALIZERS[name](**kwargs)
