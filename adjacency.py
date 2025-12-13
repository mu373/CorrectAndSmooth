"""
Adjacency matrix transformations that change the graph structure.

These define different ways to connect nodes beyond standard adjacency.
"""

from abc import ABC, abstractmethod
import torch


class BaseAdjacency(ABC):
    """Abstract base class for adjacency transformations."""

    @abstractmethod
    def transform(self, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        """
        Transform the adjacency matrix.

        Args:
            adj: Sparse adjacency matrix

        Returns:
            Transformed sparse adjacency matrix
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this adjacency type for caching."""
        pass


class StandardAdjacency(BaseAdjacency):
    """Standard adjacency matrix A (default, no transformation)."""

    def transform(self, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        return adj

    @property
    def name(self) -> str:
        return "standard"


class TwoHopAdjacency(BaseAdjacency):
    """
    2-hop adjacency: A² (neighbors of neighbors).

    Note: A² may be much denser than A for large graphs.
    """

    def transform(self, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        # A² = A @ A
        result = torch.sparse.mm(adj, adj).coalesce()
        return result

    @property
    def name(self) -> str:
        return "2hop"


class SignlessLaplacian(BaseAdjacency):
    """
    Signless Laplacian: D + A.

    Adds degree to the diagonal, emphasizing high-degree nodes.
    """

    def transform(self, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        N = adj.shape[0]
        device = adj.device

        # Compute degree
        deg = torch.sparse.sum(adj, dim=1).to_dense().float()

        # Create diagonal matrix with degrees
        diag_indices = torch.stack([torch.arange(N, device=device),
                                    torch.arange(N, device=device)])
        diag = torch.sparse_coo_tensor(diag_indices, deg, adj.shape, device=device)

        # D + A
        return (adj + diag).coalesce()

    @property
    def name(self) -> str:
        return "signless_laplacian"


class Laplacian(BaseAdjacency):
    """
    Laplacian: D - A.

    Standard graph Laplacian matrix.
    """

    def transform(self, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        N = adj.shape[0]
        device = adj.device

        # Compute degree
        deg = torch.sparse.sum(adj, dim=1).to_dense().float()

        # Create diagonal matrix with degrees
        diag_indices = torch.stack([torch.arange(N, device=device),
                                    torch.arange(N, device=device)])
        diag = torch.sparse_coo_tensor(diag_indices, deg, adj.shape, device=device)

        # D - A
        return (diag - adj).coalesce()

    @property
    def name(self) -> str:
        return "laplacian"


class KatzAdjacency(BaseAdjacency):
    """
    Katz adjacency: truncated series βA + β²A² + ... + β^k A^k.

    Captures multi-hop connections with exponential decay.

    Args:
        beta: Decay factor (default 0.1). Should be < 1/λ_max(A) for convergence.
        k: Number of hops to include (default 3)
    """

    def __init__(self, beta: float = 0.1, k: int = 3):
        self.beta = beta
        self.k = k

    def transform(self, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        device = adj.device
        N = adj.shape[0]

        # Start with βA
        A_power = adj
        result_values = self.beta * adj.values()
        result_indices = adj.indices().clone()

        # Accumulate β²A² + β³A³ + ...
        for i in range(2, self.k + 1):
            A_power = torch.sparse.mm(A_power, adj).coalesce()
            weight = self.beta ** i

            # Merge indices and values
            new_indices = A_power.indices()
            new_values = weight * A_power.values()

            result_indices = torch.cat([result_indices, new_indices], dim=1)
            result_values = torch.cat([result_values, new_values])

        # Create combined tensor and coalesce to sum duplicate indices
        result = torch.sparse_coo_tensor(result_indices, result_values, (N, N), device=device)
        return result.coalesce()

    @property
    def name(self) -> str:
        return f"katz_b{self.beta}_k{self.k}"


# Registry for easy lookup by name
ADJACENCY_TYPES = {
    'standard': StandardAdjacency,
    '2hop': TwoHopAdjacency,
    'signless_laplacian': SignlessLaplacian,
    'laplacian': Laplacian,
    'katz': KatzAdjacency,
}


def get_adjacency(name: str, **kwargs) -> BaseAdjacency:
    """
    Get an adjacency transformer by name.

    Args:
        name: Name of the adjacency type ('standard', '2hop', 'signless_laplacian', 'laplacian', 'katz')
        **kwargs: Additional arguments for the transformer

    Returns:
        BaseAdjacency instance
    """
    if name not in ADJACENCY_TYPES:
        raise ValueError(f"Unknown adjacency type: {name}. Available: {list(ADJACENCY_TYPES.keys())}")
    return ADJACENCY_TYPES[name](**kwargs)
