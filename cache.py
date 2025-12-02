"""
Caching utilities for computed matrices.

Avoids recomputation of expensive matrix operations like PageRank or 2-hop adjacency.
"""

import hashlib
import os
from typing import Any, Optional, Tuple
import torch


class MatrixCache:
    """
    Cache for computed adjacency and normalization matrices.

    Caches are keyed by a hash of the edge structure and configuration parameters.
    """

    def __init__(self, cache_dir: str = 'cache/matrices'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _hash_config(
        self,
        edge_index: torch.Tensor,
        normalizer_name: str,
        adjacency_name: str,
        params: dict
    ) -> str:
        """Generate a hash key from edge structure and configuration."""
        # Create a deterministic representation of the edge structure
        # Using shape and sum as a cheap fingerprint
        edge_fingerprint = f"{edge_index.shape}_{edge_index.sum().item()}"

        # Sort params for deterministic ordering
        sorted_params = sorted(params.items()) if params else []
        params_str = str(sorted_params)

        key = f"{edge_fingerprint}_{normalizer_name}_{adjacency_name}_{params_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(
        self,
        edge_index: torch.Tensor,
        normalizer_name: str,
        adjacency_name: str,
        params: Optional[dict] = None
    ) -> Optional[Tuple[torch.Tensor, Any]]:
        """
        Retrieve cached matrices if available.

        Args:
            edge_index: Edge index tensor for fingerprinting
            normalizer_name: Name of the normalizer used
            adjacency_name: Name of the adjacency type used
            params: Additional parameters that affect the result

        Returns:
            Cached data tuple (adj, norm_result) if found, None otherwise
        """
        if params is None:
            params = {}

        cache_key = self._hash_config(edge_index, normalizer_name, adjacency_name, params)
        path = os.path.join(self.cache_dir, f"{cache_key}.pt")

        if os.path.exists(path):
            try:
                return torch.load(path, weights_only=False)
            except Exception as e:
                print(f"Warning: Failed to load cache from {path}: {e}")
                return None
        return None

    def put(
        self,
        edge_index: torch.Tensor,
        normalizer_name: str,
        adjacency_name: str,
        data: Tuple[torch.Tensor, Any],
        params: Optional[dict] = None
    ) -> None:
        """
        Store computed matrices in cache.

        Args:
            edge_index: Edge index tensor for fingerprinting
            normalizer_name: Name of the normalizer used
            adjacency_name: Name of the adjacency type used
            data: Tuple of (adj, norm_result) to cache
            params: Additional parameters that affect the result
        """
        if params is None:
            params = {}

        cache_key = self._hash_config(edge_index, normalizer_name, adjacency_name, params)
        path = os.path.join(self.cache_dir, f"{cache_key}.pt")

        try:
            torch.save(data, path)
        except Exception as e:
            print(f"Warning: Failed to save cache to {path}: {e}")

    def clear(self) -> int:
        """
        Clear all cached matrices.

        Returns:
            Number of cache files deleted
        """
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pt'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
                except Exception:
                    pass
        return count


# Global cache instance
_global_cache: Optional[MatrixCache] = None


def get_cache(cache_dir: str = 'cache/matrices') -> MatrixCache:
    """Get or create the global matrix cache."""
    global _global_cache
    if _global_cache is None or _global_cache.cache_dir != cache_dir:
        _global_cache = MatrixCache(cache_dir)
    return _global_cache
