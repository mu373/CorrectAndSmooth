"""
Hyperparameter presets for custom datasets by category (prefix).

Each category can have tuned hyperparameters for different methods.
Unknown categories fall back to '_default' (arxiv-style parameters).
"""

# Preset hyperparameters by dataset category
CUSTOM_PRESETS = {
    # Barabasi-Albert graphs
    "ba": {
        "plain": {
            "alpha1": 0.9,
            "alpha2": 0.8,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "linear": {
            "alpha1": 0.95,
            "alpha2": 0.7,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "mlp": {
            "alpha1": 0.95,
            "alpha2": 0.75,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "gat": {
            "alpha": 0.8,
            "num_propagations": 50,
        },
        "lp": {
            "alpha": 0.9,
            "num_propagations": 50,
        },
        "fn": "double_correlation_autoscale",
    },
    # Watts-Strogatz graphs
    "ws": {
        "plain": {
            "alpha1": 0.9,
            "alpha2": 0.8,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "linear": {
            "alpha1": 0.95,
            "alpha2": 0.7,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "mlp": {
            "alpha1": 0.95,
            "alpha2": 0.75,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "gat": {
            "alpha": 0.8,
            "num_propagations": 50,
        },
        "lp": {
            "alpha": 0.9,
            "num_propagations": 50,
        },
        "fn": "double_correlation_autoscale",
    },
    # Default fallback (arxiv-style parameters)
    "_default": {
        "plain": {
            "alpha1": 0.87,
            "alpha2": 0.81,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "linear": {
            "alpha1": 0.98,
            "alpha2": 0.65,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "mlp": {
            "alpha1": 0.979,
            "alpha2": 0.756,
            "num_propagations1": 50,
            "num_propagations2": 50,
        },
        "gat": {
            "alpha": 0.8,
            "num_propagations": 50,
        },
        "lp": {
            "alpha": 0.9,
            "num_propagations": 50,
        },
        "fn": "double_correlation_autoscale",
    },
}


def get_preset(prefix, method=None):
    """
    Get hyperparameter preset for a dataset category.

    Args:
        prefix: Dataset category (e.g., 'ba', 'ws')
        method: Optional method name (e.g., 'mlp', 'linear')

    Returns:
        If method is None: entire preset dict for the category
        If method is specified: preset dict for that method
    """
    category = CUSTOM_PRESETS.get(prefix, CUSTOM_PRESETS["_default"])
    if method is None:
        return category
    return category.get(method, CUSTOM_PRESETS["_default"].get(method))


def get_fn_name(prefix):
    """Get the correlation function name for a category."""
    category = CUSTOM_PRESETS.get(prefix, CUSTOM_PRESETS["_default"])
    return category.get("fn", "double_correlation_autoscale")
