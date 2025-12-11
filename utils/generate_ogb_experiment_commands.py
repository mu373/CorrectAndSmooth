"""Generate commands for OGB (arxiv, products) experiment runs.

Usage:
    python utils/generate_ogb_experiment_commands.py

    # Then run:
    bash ogb_experiment_commands.txt
    # Or parallel:
    parallel -j 4 < ogb_experiment_commands.txt
"""

import itertools

# Experiment parameters
DATASETS = ["arxiv", "products"]
NORM_STYLES = ["symmetric", "left", "right"]
ADJACENCIES = ["standard", "signless_laplacian"]
METHOD = "mlp"


def generate_command(dataset, norm_style, adjacency):
    """Generate a run_experiments command."""
    return f"python run_experiments.py --dataset {dataset} --method {METHOD} --norm-style {norm_style} --adjacency {adjacency}"


def main():
    lines = []

    # Header
    lines.append("# OGB Experiment Commands")
    lines.append(f"# Datasets: {', '.join(DATASETS)}")
    lines.append(f"# Norm styles: {', '.join(NORM_STYLES)}")
    lines.append(f"# Adjacencies: {', '.join(ADJACENCIES)}")
    lines.append(f"# Method: {METHOD}")
    lines.append("")

    # Generate all combinations
    for dataset, norm_style, adjacency in itertools.product(DATASETS, NORM_STYLES, ADJACENCIES):
        lines.append(generate_command(dataset, norm_style, adjacency))

    # Write to file
    output_file = "ogb_experiment_commands.txt"
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    num_commands = len(DATASETS) * len(NORM_STYLES) * len(ADJACENCIES)
    print(f"Generated {num_commands} commands")
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Norm styles: {', '.join(NORM_STYLES)}")
    print(f"  Adjacencies: {', '.join(ADJACENCIES)}")
    print(f"\nSaved to: {output_file}")
    print("\nRun with:")
    print(f"  bash {output_file}")
    print(f"  # or parallel: parallel -j 4 < {output_file}")


if __name__ == "__main__":
    main()
