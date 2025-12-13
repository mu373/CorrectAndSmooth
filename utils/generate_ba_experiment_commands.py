"""Generate commands for BA graph experiment runs.

Usage:
    python utils/generate_ba_experiment_commands.py --start 1 --end 85
    # Creates ba_experiment_commands.txt for ba001 to ba085

    python utils/generate_ba_experiment_commands.py --indices 1 2 3 10 20
    # Creates commands for specific indices

    python utils/generate_ba_experiment_commands.py --start 1 --end 85 --model linear
    # Use a different model (default: mlp, choices: mlp, linear, plain)

    # Then run:
    bash ba_experiment_commands.txt
    # Or parallel:
    parallel -j 4 < ba_experiment_commands.txt
"""

import argparse

# Experiment parameters
DEFAULT_MODEL = "mlp"
MODEL_CHOICES = ["mlp", "linear", "plain"]
EPOCHS = 300


def generate_commands(dataname, model):
    """Generate gen_models and run_experiments commands for a dataset."""
    return [
        f"python gen_models.py --dataset custom --model {model} --epochs {EPOCHS} --dataname {dataname}",
        f"python run_experiments.py --dataset custom --method {model} --dataname {dataname}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate BA experiment commands")
    parser.add_argument("--start", type=int, help="Start index (e.g., 1 for ba001)")
    parser.add_argument("--end", type=int, help="End index inclusive (e.g., 85 for ba085)")
    parser.add_argument("--indices", type=int, nargs="+", help="Specific indices (e.g., 1 2 3 10 20)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=MODEL_CHOICES,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # Determine indices
    if args.indices:
        indices = args.indices
    elif args.start and args.end:
        indices = list(range(args.start, args.end + 1))
    else:
        parser.error("Provide either --start/--end or --indices")

    model = args.model
    lines = []

    # Header
    lines.append("# BA Graph Experiment Commands")
    lines.append(f"# Model: {model}, Epochs: {EPOCHS}")
    lines.append(f"# Datasets: ba{indices[0]:03d} to ba{indices[-1]:03d} ({len(indices)} datasets)")
    lines.append("")

    for idx in indices:
        dataname = f"ba{idx:03d}"
        lines.append(f"# {dataname}")
        for cmd in generate_commands(dataname, model):
            lines.append(cmd)

    # Write to file
    output_file = "ba_experiment_commands.txt"
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Generated {len(indices) * 2} commands ({len(indices)} datasets × 2 commands each)")
    print(f"  Datasets: ba{indices[0]:03d} to ba{indices[-1]:03d}")
    print(f"\nSaved to: {output_file}")
    print("\nRun with:")
    print(f"  bash {output_file}")
    print(f"  # or parallel: parallel -j 4 < {output_file}")


if __name__ == "__main__":
    main()
