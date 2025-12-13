"""Generate commands for WS graph experiment runs.

Usage:
    python utils/generate_ws_experiment_commands.py --start 1 --end 50
    # Creates ws_experiment_commands.txt for ws001 to ws050

    python utils/generate_ws_experiment_commands.py --indices 1 2 3 10 20
    # Creates commands for specific indices

    python utils/generate_ws_experiment_commands.py --start 1 --end 50 --methods mlp linear plain
    # Emit runs for multiple methods

    python utils/generate_ws_experiment_commands.py --start 1 --end 50 --norm-styles symmetric left --adjacencies standard laplacian
    # Emit run_experiments commands for each norm_style/adjacency combo (defaults if not provided)

    # Then run:
    bash ws_experiment_commands.txt
    # Or parallel:
    parallel -j 4 < ws_experiment_commands.txt
"""

import argparse

# Experiment parameters
DEFAULT_MODEL = "mlp"
MODEL_CHOICES = ["mlp", "linear", "plain"]
EPOCHS = 300
NORM_STYLE_CHOICES = ["symmetric", "left", "right"]
ADJACENCY_CHOICES = ["standard", "2hop", "signless_laplacian", "laplacian", "katz"]


def main():
    parser = argparse.ArgumentParser(description="Generate WS experiment commands")
    parser.add_argument("--start", type=int, help="Start index (e.g., 1 for ws001)")
    parser.add_argument("--end", type=int, help="End index inclusive (e.g., 50 for ws050)")
    parser.add_argument("--indices", type=int, nargs="+", help="Specific indices (e.g., 1 2 3 10 20)")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=MODEL_CHOICES,
        help="List of methods to run (default: uses --model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ws_experiment_commands.txt",
        help="Output filename for generated commands",
    )
    parser.add_argument(
        "--skip-gen-models",
        action="store_true",
        help="If set, only emit run_experiments commands (assumes models already trained)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=MODEL_CHOICES,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--norm-styles",
        type=str,
        nargs="+",
        choices=NORM_STYLE_CHOICES,
        default=None,
        help="Normalization styles to sweep (default: run_experiments default, symmetric)",
    )
    parser.add_argument(
        "--adjacencies",
        type=str,
        nargs="+",
        choices=ADJACENCY_CHOICES,
        default=None,
        help="Adjacency variants to sweep (default: run_experiments default, standard)",
    )
    args = parser.parse_args()

    # Determine indices
    if args.indices:
        indices = args.indices
    elif args.start and args.end:
        indices = list(range(args.start, args.end + 1))
    else:
        parser.error("Provide either --start/--end or --indices")

    methods = args.methods if args.methods else [args.model]
    norm_styles = args.norm_styles if args.norm_styles else [None]
    adjacencies = args.adjacencies if args.adjacencies else [None]
    lines = []

    # Header
    lines.append("# WS Graph Experiment Commands")
    lines.append(f"# Methods: {methods}, Epochs: {EPOCHS}")
    lines.append(f"# Norm styles: {norm_styles if norm_styles != [None] else 'default (symmetric)'}")
    lines.append(f"# Adjacencies: {adjacencies if adjacencies != [None] else 'default (standard)'}")
    lines.append(f"# Skip gen_models: {args.skip_gen_models}")
    lines.append(f"# Datasets: ws{indices[0]:03d} to ws{indices[-1]:03d} ({len(indices)} datasets)")
    lines.append("")

    for idx in indices:
        dataname = f"ws{idx:03d}"
        lines.append(f"# {dataname}")
        for method in methods:
            lines.append(f"# {dataname} - {method}")
            # Train once per method (unless skipped)
            if not args.skip_gen_models:
                lines.append(
                    f"python gen_models.py --dataset custom --model {method} --epochs {EPOCHS} --dataname {dataname}"
                )
            # One run_experiments per norm-style/adjacency combo
            for norm_style in norm_styles:
                for adjacency in adjacencies:
                    cmd = f"python run_experiments.py --dataset custom --method {method} --dataname {dataname}"
                    if norm_style:
                        cmd += f" --norm-style {norm_style}"
                    if adjacency:
                        cmd += f" --adjacency {adjacency}"
                    lines.append(cmd)

    # Write to file
    output_file = args.output
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    gen_cmds = 0 if args.skip_gen_models else 1
    cmds_per_method = gen_cmds + len(norm_styles) * len(adjacencies)
    total_cmds = len(indices) * len(methods) * cmds_per_method
    print(f"Generated {total_cmds} commands")
    print(f"  Datasets: ws{indices[0]:03d} to ws{indices[-1]:03d} ({len(indices)} datasets)")
    print(f"  Methods: {methods}")
    print(f"  Norm styles: {norm_styles}")
    print(f"  Adjacencies: {adjacencies}")
    print(f"\nSaved to: {output_file}")
    print("\nRun with:")
    print(f"  bash {output_file}")
    print(f"  # or parallel: parallel -j 4 < {output_file}")


if __name__ == "__main__":
    main()
