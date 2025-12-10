"""Generate commands for BA graph parameter sweep experiments.

Usage:
    python generate_ba_sweep_commands.py
    # Creates ba_sweep_commands.txt

    # Then run:
    bash ba_sweep_commands.txt
    # Or parallel:
    parallel -j 4 < ba_sweep_commands.txt
"""

# Sweep parameters
M_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_VALUES = [1000, 2000, 5000, 10000, 20000, 50000, 100000]  # log scale
SEEDS = [1, 2, 3, 4, 5]

# Common parameters
LABELING = "louvain"
N_CLASSES = 10
SIGMA = 5
DIM = 128

# Fixed values for each sweep
M_SWEEP_N = 10000  # N for m sweep
N_SWEEP_M = 2      # m for N sweep


def generate_command(n_nodes, m, seed, name):
    return (
        f"python generate_synthetic_graph_ba.py "
        f"--n_nodes {n_nodes} "
        f"--labeling {LABELING} "
        f"--n_classes {N_CLASSES} "
        f"--sigma {SIGMA} "
        f"--dim_features {DIM} "
        f"--m {m} "
        f"--seed {seed} "
        f"--name {name}"
    )


def main():
    lines = []

    # Header
    lines.append("# BA Graph Sweep Experiment")
    lines.append(f"# Common params: labeling={LABELING}, n_classes={N_CLASSES}, sigma={SIGMA}, dim={DIM}")
    lines.append("")

    # m sweep: vary m, fixed N=10000
    print(f"m sweep: m={M_VALUES}, N={M_SWEEP_N}, seeds={SEEDS}")
    lines.append(f"# === m Sweep: m={M_VALUES}, N={M_SWEEP_N}, seeds={SEEDS} ===")
    for m in M_VALUES:
        lines.append(f"# m={m}")
        for seed in SEEDS:
            name = f"ba_m{m}_seed{seed}"
            cmd = generate_command(M_SWEEP_N, m, seed, name)
            lines.append(cmd)
    lines.append("")

    # N sweep: vary N, fixed m=2
    print(f"N sweep: N={N_VALUES}, m={N_SWEEP_M}, seeds={SEEDS}")
    lines.append(f"# === N Sweep: N={N_VALUES}, m={N_SWEEP_M}, seeds={SEEDS} ===")
    for n in N_VALUES:
        lines.append(f"# N={n}")
        for seed in SEEDS:
            name = f"ba_n{n}_seed{seed}"
            cmd = generate_command(n, N_SWEEP_M, seed, name)
            lines.append(cmd)

    # Write to file
    output_file = "ba_sweep_commands.txt"
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    n_m_sweep = len(M_VALUES) * len(SEEDS)
    n_n_sweep = len(N_VALUES) * len(SEEDS)
    print(f"\nGenerated {n_m_sweep + n_n_sweep} commands")
    print(f"  - m sweep: {n_m_sweep} graphs")
    print(f"  - N sweep: {n_n_sweep} graphs")
    print(f"\nSaved to: {output_file}")
    print(f"\nRun with:")
    print(f"  bash {output_file}")
    print(f"  # or parallel: parallel -j 4 < {output_file}")


if __name__ == "__main__":
    main()
