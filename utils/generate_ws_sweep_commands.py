"""Generate commands for WS graph parameter sweep experiments.

Usage:
    python utils/generate_ws_sweep_commands.py
    # Creates ws_sweep_commands.txt

    # Then run:
    bash ws_sweep_commands.txt
    # Or parallel:
    parallel -j 4 < ws_sweep_commands.txt
"""

# Sweep parameters
P_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
N_VALUES = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
SEEDS = [1, 2, 3, 4, 5]
LABELINGS = ["equal", "highfreq"]

# Common parameters
K = 10
N_CLASSES = 10
DIM = 128
FEATURE_NOISE_SIGMA = 0

# Fixed values for each sweep
P_SWEEP_N = 10000  # n_nodes for p sweep (matches BA experiments)
N_SWEEP_P = 0.1    # p for N sweep baseline


def generate_command(n_nodes, k, p, labeling, seed):
    return (
        f"python generate_synthetic_graph_ws.py "
        f"--n_nodes {n_nodes} "
        f"--k {k} "
        f"--p {p} "
        f"--labeling {labeling} "
        f"--n_classes {N_CLASSES} "
        f"--dim_features {DIM} "
        f"--feature_noise_sigma {FEATURE_NOISE_SIGMA} "
        f"--seed {seed}"
    )


def main():
    lines = []

    # Header
    lines.append("# WS Graph Sweep Experiment")
    lines.append(f"# k={K}, n_classes={N_CLASSES}, dim={DIM}, feature_noise_sigma={FEATURE_NOISE_SIGMA}")
    lines.append(f"# labelings={LABELINGS}")
    lines.append("")

    # p sweep: vary p, fixed N=10000
    print(f"p sweep: p={P_VALUES}, N={P_SWEEP_N}, k={K}, seeds={SEEDS}, labelings={LABELINGS}")
    lines.append(f"# === p Sweep: p={P_VALUES}, N={P_SWEEP_N}, k={K}, seeds={SEEDS}, labelings={LABELINGS} ===")
    for p in P_VALUES:
        lines.append(f"# p={p}")
        for labeling in LABELINGS:
            for seed in SEEDS:
                cmd = generate_command(P_SWEEP_N, K, p, labeling, seed)
                lines.append(cmd)
    lines.append("")

    # N sweep: vary N, fixed p=0.1
    print(f"N sweep: N={N_VALUES}, p={N_SWEEP_P}, k={K}, seeds={SEEDS}, labelings={LABELINGS}")
    lines.append(f"# === N Sweep: N={N_VALUES}, p={N_SWEEP_P}, k={K}, seeds={SEEDS}, labelings={LABELINGS} ===")
    for n in N_VALUES:
        lines.append(f"# N={n}")
        for labeling in LABELINGS:
            for seed in SEEDS:
                cmd = generate_command(n, K, N_SWEEP_P, labeling, seed)
                lines.append(cmd)

    # Write to file
    output_file = "ws_sweep_commands.txt"
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    n_p_sweep = len(P_VALUES) * len(SEEDS) * len(LABELINGS)
    n_n_sweep = len(N_VALUES) * len(SEEDS) * len(LABELINGS)
    print(f"\nGenerated {n_p_sweep + n_n_sweep} commands")
    print(f"  - p sweep: {n_p_sweep} graphs")
    print(f"  - N sweep: {n_n_sweep} graphs")
    print(f"\nSaved to: {output_file}")
    print("\nRun with:")
    print(f"  bash {output_file}")
    print(f"  # or parallel: parallel -j 4 < {output_file}")


if __name__ == "__main__":
    main()
