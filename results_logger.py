"""
CSV result logging for experiments.

Saves results to:
- results/gen_models.csv - base model training results
- results/run_experiments.csv - C&S post-processing results
"""

import csv
import os

CSV_DIR = "results"


def save_gen_models_result(dataname, args, run, train, valid, test):
    """Append one row to results/gen_models.csv"""
    csv_path = os.path.join(CSV_DIR, "gen_models.csv")
    os.makedirs(CSV_DIR, exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "dataname",
                    "model",
                    "epochs",
                    "hidden_channels",
                    "use_embeddings",
                    "run",
                    "train",
                    "valid",
                    "test",
                ]
            )
        writer.writerow(
            [
                dataname,
                args.model,
                args.epochs,
                args.hidden_channels,
                args.use_embeddings,
                run,
                train,
                valid,
                test,
            ]
        )


def save_run_experiments_result(
    dataname, args, run, orig_valid, orig_test, cs_valid, cs_test
):
    """Append one row to results/run_experiments.csv"""
    csv_path = os.path.join(CSV_DIR, "run_experiments.csv")
    os.makedirs(CSV_DIR, exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "dataname",
                    "method",
                    "normalizer",
                    "adjacency",
                    "norm_style",
                    "run",
                    "orig_valid",
                    "orig_test",
                    "cs_valid",
                    "cs_test",
                ]
            )
        writer.writerow(
            [
                dataname,
                args.method,
                args.normalizer,
                args.adjacency,
                args.norm_style,
                run,
                orig_valid,
                orig_test,
                cs_valid,
                cs_test,
            ]
        )
