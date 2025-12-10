#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate saved model outputs from the models/ directory.

Works with all model types: plain, linear, mlp, gat

Usage:
    python evaluate_saved.py --model-dir models/arxiv_gat
    python evaluate_saved.py --model-dir models/arxiv_mlp
    python evaluate_saved.py --model-dir models/arxiv_plain
    python evaluate_saved.py --model-dir models/arxiv_linear

    # For custom datasets:
    python evaluate_saved.py --model-dir models/ba/ba001-mlp --dataset custom --dataname ba001
"""

import argparse
import glob
import os

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from custom_dataset import CustomDataset
from custom_evaluator import CustomEvaluator


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def get_run_from_file(filepath):
    return int(os.path.splitext(os.path.basename(filepath))[0])


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model outputs")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing saved model outputs")
    parser.add_argument("--dataset", type=str, default="arxiv",
                        help="Dataset name: arxiv, products, or custom (default: arxiv)")
    parser.add_argument("--dataname", type=str, default=None,
                        help="Dataset name for custom datasets (e.g., ba001)")
    args = parser.parse_args()

    # Validate arguments
    if args.dataset == "custom" and args.dataname is None:
        parser.error("--dataname is required when --dataset is 'custom'")

    # Load dataset
    if args.dataset == "custom":
        dataset = CustomDataset(args.dataname)
        evaluator = CustomEvaluator()
        dataset_name = args.dataname
    else:
        dataset = PygNodePropPredDataset(name=f"ogbn-{args.dataset}")
        evaluator = Evaluator(name=f"ogbn-{args.dataset}")
        dataset_name = f"ogbn-{args.dataset}"

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    data = dataset[0]
    labels = data.y

    # Find all model outputs
    model_outs = sorted(glob.glob(f"{args.model_dir}/*.pt"))

    if not model_outs:
        print(f"No .pt files found in {args.model_dir}")
        return

    print(f"Dataset: {dataset_name}")
    print(f"Found {len(model_outs)} model outputs in {args.model_dir}")
    print("-" * 70)
    print(f"{'Run':<6} {'Train Acc':<12} {'Valid Acc':<12} {'Test Acc':<12}")
    print("-" * 70)

    train_accs, val_accs, test_accs = [], [], []

    for out_file in model_outs:
        run = get_run_from_file(out_file)
        model_out = torch.load(out_file, map_location="cpu", weights_only=False)

        # Handle case where model_out might be a tuple (model_out, split_idx)
        if isinstance(model_out, tuple):
            model_out, custom_split = model_out
            train_idx = custom_split["train"]
            val_idx = custom_split["valid"]
            test_idx = custom_split["test"]

        train_acc = compute_acc(model_out[train_idx], labels[train_idx], evaluator)
        val_acc = compute_acc(model_out[val_idx], labels[val_idx], evaluator)
        test_acc = compute_acc(model_out[test_idx], labels[test_idx], evaluator)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"{run:<6} {train_acc:<12.4f} {val_acc:<12.4f} {test_acc:<12.4f}")

    print("-" * 70)

    print(f"\nSummary ({len(model_outs)} runs):")
    print(f"  Train Acc: {np.mean(train_accs):.4f} +/- {np.std(train_accs):.4f}")
    print(f"  Valid Acc: {np.mean(val_accs):.4f} +/- {np.std(val_accs):.4f}")
    print(f"  Test Acc:  {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")


if __name__ == "__main__":
    main()
