"""Aggregate metadata.json files from synthetic graph datasets into a single CSV.

Scans dataset/{datatype}/**/metadata.json and combines them into one CSV file
for analysis. Useful for comparing graph properties across experiments.

Usage:
    python aggregate_dataset_metadata.py --datatype ba
    python aggregate_dataset_metadata.py --datatype ws -o ws_metadata.csv

Output:
    {datatype}_metadata.csv (or custom path with -o)
"""

import argparse
import csv
import json
import os
from pathlib import Path


def aggregate_metadata(dataset_type: str, output: str = None) -> str:
    """Aggregate all metadata.json files from a dataset type directory.

    Args:
        dataset_type: Type of dataset (e.g., "ba" for Barabasi-Albert)
        output: Output CSV filename. If None, uses "{dataset_type}_metadata.csv"

    Returns:
        Path to the output CSV file
    """
    base_dir = Path(f"dataset/{dataset_type}")

    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    # Collect all metadata files
    metadata_files = list(base_dir.glob("*/metadata.json"))

    if not metadata_files:
        raise FileNotFoundError(f"No metadata.json files found in {base_dir}")

    print(f"Found {len(metadata_files)} metadata files in {base_dir}")

    # Read all metadata
    all_metadata = []
    all_keys = set()

    for metadata_file in sorted(metadata_files):
        with open(metadata_file, "r") as f:
            data = json.load(f)
            all_metadata.append(data)
            all_keys.update(data.keys())

    # Sort keys for consistent column order
    # Put 'name' first, then sort the rest alphabetically
    sorted_keys = ["name"] + sorted(k for k in all_keys if k != "name")

    # Determine output filename
    if output is None:
        output = f"{dataset_type}_metadata.csv"

    # Write to CSV
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys, extrasaction="ignore")
        writer.writeheader()
        for data in all_metadata:
            writer.writerow(data)

    print(f"Exported {len(all_metadata)} records to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metadata from synthetic graph datasets"
    )
    parser.add_argument(
        "--datatype",
        type=str,
        required=True,
        help="Dataset type to aggregate (e.g., 'ba' for Barabasi-Albert)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV filename (default: {datatype}_metadata.csv)",
    )
    args = parser.parse_args()

    aggregate_metadata(args.datatype, args.output)


if __name__ == "__main__":
    main()
