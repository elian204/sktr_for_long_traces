#!/usr/bin/env python3
"""Verify low-data ablation manifests and DiffAct dataset views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from low_data_common import (
    DEFAULT_EXPERIMENT_DIR,
    FRACTIONS,
    bundle_name,
    fraction_size,
    load_ground_truth_indices,
    load_mapping,
    read_bundle,
    read_case_manifest,
    read_lines,
)


def assert_no_overlap(groups: Dict[str, List[str]]) -> None:
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sorted(set(groups[a]).intersection(groups[b]))
            if overlap:
                raise AssertionError(f"{a}/{b} overlap: {overlap[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    metadata_path = experiment_dir / "experiment_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    dataset = metadata["dataset"]
    data_root = Path(metadata["data_root"])
    seeds = [int(x) for x in metadata.get("seeds", [])]
    fractions = [int(x) for x in metadata.get("fractions", FRACTIONS)]

    manifests = experiment_dir / "manifests"
    train_pool = read_case_manifest(manifests / "train_pool_cases.txt")
    val_cases = read_case_manifest(manifests / "val_cases.txt")
    test_cases = read_case_manifest(manifests / "test_cases.txt")
    assert train_pool, "train_pool_cases.txt is empty"
    assert val_cases, "val_cases.txt is empty"
    assert test_cases, "test_cases.txt is empty"
    assert_no_overlap({"train_pool": train_pool, "val": val_cases, "test": test_cases})

    split_manifest_path = manifests / "split_manifest.csv"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest_path}")
    split_manifest = pd.read_csv(split_manifest_path)
    expected_cols = ["case_id", "split"]
    if list(split_manifest.columns) != expected_cols:
        raise AssertionError(f"split_manifest.csv columns must be {expected_cols}")
    expected_split_rows = (
        [{"case_id": case_id, "split": "train_pool"} for case_id in train_pool]
        + [{"case_id": case_id, "split": "val"} for case_id in val_cases]
        + [{"case_id": case_id, "split": "test"} for case_id in test_cases]
    )
    actual_split_rows = split_manifest[expected_cols].to_dict("records")
    if actual_split_rows != expected_split_rows:
        raise AssertionError("split_manifest.csv does not match train/val/test manifest files")
    if split_manifest["case_id"].duplicated().any():
        dupes = split_manifest.loc[split_manifest["case_id"].duplicated(), "case_id"].tolist()
        raise AssertionError(f"split_manifest.csv has duplicate case ids: {dupes[:10]}")

    expected_val_bundle = [bundle_name(c) for c in val_cases]
    failures: List[str] = []

    for seed in seeds:
        seed_dir = manifests / f"seed_{seed}"
        previous_set = set()
        previous_fraction = None
        full_subset = None

        for fraction in sorted(fractions):
            subset_path = seed_dir / f"train_cases_frac_{fraction}.txt"
            subset = read_case_manifest(subset_path)
            subset_set = set(subset)
            expected_size = fraction_size(len(train_pool), fraction)
            if not subset:
                failures.append(f"seed {seed} frac {fraction}: empty subset")
            if len(subset) != expected_size:
                failures.append(
                    f"seed {seed} frac {fraction}: subset size {len(subset)} != expected {expected_size}"
                )
            if not subset_set.issubset(set(train_pool)):
                failures.append(f"seed {seed} frac {fraction}: subset contains cases outside train_pool")
            if previous_fraction is not None and not previous_set.issubset(subset_set):
                failures.append(
                    f"seed {seed}: fraction {previous_fraction} is not subset of fraction {fraction}"
                )
            if len(subset) != len(subset_set):
                failures.append(f"seed {seed} frac {fraction}: duplicate cases")
            if fraction == 100:
                full_subset = subset

            view_root = (
                experiment_dir
                / "diffact_dataset_views"
                / f"seed_{seed}"
                / f"frac_{fraction}"
                / dataset
                / "splits"
            )
            train_bundle = read_bundle(view_root / "train.split1.bundle")
            train_bundle_raw = read_lines(view_root / "train.split1.bundle")
            val_bundle = read_lines(view_root / "test.split1.bundle")
            if train_bundle != subset:
                failures.append(f"seed {seed} frac {fraction}: train split bundle mismatch")
            if train_bundle_raw != [bundle_name(c) for c in subset]:
                failures.append(f"seed {seed} frac {fraction}: train split bundle suffix/order mismatch")
            if val_bundle != expected_val_bundle:
                failures.append(f"seed {seed} frac {fraction}: validation split bundle mismatch")
            for required in ("features", "groundTruth", "mapping.txt"):
                if not (view_root.parent / required).exists():
                    failures.append(f"seed {seed} frac {fraction}: missing dataset view {required}")

            summary_path = seed_dir / f"subset_summary_frac_{fraction}.csv"
            if not summary_path.is_file():
                failures.append(f"seed {seed} frac {fraction}: missing subset summary")
            else:
                summary = pd.read_csv(summary_path)
                if len(summary) != 1:
                    failures.append(f"seed {seed} frac {fraction}: subset summary should have one row")
                else:
                    row = summary.iloc[0]
                    if int(row["n_cases"]) != len(subset):
                        failures.append(f"seed {seed} frac {fraction}: summary n_cases mismatch")
                    expected_pct = 100.0 * len(subset) / len(train_pool)
                    if abs(float(row["percentage_of_train_pool"]) - expected_pct) > 1e-9:
                        failures.append(
                            f"seed {seed} frac {fraction}: summary percentage_of_train_pool mismatch"
                        )

            previous_set = subset_set
            previous_fraction = fraction

        if full_subset is None or set(full_subset) != set(train_pool):
            failures.append(f"seed {seed}: 100 percent subset does not equal train_pool")

    label_to_idx, _ = load_mapping(data_root / dataset / "mapping.txt")
    for split_name, expected_cases in (("val", val_cases), ("test", test_cases)):
        map_path = experiment_dir / "align" / split_name / "case_id_map.csv"
        video_map = experiment_dir / "align" / split_name / "video_index_map.txt"
        ground_truth = experiment_dir / "align" / split_name / "ground_truth.csv"
        if not map_path.is_file() or not video_map.is_file() or not ground_truth.is_file():
            failures.append(f"{split_name}: missing alignment files")
            continue
        rows = read_lines(video_map)
        if len(rows) != len(expected_cases):
            failures.append(f"{split_name}: alignment case count mismatch")
        for i, line in enumerate(rows):
            parts = line.split()
            if len(parts) < 2 or parts[0] != str(i) or parts[1] != expected_cases[i]:
                failures.append(f"{split_name}: bad video_index_map line {i}: {line}")
                break
        case_map = pd.read_csv(map_path)
        expected_map = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "original_case_id": case_id,
                    "sequential_case_id": str(i),
                    "softmax_list_index": i,
                }
                for i, case_id in enumerate(expected_cases)
            ]
        )
        actual_map = case_map.astype({"sequential_case_id": str})
        if actual_map.to_dict("records") != expected_map.to_dict("records"):
            failures.append(f"{split_name}: case_id_map.csv mismatch")
        gt = pd.read_csv(ground_truth)
        if list(gt.columns) != ["case:concept:name", "concept:name"]:
            failures.append(f"{split_name}: ground_truth.csv columns mismatch")
        else:
            for i, case_id in enumerate(expected_cases):
                expected_labels = [
                    str(x) for x in load_ground_truth_indices(data_root, dataset, case_id, label_to_idx)
                ]
                actual_labels = (
                    gt[gt["case:concept:name"].astype(str) == str(i)]["concept:name"]
                    .astype(str)
                    .tolist()
                )
                if actual_labels != expected_labels:
                    failures.append(f"{split_name}: ground_truth.csv labels mismatch for {case_id}")
                    break

    if failures:
        raise AssertionError("\n".join(failures))

    print(f"Verified manifests in {experiment_dir}")
    print(f"Dataset: {dataset}")
    print(f"Seeds: {seeds}")
    print(f"Fractions: {fractions}")
    print(f"Train pool / val / test: {len(train_pool)} / {len(val_cases)} / {len(test_cases)}")
    print("Nested subset, fixed split, dataset view, and alignment checks passed.")


if __name__ == "__main__":
    main()
