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
    LOW_DATA_PROTOCOL_VERSION,
    build_case_table,
    bundle_name,
    expected_fraction_size,
    file_sha256,
    fraction_size,
    load_ground_truth_indices,
    load_mapping,
    read_bundle,
    read_case_manifest,
    read_lines,
    stratified_order_by_variant,
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
    protocol_version = metadata.get("protocol_version")
    if protocol_version not in (None, LOW_DATA_PROTOCOL_VERSION):
        raise AssertionError(f"Unsupported low-data protocol version: {protocol_version}")
    official_train_protocol = protocol_version == LOW_DATA_PROTOCOL_VERSION

    manifests = experiment_dir / "manifests"
    official_train = read_case_manifest(manifests / "official_train_cases.txt")
    train_pool = read_case_manifest(manifests / "train_pool_cases.txt")
    val_cases = read_case_manifest(manifests / "val_cases.txt")
    test_cases = read_case_manifest(manifests / "test_cases.txt")
    assert train_pool, "train_pool_cases.txt is empty"
    assert test_cases, "test_cases.txt is empty"
    if official_train_protocol:
        if not (manifests / "val_cases.txt").is_file():
            raise FileNotFoundError("No-validation protocol requires an empty val_cases.txt")
        if val_cases:
            raise AssertionError("No-validation protocol requires val_cases.txt to be empty")
        if not official_train:
            raise AssertionError("official_train_cases.txt is empty")
        if train_pool != official_train:
            raise AssertionError(
                "train_pool_cases.txt must exactly equal the complete official training manifest"
            )

        official_train_path = Path(metadata["official_train_split_path"])
        official_test_path = Path(metadata["official_test_split_path"])
        source_train = read_bundle(official_train_path)
        source_test = read_bundle(official_test_path)
        if official_train != source_train:
            raise AssertionError("Generated official training manifest differs from its source bundle")
        if test_cases != source_test:
            raise AssertionError("Generated test manifest differs from its source bundle")
        if file_sha256(official_train_path) != metadata.get("official_train_split_sha256"):
            raise AssertionError("Official training bundle SHA-256 differs from recorded provenance")
        if file_sha256(official_test_path) != metadata.get("official_test_split_sha256"):
            raise AssertionError("Official test bundle SHA-256 differs from recorded provenance")
        if int(metadata.get("n_official_train", -1)) != len(official_train):
            raise AssertionError("Metadata n_official_train does not match official training bundle")
        if int(metadata.get("n_train_pool", -1)) != len(train_pool):
            raise AssertionError("Metadata n_train_pool does not match complete official training fold")
        if int(metadata.get("n_val", -1)) != 0:
            raise AssertionError("Metadata n_val must be zero for the no-validation protocol")
        if int(metadata.get("n_test", -1)) != len(test_cases):
            raise AssertionError("Metadata n_test does not match official test bundle")
        assert_no_overlap({"official_train": official_train, "test": test_cases})
    else:
        assert val_cases, "val_cases.txt is empty"
        assert_no_overlap({"train_pool": train_pool, "val": val_cases, "test": test_cases})

    split_manifest_path = manifests / "split_manifest.csv"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest_path}")
    split_manifest = pd.read_csv(split_manifest_path)
    expected_cols = ["case_id", "split"]
    if list(split_manifest.columns) != expected_cols:
        raise AssertionError(f"split_manifest.csv columns must be {expected_cols}")
    expected_split_rows = [{"case_id": case_id, "split": "train_pool"} for case_id in train_pool]
    if not official_train_protocol:
        expected_split_rows += [{"case_id": case_id, "split": "val"} for case_id in val_cases]
    expected_split_rows += [{"case_id": case_id, "split": "test"} for case_id in test_cases]
    actual_split_rows = split_manifest[expected_cols].to_dict("records")
    if actual_split_rows != expected_split_rows:
        raise AssertionError("split_manifest.csv does not match train/val/test manifest files")
    if split_manifest["case_id"].duplicated().any():
        dupes = split_manifest.loc[split_manifest["case_id"].duplicated(), "case_id"].tolist()
        raise AssertionError(f"split_manifest.csv has duplicate case ids: {dupes[:10]}")

    failures: List[str] = []
    deterministic_orders: Dict[int, List[str]] = {}
    if official_train_protocol:
        case_table = build_case_table(data_root, dataset, official_train)
        for seed in seeds:
            deterministic_orders[seed] = stratified_order_by_variant(
                case_table, official_train, seed
            )
        expected_count_metadata = {
            str(fraction): expected_fraction_size(dataset, len(train_pool), fraction)
            for fraction in fractions
        }
        if metadata.get("expected_subset_case_counts") != expected_count_metadata:
            failures.append("metadata expected_subset_case_counts does not match protocol counts")

    for seed in seeds:
        seed_dir = manifests / f"seed_{seed}"
        if official_train_protocol:
            actual_order = read_case_manifest(seed_dir / "train_pool_stratified_order.txt")
            expected_order = deterministic_orders[seed]
            if actual_order != expected_order:
                failures.append(
                    f"seed {seed}: stratified order is not deterministic from official train"
                )
        previous_set = set()
        previous_fraction = None
        full_subset = None

        for fraction in sorted(fractions):
            subset_path = seed_dir / f"train_cases_frac_{fraction}.txt"
            subset = read_case_manifest(subset_path)
            subset_set = set(subset)
            expected_size = (
                expected_fraction_size(dataset, len(train_pool), fraction)
                if official_train_protocol
                else fraction_size(len(train_pool), fraction)
            )
            if not subset:
                failures.append(f"seed {seed} frac {fraction}: empty subset")
            if len(subset) != expected_size:
                failures.append(
                    f"seed {seed} frac {fraction}: subset size {len(subset)} != expected {expected_size}"
                )
            if not subset_set.issubset(set(train_pool)):
                failures.append(f"seed {seed} frac {fraction}: subset contains cases outside train_pool")
            if subset_set.intersection(test_cases):
                failures.append(f"seed {seed} frac {fraction}: subset contains official test cases")
            if official_train_protocol:
                expected_subset = deterministic_orders[seed][:expected_size]
                if subset != expected_subset:
                    failures.append(
                        f"seed {seed} frac {fraction}: subset is not the expected "
                        "deterministic stratified prefix"
                    )
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
            if train_bundle != subset:
                failures.append(f"seed {seed} frac {fraction}: train split bundle mismatch")
            if train_bundle_raw != [bundle_name(c) for c in subset]:
                failures.append(f"seed {seed} frac {fraction}: train split bundle suffix/order mismatch")
            evaluation_bundle = view_root / "test.split1.bundle"
            if official_train_protocol:
                if evaluation_bundle.exists() or evaluation_bundle.is_symlink():
                    failures.append(
                        f"seed {seed} frac {fraction}: no-validation view must not expose a test bundle"
                    )
            else:
                val_bundle = read_lines(evaluation_bundle)
                expected_val_bundle = [bundle_name(c) for c in val_cases]
                if val_bundle != expected_val_bundle:
                    failures.append(
                        f"seed {seed} frac {fraction}: validation split bundle mismatch"
                    )
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

        if 100 in fractions and (full_subset is None or set(full_subset) != set(train_pool)):
            failures.append(f"seed {seed}: 100 percent subset does not equal train_pool")

    label_to_idx, _ = load_mapping(data_root / dataset / "mapping.txt")
    alignment_splits = [("test", test_cases)]
    if not official_train_protocol:
        alignment_splits.insert(0, ("val", val_cases))
    elif (experiment_dir / "align" / "val").exists():
        failures.append("no-validation protocol must not generate validation alignment files")

    for split_name, expected_cases in alignment_splits:
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
    print(f"Protocol: {protocol_version or 'legacy-validation-pilot'}")
    print(f"Seeds: {seeds}")
    print(f"Fractions: {fractions}")
    train_label = "Official train" if official_train_protocol else "Train pool"
    print(f"{train_label} / val / test: {len(train_pool)} / {len(val_cases)} / {len(test_cases)}")
    print("Nested subset, official split, dataset view, and alignment checks passed.")


if __name__ == "__main__":
    main()
