#!/usr/bin/env python3
"""Prepare one fold for epoch-scarcity checkpoint import and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from epoch_scarcity_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    PROTOCOL_VERSION,
    STUDY_TYPE,
    atomic_json,
    dataset_spec,
    file_sha256,
    read_json,
    read_lines,
    write_lines,
)


LOW_DATA_DIR = Path(__file__).resolve().parents[1] / "low_data_petri_diffact"
if str(LOW_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_DATA_DIR))

from low_data_common import create_dataset_view, load_official_split, write_alignment_dir  # noqa: E402


def source_d100_task(source_metadata: dict, dataset: str, fold: int, seed: int) -> dict:
    matches = [
        task
        for task in source_metadata.get("tasks", [])
        if task.get("dataset") == dataset
        and int(task.get("official_fold", -1)) == fold
        and int(task.get("subset_seed", -1)) == seed
        and int(task.get("diffact_fraction", -1)) == 100
        and task.get("condition") == "honest_low_data_petri"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source D100 honest task for {dataset} fold={fold} seed={seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--dataset", choices=sorted(("gtea", "50salads")), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_study_dir = args.source_study_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    source_metadata_path = source_study_dir / "study_metadata.json"
    source_metadata = read_json(source_metadata_path)
    if not source_metadata.get("immutable"):
        raise ValueError(f"Source study is not marked immutable: {source_metadata_path}")
    task = source_d100_task(source_metadata, args.dataset, args.fold, args.seed)
    source_experiment_dir = Path(task["experiment_dir"])

    spec = dataset_spec(args.dataset)
    if args.fold not in spec["folds"]:
        raise ValueError(f"Unsupported {args.dataset} fold: {args.fold}")
    official_train, official_test = load_official_split(data_root, args.dataset, args.fold)
    if len(official_train) != int(spec["train_cases"]):
        raise ValueError(
            f"{args.dataset} fold {args.fold} official train has {len(official_train)} cases; "
            f"expected {spec['train_cases']}"
        )
    if set(official_train).intersection(official_test):
        raise ValueError("Official train and test folds overlap")

    source_train_pool = read_lines(source_experiment_dir / "manifests" / "train_pool_cases.txt")
    source_test = read_lines(source_experiment_dir / "manifests" / "test_cases.txt")
    if source_train_pool != official_train:
        raise ValueError("Source train_pool_cases.txt does not match the official train bundle")
    if source_test != official_test:
        raise ValueError("Source test_cases.txt does not match the official test bundle")

    manifests = experiment_dir / "manifests"
    write_lines(manifests / "train_pool_cases.txt", official_train)
    write_lines(manifests / "official_train_cases.txt", official_train)
    write_lines(manifests / "test_cases.txt", official_test)
    create_dataset_view(
        data_root=data_root,
        dataset=args.dataset,
        view_root=experiment_dir / "diffact_dataset_view",
        train_cases=official_train,
    )
    write_alignment_dir(
        data_root,
        args.dataset,
        experiment_dir / "align" / "test",
        "test",
        official_test,
    )

    official_train_path = data_root / args.dataset / "splits" / f"train.split{args.fold}.bundle"
    official_test_path = data_root / args.dataset / "splits" / f"test.split{args.fold}.bundle"
    metadata = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "study_type": STUDY_TYPE,
        "dataset": args.dataset,
        "fold": args.fold,
        "trajectory_seed": args.seed,
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "official_train_split_path": str(official_train_path),
        "official_test_split_path": str(official_test_path),
        "official_train_split_sha256": file_sha256(official_train_path),
        "official_test_split_sha256": file_sha256(official_test_path),
        "n_official_train_cases": len(official_train),
        "n_official_test_cases": len(official_test),
        "checkpoint_epoch_indices": list(spec["checkpoint_epochs"]),
        "epoch_indexing": "zero_based",
        "source_study_dir": str(source_study_dir),
        "source_study_metadata": str(source_metadata_path),
        "source_study_spec_sha256": source_metadata.get("spec_sha256"),
        "source_d100_task_id": task["task_id"],
        "source_d100_task_state_path": task["state_path"],
        "source_experiment_dir": str(source_experiment_dir),
        "source_reuse_policy": (
            "Import only the five pre-specified checkpoints after the source D100 trajectory "
            "completes; copy and hash them; never modify source artifacts."
        ),
        "primary_condition": "official_full_train_petri",
        "oracle_condition": "oracle_test_fold_petri",
    }
    atomic_json(experiment_dir / "experiment_metadata.json", metadata)
    print(f"Prepared epoch-scarcity fold: {experiment_dir}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
