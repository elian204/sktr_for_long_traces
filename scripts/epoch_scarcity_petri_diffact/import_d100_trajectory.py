#!/usr/bin/env python3
"""Validate and copy the pre-specified checkpoints from a completed D100 trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from epoch_scarcity_common import (
    atomic_json,
    checkpoint_metadata_path,
    checkpoint_model_path,
    checkpoint_records,
    dataset_spec,
    file_sha256,
    read_json,
    read_lines,
    record_to_dict,
)


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and file_sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    actual = file_sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checkpoint changed during copy: {source}; expected={expected_sha256}, actual={actual}"
        )
    temporary.replace(destination)


def validate_completed_import(experiment_dir: Path) -> Dict[str, Any]:
    """Revalidate an imported trajectory and every copied checkpoint."""

    experiment_dir = experiment_dir.resolve()
    metadata = read_json(experiment_dir / "experiment_metadata.json")
    dataset = str(metadata["dataset"])
    fold = int(metadata["fold"])
    seed = int(metadata["trajectory_seed"])
    spec = dataset_spec(dataset)
    trajectory_dir = experiment_dir / "trajectory" / f"seed_{seed}"
    complete_path = trajectory_dir / "import_complete.json"
    complete = read_json(complete_path)
    expected_epochs = list(map(int, spec["checkpoint_epochs"]))
    records = complete.get("checkpoint_records")
    mismatches: Dict[str, Any] = {}
    expected_header = {
        "completed": True,
        "dataset": dataset,
        "fold": fold,
        "trajectory_seed": seed,
        "source_task_id": metadata["source_d100_task_id"],
    }
    for key, expected in expected_header.items():
        if complete.get(key) != expected:
            mismatches[key] = {"expected": expected, "actual": complete.get(key)}
    if not isinstance(records, list):
        mismatches["checkpoint_records"] = "missing or not a list"
        records = []
    observed_epochs = [int(record.get("epoch_index", -1)) for record in records]
    if observed_epochs != expected_epochs:
        mismatches["checkpoint_epoch_indices"] = {
            "expected": expected_epochs,
            "actual": observed_epochs,
        }
    calculated_records: Dict[int, Dict[str, Any]] = {}
    if records:
        try:
            n_train_cases = int(records[0]["n_train_cases"])
            batch_size = int(records[0]["batch_size"])
            calculated_records = {
                record.epoch_index: record_to_dict(record)
                for record in checkpoint_records(
                    dataset,
                    n_train_cases=n_train_cases,
                    batch_size=batch_size,
                )
            }
        except (KeyError, TypeError, ValueError) as exc:
            mismatches["checkpoint_provenance"] = str(exc)
    source_complete = Path(str(complete.get("source_train_complete", "")))
    if not source_complete.is_file():
        mismatches["source_train_complete"] = f"missing: {source_complete}"
    elif file_sha256(source_complete) != complete.get("source_train_complete_sha256"):
        mismatches["source_train_complete_sha256"] = "content hash changed"
    for record in records:
        epoch = int(record.get("epoch_index", -1))
        if epoch not in expected_epochs:
            continue
        calculated = calculated_records.get(epoch, {})
        for key, expected_value in calculated.items():
            if record.get(key) != expected_value:
                mismatches[f"epoch_{epoch}_{key}"] = {
                    "expected": expected_value,
                    "actual": record.get(key),
                }
        for key, expected_value in {
            "dataset": dataset,
            "fold": fold,
            "trajectory_seed": seed,
        }.items():
            if record.get(key) != expected_value:
                mismatches[f"epoch_{epoch}_{key}"] = {
                    "expected": expected_value,
                    "actual": record.get(key),
                }
        copied = checkpoint_model_path(experiment_dir, seed, epoch)
        recorded_copied = Path(str(record.get("copied_checkpoint", "")))
        expected_sha = str(record.get("source_checkpoint_sha256", ""))
        if recorded_copied.resolve() != copied.resolve():
            mismatches[f"epoch_{epoch}_copied_checkpoint"] = {
                "expected": str(copied),
                "actual": str(recorded_copied),
            }
        if not copied.is_file():
            mismatches[f"epoch_{epoch}_copied_checkpoint"] = f"missing: {copied}"
        elif file_sha256(copied) != expected_sha:
            mismatches[f"epoch_{epoch}_copied_sha256"] = "content hash changed"
        source = Path(str(record.get("source_checkpoint", "")))
        if not source.is_file():
            mismatches[f"epoch_{epoch}_source_checkpoint"] = f"missing: {source}"
        elif file_sha256(source) != expected_sha:
            mismatches[f"epoch_{epoch}_source_sha256"] = "content hash changed"
        source_config = Path(str(record.get("source_config", "")))
        if not source_config.is_file():
            mismatches[f"epoch_{epoch}_source_config"] = f"missing: {source_config}"
        elif file_sha256(source_config) != record.get("source_config_sha256"):
            mismatches[f"epoch_{epoch}_source_config_sha256"] = "content hash changed"
        checkpoint_metadata = checkpoint_metadata_path(experiment_dir, seed, epoch)
        try:
            recorded_metadata = read_json(checkpoint_metadata)
        except ValueError as exc:
            mismatches[f"epoch_{epoch}_checkpoint_metadata"] = str(exc)
        else:
            if recorded_metadata != record:
                mismatches[f"epoch_{epoch}_checkpoint_metadata"] = "record mismatch"
    if mismatches:
        raise ValueError(
            f"Completed trajectory import failed immutable validation: {complete_path}: "
            f"{mismatches}"
        )
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    metadata = read_json(experiment_dir / "experiment_metadata.json")
    dataset = str(metadata["dataset"])
    fold = int(metadata["fold"])
    seed = int(metadata["trajectory_seed"])
    source_experiment_dir = Path(metadata["source_experiment_dir"])
    source_state = read_json(Path(metadata["source_d100_task_state_path"]))
    if source_state.get("status") != "complete" or source_state.get("returncode") != 0:
        raise RuntimeError(
            f"Source D100 task {metadata['source_d100_task_id']} is not complete: "
            f"status={source_state.get('status')!r}, returncode={source_state.get('returncode')!r}"
        )

    source_run = source_experiment_dir / "diffact" / f"seed_{seed}" / "frac_100"
    source_config_path = source_run / "config.json"
    source_config = read_json(source_config_path)
    source_complete = read_json(source_run / "train_complete.json")
    spec = dataset_spec(dataset)
    if not source_complete.get("completed") or source_complete.get("returncode") != 0:
        raise RuntimeError(f"Source train_complete.json is not successful: {source_run}")
    if int(source_config.get("num_epochs", -1)) != int(spec["num_epochs"]):
        raise ValueError("Source D100 schedule does not match the approved published schedule")
    if int(source_config.get("actual_train_case_count", -1)) != int(spec["train_cases"]):
        raise ValueError("Source D100 trajectory does not use the complete official train fold")
    if int(source_config.get("low_data_fraction", -1)) != 100:
        raise ValueError("Source trajectory is not the D100 condition")
    if bool(source_config.get("evaluate_during_training", True)):
        raise ValueError("Source trajectory evaluated during training")
    if bool(source_config.get("log_train_results", True)):
        raise ValueError("Source trajectory logged training-set evaluation")
    log_freq = int(source_config.get("log_freq", -1))
    if log_freq < 1 or any(epoch % log_freq for epoch in spec["checkpoint_epochs"]):
        raise ValueError(
            f"Source log_freq={log_freq} does not save every approved checkpoint: "
            f"{list(spec['checkpoint_epochs'])}"
        )

    source_subset = read_lines(Path(source_config["training_subset_manifest"]))
    source_train_pool = read_lines(source_experiment_dir / "manifests" / "train_pool_cases.txt")
    local_train_pool = read_lines(experiment_dir / "manifests" / "train_pool_cases.txt")
    if set(source_subset) != set(source_train_pool) or len(source_subset) != len(source_train_pool):
        raise ValueError("Source D100 training manifest is not the complete source train pool")
    if source_train_pool != local_train_pool:
        raise ValueError("Prepared epoch-scarcity train pool differs from the source study")

    naming = str(source_config["naming"])
    source_model_dir = source_run / "training" / naming
    expected_final_checkpoint = source_model_dir / f"epoch-{spec['final_epoch_index']}.model"
    recorded_final_checkpoint = Path(str(source_complete.get("final_checkpoint", "")))
    if recorded_final_checkpoint.resolve() != expected_final_checkpoint.resolve():
        raise ValueError(
            "Source train_complete.json does not identify the approved final checkpoint: "
            f"recorded={recorded_final_checkpoint}, expected={expected_final_checkpoint}"
        )
    if not expected_final_checkpoint.is_file():
        raise FileNotFoundError(expected_final_checkpoint)
    recorded_latest = Path(str(source_complete.get("latest_pt", "")))
    expected_latest = source_model_dir / "latest.pt"
    if recorded_latest.resolve() != expected_latest.resolve() or not expected_latest.is_file():
        raise ValueError(
            "Source train_complete.json does not identify a valid latest.pt state: "
            f"recorded={recorded_latest}, expected={expected_latest}"
        )
    records = checkpoint_records(
        dataset,
        n_train_cases=len(source_train_pool),
        batch_size=int(source_config["batch_size"]),
    )
    prepared: List[Dict[str, Any]] = []
    for record in records:
        source_model = source_model_dir / f"epoch-{record.epoch_index}.model"
        if not source_model.is_file():
            raise FileNotFoundError(f"Required source checkpoint is missing: {source_model}")
        source_sha = file_sha256(source_model)
        destination = checkpoint_model_path(experiment_dir, seed, record.epoch_index)
        row = {
            **record_to_dict(record),
            "dataset": dataset,
            "fold": fold,
            "trajectory_seed": seed,
            "source_checkpoint": str(source_model),
            "source_checkpoint_sha256": source_sha,
            "copied_checkpoint": str(destination),
            "source_config": str(source_config_path),
            "source_config_sha256": file_sha256(source_config_path),
            "n_train_cases": len(source_train_pool),
            "batch_size": int(source_config["batch_size"]),
            "step_count_provenance": (
                "Derived from DiffAct's deterministic loop: one trainer step per training case "
                "presentation and one optimizer update whenever trainer_step % batch_size == 0."
            ),
        }
        prepared.append(row)
        if args.execute:
            copy_verified(source_model, destination, source_sha)
            atomic_json(
                checkpoint_metadata_path(experiment_dir, seed, record.epoch_index),
                row,
            )
        else:
            print(f"PREPARED epoch={record.epoch_index}: {source_model} -> {destination}")

    if not args.execute:
        print("No checkpoints copied; pass --execute after reviewing the import plan.")
        return

    trajectory_dir = experiment_dir / "trajectory" / f"seed_{seed}"
    trajectory_config = dict(source_config)
    trajectory_config.update(
        {
            "naming": f"epochscarcity_{dataset}_fold{fold}_seed{seed}",
            "root_data_dir": str(experiment_dir / "diffact_dataset_view"),
            "result_dir": str(trajectory_dir),
            "epoch_scarcity_protocol_version": metadata["protocol_version"],
            "epoch_scarcity_source_config": str(source_config_path),
            "epoch_scarcity_import_only": True,
        }
    )
    atomic_json(trajectory_dir / "config.json", trajectory_config)
    csv_path = trajectory_dir / "checkpoint_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prepared[0]))
        writer.writeheader()
        writer.writerows(prepared)
    payload = {
        "schema_version": 1,
        "completed": True,
        "dataset": dataset,
        "fold": fold,
        "trajectory_seed": seed,
        "source_task_id": metadata["source_d100_task_id"],
        "source_train_complete": str(source_run / "train_complete.json"),
        "source_train_complete_sha256": file_sha256(source_run / "train_complete.json"),
        "checkpoint_records": prepared,
    }
    atomic_json(trajectory_dir / "import_complete.json", payload)
    print(f"Imported {len(prepared)} checkpoints into {trajectory_dir}")


if __name__ == "__main__":
    main()
