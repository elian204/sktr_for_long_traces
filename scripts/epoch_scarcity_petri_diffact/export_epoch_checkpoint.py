#!/usr/bin/env python3
"""Export raw/canonical/official DiffAct streams for one imported checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from epoch_scarcity_common import (
    DEFAULT_DIFFACT_ROOT,
    atomic_json,
    checkpoint_metadata_path,
    checkpoint_model_path,
    dataset_spec,
    file_sha256,
    read_json,
    read_lines,
    softmax_dir,
)


def bundle_files(output_dir: Path, expected_cases: int) -> list[Path]:
    files = [
        output_dir / "video_index_map.txt",
        output_dir / "mapping.txt",
        output_dir / "ground_truth.csv",
    ]
    for case_index in range(expected_cases):
        files.extend(
            [
                output_dir / f"{case_index}.npy",
                output_dir / f"{case_index}_raw.npy",
                output_dir / f"{case_index}_pred.npy",
            ]
        )
    return files


def validate_bundle(output_dir: Path, expected_cases: int) -> None:
    for path in bundle_files(output_dir, expected_cases):
        if not path.is_file():
            raise FileNotFoundError(f"Missing exported bundle artifact: {path}")
    map_path = output_dir / "video_index_map.txt"
    case_rows = [line.strip() for line in map_path.read_text().splitlines() if line.strip()]
    if len(case_rows) != expected_cases:
        raise ValueError(
            f"Exported case map has {len(case_rows)} cases; expected {expected_cases}: {map_path}"
        )
    for case_index in range(expected_cases):
        canonical = output_dir / f"{case_index}.npy"
        raw = output_dir / f"{case_index}_raw.npy"
        official = output_dir / f"{case_index}_pred.npy"
        for path in (canonical, raw, official):
            if not path.is_file():
                raise FileNotFoundError(f"Missing exported comparator artifact: {path}")
        canonical_array = np.load(canonical, mmap_mode="r")
        raw_array = np.load(raw, mmap_mode="r")
        official_array = np.load(official, mmap_mode="r")
        if canonical_array.ndim != 2 or official_array.ndim != 1:
            raise ValueError(
                f"Invalid exported shapes for case {case_index}: "
                f"canonical={canonical_array.shape}, official={official_array.shape}"
            )
        if canonical_array.shape != raw_array.shape:
            raise ValueError(f"Raw/canonical shape mismatch for case {case_index}")
        if canonical_array.shape[1] != len(official_array):
            raise ValueError(f"Official prediction length mismatch for case {case_index}")


def bundle_sha256(output_dir: Path, expected_cases: int) -> Dict[str, str]:
    validate_bundle(output_dir, expected_cases)
    return {
        path.relative_to(output_dir).as_posix(): file_sha256(path)
        for path in bundle_files(output_dir, expected_cases)
    }


def validate_completed_export(
    experiment_dir: Path,
    *,
    seed: int,
    checkpoint_epoch: int,
) -> Dict[str, object]:
    output_dir = softmax_dir(experiment_dir, seed, checkpoint_epoch)
    complete_path = output_dir / "export_complete.json"
    complete = read_json(complete_path)
    checkpoint = checkpoint_model_path(experiment_dir, seed, checkpoint_epoch)
    checkpoint_meta = read_json(checkpoint_metadata_path(experiment_dir, seed, checkpoint_epoch))
    checkpoint_sha = file_sha256(checkpoint)
    expected_cases = len(read_lines(experiment_dir / "manifests" / "test_cases.txt"))
    expected = {
        "completed": True,
        "returncode": 0,
        "trajectory_seed": seed,
        "checkpoint_epoch_index": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "n_test_cases": expected_cases,
    }
    mismatches = {
        key: {"expected": value, "actual": complete.get(key)}
        for key, value in expected.items()
        if complete.get(key) != value
    }
    if checkpoint_sha != checkpoint_meta.get("source_checkpoint_sha256"):
        mismatches["checkpoint_metadata_sha256"] = {
            "expected": checkpoint_meta.get("source_checkpoint_sha256"),
            "actual": checkpoint_sha,
        }
    actual_artifacts = bundle_sha256(output_dir, expected_cases)
    if complete.get("artifact_sha256") != actual_artifacts:
        mismatches["artifact_sha256"] = {
            "expected": complete.get("artifact_sha256"),
            "actual": actual_artifacts,
        }
    if mismatches:
        raise ValueError(
            f"Completed checkpoint export failed immutable validation: {complete_path}: "
            f"{mismatches}"
        )
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    diffact_root = args.diffact_root.resolve()
    metadata = read_json(experiment_dir / "experiment_metadata.json")
    spec = dataset_spec(str(metadata["dataset"]))
    if args.checkpoint_epoch not in spec["checkpoint_epochs"]:
        raise ValueError(
            f"Checkpoint {args.checkpoint_epoch} is not in the approved grid: "
            f"{list(spec['checkpoint_epochs'])}"
        )
    checkpoint_meta = read_json(
        checkpoint_metadata_path(experiment_dir, args.seed, args.checkpoint_epoch)
    )
    checkpoint = checkpoint_model_path(experiment_dir, args.seed, args.checkpoint_epoch)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = file_sha256(checkpoint)
    if actual_sha != checkpoint_meta["source_checkpoint_sha256"]:
        raise ValueError(f"Imported checkpoint hash mismatch: {checkpoint}")

    output_dir = softmax_dir(experiment_dir, args.seed, args.checkpoint_epoch)
    complete_path = output_dir / "export_complete.json"
    expected_cases = len(read_lines(experiment_dir / "manifests" / "test_cases.txt"))
    if complete_path.is_file() and not args.force:
        validate_completed_export(
            experiment_dir,
            seed=args.seed,
            checkpoint_epoch=args.checkpoint_epoch,
        )
        print(f"SKIP existing validated export: {output_dir}")
        return
    config_path = experiment_dir / "trajectory" / f"seed_{args.seed}" / "config.json"
    align_dir = experiment_dir / "align" / "test"
    command = [
        sys.executable,
        "-u",
        "export_softmax.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--align_dir",
        str(align_dir),
        "--output_dir",
        str(output_dir),
        "--device",
        str(args.device),
    ]
    if not args.execute:
        print("PREPARED " + " ".join(command))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "export.log"
    environment = dict(os.environ)
    if args.device not in (1, 2, -1):
        environment["ALLOW_NON_PREFERRED_GPU"] = "1"
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=str(diffact_root),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Checkpoint export failed; see {log_path}")
    artifact_sha256 = bundle_sha256(output_dir, expected_cases)
    payload = {
        "schema_version": 1,
        "completed": True,
        "returncode": result.returncode,
        "dataset": metadata["dataset"],
        "fold": metadata["fold"],
        "trajectory_seed": args.seed,
        "checkpoint_epoch_index": args.checkpoint_epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "checkpoint_metadata": checkpoint_meta,
        "config": str(config_path),
        "align_dir": str(align_dir),
        "output_dir": str(output_dir),
        "n_test_cases": expected_cases,
        "artifact_sha256": artifact_sha256,
        "device": args.device,
        "command": command,
    }
    atomic_json(complete_path, payload)
    print(f"Exported checkpoint epoch {args.checkpoint_epoch}: {output_dir}")


if __name__ == "__main__":
    main()
