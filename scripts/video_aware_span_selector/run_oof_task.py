#!/usr/bin/env python3
"""Run one resumable inner-fold DiffAct trajectory and verify its OOF export."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from common import (
    DEFAULT_STUDY_DIR,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    normalize_case_id,
    read_bundle,
    verify_source_digest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_task(study_dir: Path, task_id: str) -> Dict[str, Any]:
    payload = load_json(study_dir / "tasks.json")
    matches = [task for task in payload["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one task named {task_id!r}, found {len(matches)}")
    return matches[0]


def stream_command(
    command: List[str], cwd: Path, log_path: Path, env: Dict[str, str] | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ (cd {cwd} && {' '.join(command)})", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"Command returned {returncode}; see {log_path}: {' '.join(command)}"
        )


def map_rows(path: Path) -> List[tuple[int, str]]:
    rows: List[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        fields = raw.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed video index row in {path}: {raw!r}")
        rows.append((int(fields[0]), normalize_case_id(fields[1])))
    rows.sort()
    return rows


def verify_export(task: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(task["softmax_output_dir"])
    expected_cases = read_bundle(Path(task["heldout_manifest"]))
    map_path = output_dir / "video_index_map.txt"
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    rows = map_rows(map_path)
    if [index for index, _ in rows] != list(range(len(rows))):
        raise ValueError("Export numeric IDs must be contiguous from zero")
    observed_cases = [case for _, case in rows]
    if observed_cases != expected_cases:
        raise ValueError(
            "Export case order/content does not match the immutable held-out manifest: "
            f"expected={len(expected_cases)}, observed={len(observed_cases)}"
        )
    class_count: int | None = None
    total_frames = 0
    for index, case in rows:
        raw_path = output_dir / f"{index}_raw.npy"
        canonical_path = output_dir / f"{index}.npy"
        pred_path = output_dir / f"{index}_pred.npy"
        for path in (raw_path, canonical_path, pred_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"Incomplete held-out export: {path}")
        raw = np.load(raw_path, mmap_mode="r")
        canonical = np.load(canonical_path, mmap_mode="r")
        pred = np.load(pred_path, mmap_mode="r")
        if raw.ndim != 2 or canonical.shape != raw.shape:
            raise ValueError(f"{case}: probability shapes raw={raw.shape}, canonical={canonical.shape}")
        if pred.shape != (raw.shape[1],):
            raise ValueError(f"{case}: pred shape={pred.shape}, expected={(raw.shape[1],)}")
        if class_count is None:
            class_count = int(raw.shape[0])
        if raw.shape[0] != class_count:
            raise ValueError(f"{case}: inconsistent class count {raw.shape[0]} vs {class_count}")
        total_frames += int(raw.shape[1])
    if not (output_dir / "mapping.txt").is_file() or not (
        output_dir / "ground_truth.csv"
    ).is_file():
        raise FileNotFoundError("Export is missing copied mapping.txt or ground_truth.csv")
    return {
        "case_count": len(rows),
        "frame_count": total_frames,
        "class_count": class_count,
        "case_order_matches_manifest": True,
        "raw_stream_verified": True,
        "canonical_stream_verified": True,
        "official_prediction_verified": True,
    }


def export_artifact_hashes(task: Dict[str, Any]) -> Dict[str, str]:
    output_dir = Path(task["softmax_output_dir"])
    rows = map_rows(output_dir / "video_index_map.txt")
    paths = [
        output_dir / "video_index_map.txt",
        output_dir / "mapping.txt",
        output_dir / "ground_truth.csv",
    ]
    for index, _ in rows:
        paths.extend(
            [
                output_dir / f"{index}_raw.npy",
                output_dir / f"{index}.npy",
                output_dir / f"{index}_pred.npy",
            ]
        )
    return {
        path.relative_to(Path(task["artifact_run_dir"])).as_posix(): file_sha256(path)
        for path in paths
    }


def assert_runtime_config(
    task: Dict[str, Any], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    config = load_json(Path(task["config_path"]))
    expected = {
        "selector_protocol_version": metadata["protocol_version"],
        "outer_fold": int(task["outer_fold"]),
        "inner_fold": int(task["inner_fold"]),
        "random_seed": int(task["seed"]),
        "initialization_seed": int(task["seed"]),
        "training_subset_manifest": task["train_manifest"],
        "heldout_oof_manifest": task["heldout_manifest"],
        "pre_specified_final_epoch": 1000,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Immutable task config mismatch: {mismatches}")
    return config


def completed_payload(task: Dict[str, Any]) -> Dict[str, Any] | None:
    if task.get("execution_mode") == "imported":
        import_path = Path(task["import_manifest_path"])
        if not import_path.is_file():
            raise FileNotFoundError(import_path)
        payload = load_json(import_path)
        verify_export(task)
        checkpoint = Path(task["final_checkpoint"])
        if payload.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise ValueError(f"Imported checkpoint hash changed: {task['task_id']}")
        hashes = export_artifact_hashes(task)
        recorded = {
            key: value
            for key, value in payload["artifact_sha256"].items()
            if key.startswith("softmax_heldout/")
        }
        if recorded != hashes:
            raise ValueError(f"Imported OOF export hashes changed: {task['task_id']}")
        if payload.get("artifact_digest") != canonical_digest(
            payload["artifact_sha256"]
        ):
            raise ValueError(f"Imported artifact digest is inconsistent: {task['task_id']}")
        return payload

    complete_path = Path(task["run_dir"]) / "task_complete.json"
    if not complete_path.is_file():
        return None
    payload = load_json(complete_path)
    if payload.get("task_id") != task["task_id"]:
        return None
    checkpoint = Path(task["final_checkpoint"])
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        return None
    verify_export(task)
    if payload.get("checkpoint_sha256") != file_sha256(checkpoint):
        return None
    if payload.get("export_artifact_sha256") != export_artifact_hashes(task):
        return None
    return payload


def run_task(study_dir: Path, task_id: str) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    diffact_root = Path(metadata["diffact_root"])
    verify_source_digest(metadata, diffact_root)
    task = load_task(study_dir, task_id)
    assert_runtime_config(task, metadata)
    existing = completed_payload(task)
    if existing is not None:
        print(f"COMPLETE {task_id}: verified existing task outputs", flush=True)
        return
    if task.get("execution_mode") != "train":
        raise RuntimeError(f"Imported task failed validation and cannot be retrained: {task_id}")

    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "task_status.json"
    status: Dict[str, Any] = {
        "task_id": task_id,
        "status": "running",
        "gpu": task["gpu"],
        "started_utc": utc_now(),
        "pid": os.getpid(),
    }
    atomic_write_json(status_path, status)
    try:
        checkpoint = Path(task["final_checkpoint"])
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            status["stage"] = "training"
            atomic_write_json(status_path, status)
            stream_command(
                [
                    sys.executable,
                    "-u",
                    "main.py",
                    "--config",
                    task["config_path"],
                    "--device",
                    str(task["gpu"]),
                ],
                diffact_root,
                run_dir / "train.log",
            )
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Training ended without pre-specified checkpoint: {checkpoint}"
            )
        atomic_write_json(
            run_dir / "train_complete.json",
            {
                "task_id": task_id,
                "completed_utc": utc_now(),
                "final_checkpoint": str(checkpoint),
                "final_epoch": 1000,
                "checkpoint_selection": "pre_specified_no_heldout_selection",
            },
        )

        status["stage"] = "softmax_export"
        atomic_write_json(status_path, status)
        output_dir = Path(task["softmax_output_dir"])
        try:
            export_summary = verify_export(task)
            print(f"Export already verifies for {task_id}; skipping inference", flush=True)
        except (FileNotFoundError, ValueError):
            env = dict(os.environ)
            env["ALLOW_NON_PREFERRED_GPU"] = "1"
            stream_command(
                [
                    sys.executable,
                    "-u",
                    "export_softmax.py",
                    "--config",
                    task["config_path"],
                    "--checkpoint",
                    str(checkpoint),
                    "--align_dir",
                    task["align_dir"],
                    "--output_dir",
                    str(output_dir),
                    "--device",
                    str(task["gpu"]),
                ],
                diffact_root,
                run_dir / "export.log",
                env=env,
            )
            export_summary = verify_export(task)

        complete = {
            "task_id": task_id,
            "status": "complete",
            "completed_utc": utc_now(),
            "gpu": task["gpu"],
            "final_checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "export": export_summary,
            "export_artifact_sha256": export_artifact_hashes(task),
            "source_digest": metadata["source_provenance"]["source_digest"],
        }
        atomic_write_json(run_dir / "task_complete.json", complete)
        status.update({"status": "complete", "stage": "complete", "completed_utc": utc_now()})
        atomic_write_json(status_path, status)
        print(
            f"COMPLETE {task_id}: {export_summary['case_count']} OOF videos, "
            f"{export_summary['frame_count']} frames",
            flush=True,
        )
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    run_task(args.study_dir.resolve(), args.task_id)


if __name__ == "__main__":
    main()
