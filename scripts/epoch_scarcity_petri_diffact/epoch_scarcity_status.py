#!/usr/bin/env python3
"""Summarize task, dependency, and artifact status for an epoch-scarcity study."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from epoch_scarcity_common import (
    DISCOVERY_BY_CONDITION,
    read_json,
    softmax_dir,
)
from export_epoch_checkpoint import validate_completed_export
from import_d100_trajectory import validate_completed_import
from run_epoch_petri_bounds import validate_completed_decode_grid


def completion_artifact(task: Dict[str, Any], study_dir: Path) -> Path:
    experiment_dir = (
        study_dir
        / "experiments"
        / str(task["dataset"])
        / f"fold_{int(task['official_fold'])}"
    )
    seed = int(task["trajectory_seed"])
    task_type = str(task["task_type"])
    if task_type == "import_trajectory":
        return experiment_dir / "trajectory" / f"seed_{seed}" / "import_complete.json"
    if task_type == "export_checkpoint":
        return softmax_dir(
            experiment_dir,
            seed,
            int(task["checkpoint_epoch_index"]),
        ) / "export_complete.json"
    if task_type == "decode_checkpoint_grid":
        return (
            experiment_dir
            / "petri"
            / str(task["condition"])
            / f"seed_{seed}"
            / "decode_grid_complete.json"
        )
    raise ValueError(f"Unknown task type: {task_type!r}")


def validate_task_artifact(
    task: Dict[str, Any],
    study_dir: Path,
) -> tuple[bool, str]:
    artifact_path = completion_artifact(task, study_dir)
    if not artifact_path.is_file():
        return False, f"missing completion artifact: {artifact_path}"
    experiment_dir = (
        study_dir
        / "experiments"
        / str(task["dataset"])
        / f"fold_{int(task['official_fold'])}"
    )
    seed = int(task["trajectory_seed"])
    task_type = str(task["task_type"])
    try:
        if task_type == "import_trajectory":
            validate_completed_import(experiment_dir)
        elif task_type == "export_checkpoint":
            validate_completed_export(
                experiment_dir,
                seed=seed,
                checkpoint_epoch=int(task["checkpoint_epoch_index"]),
            )
        elif task_type == "decode_checkpoint_grid":
            condition = str(task["condition"])
            validate_completed_decode_grid(
                experiment_dir,
                seed=seed,
                condition=condition,
                discovery_source=DISCOVERY_BY_CONDITION[condition],
            )
        else:
            raise ValueError(f"Unknown task type: {task_type!r}")
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        return False, str(exc)
    return True, "validated completion artifact and immutable contents"


def task_row(task: Dict[str, Any], study_dir: Path) -> Dict[str, Any]:
    state_path = Path(task["state_path"])
    state: Dict[str, Any] = {}
    if state_path.is_file():
        try:
            state = read_json(state_path)
        except ValueError:
            state = {"status": "invalid_state"}
    artifact_path = completion_artifact(task, study_dir)
    artifact_ok, artifact_detail = validate_task_artifact(task, study_dir)
    state_status = str(state.get("status", "pending"))
    returncode = state.get("returncode")
    if state_status == "complete" and returncode == 0 and artifact_ok:
        effective_status = "complete"
    elif state_status == "complete" and (returncode != 0 or not artifact_ok):
        effective_status = "inconsistent"
    elif state_status in {"failed", "invalid_state"}:
        effective_status = state_status
    elif state_status == "running":
        effective_status = "running"
    elif artifact_ok:
        effective_status = "artifact_without_state"
    else:
        effective_status = "pending"
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "dataset": task["dataset"],
        "official_fold": task["official_fold"],
        "trajectory_seed": task["trajectory_seed"],
        "checkpoint_epoch_index": task.get("checkpoint_epoch_index"),
        "condition": task.get("condition"),
        "gpu": task["gpu"],
        "queue": task["queue"],
        "effective_status": effective_status,
        "state_status": state_status,
        "returncode": returncode,
        "heartbeat_timestamp": state.get("heartbeat_timestamp"),
        "artifact_complete": artifact_ok,
        "artifact_detail": artifact_detail,
        "artifact_path": str(artifact_path),
        "state_path": str(state_path),
        "log_path": task["log_path"],
    }


def collect_status(study_dir: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    study_dir = study_dir.resolve()
    metadata = read_json(study_dir / "study_metadata.json")
    rows = [task_row(task, study_dir) for task in metadata.get("tasks", [])]
    frame = pd.DataFrame(rows)
    counts = Counter(frame["effective_status"].tolist()) if not frame.empty else Counter()
    summary = {
        "schema_version": 1,
        "study_id": metadata["study_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "all_complete": bool(rows) and counts.get("complete", 0) == len(rows),
        "by_gpu": {
            str(gpu): dict(sorted(Counter(group["effective_status"]).items()))
            for gpu, group in frame.groupby("gpu")
        }
        if not frame.empty
        else {},
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    frame, summary = collect_status(study_dir)
    output_dir = study_dir / "status"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "task_status.csv", index=False)
    (output_dir / "status_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
