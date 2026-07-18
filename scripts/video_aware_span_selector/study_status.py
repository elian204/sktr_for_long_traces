#!/usr/bin/env python3
"""Print a compact progress summary for the Breakfast selector OOF study."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from common import DEFAULT_STUDY_DIR, atomic_write_json, load_json


def latest_training_epoch(task: Dict[str, Any]) -> int | None:
    model_dir = Path(task["final_checkpoint"]).parent
    epochs = []
    for path in model_dir.glob("epoch-*.model"):
        try:
            epochs.append(int(path.stem.split("-")[-1]))
        except ValueError:
            continue
    return max(epochs) if epochs else None


def task_snapshot(task: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(task["run_dir"])
    complete_path = run_dir / "task_complete.json"
    status_path = run_dir / "task_status.json"
    if complete_path.is_file():
        payload = load_json(complete_path)
        return {
            "task_id": task["task_id"],
            "gpu": task["gpu"],
            "inner_fold": task["inner_fold"],
            "status": "complete",
            "epoch": 1000,
            "exported_cases": payload.get("export", {}).get("case_count"),
        }
    payload: Dict[str, Any] = {}
    if status_path.is_file():
        try:
            payload = load_json(status_path)
        except json.JSONDecodeError:
            payload = {"status": "unreadable_status"}
    return {
        "task_id": task["task_id"],
        "gpu": task["gpu"],
        "inner_fold": task["inner_fold"],
        "status": payload.get("status", "not_started"),
        "stage": payload.get("stage"),
        "epoch": latest_training_epoch(task),
        "error": payload.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    tasks = load_json(study_dir / "tasks.json")["tasks"]
    snapshots = [task_snapshot(task) for task in tasks]
    print(f"Breakfast video-selector OOF study: {study_dir}")
    for item in snapshots:
        epoch = "-" if item.get("epoch") is None else str(item["epoch"])
        stage = f"/{item['stage']}" if item.get("stage") else ""
        cases = (
            f" cases={item['exported_cases']}" if item.get("exported_cases") is not None else ""
        )
        print(
            f"inner={item['inner_fold']} gpu={item['gpu']} "
            f"status={item['status']}{stage} epoch={epoch}/1000{cases}"
        )
        if item.get("error"):
            print(f"  error: {item['error']}")
    counts: Dict[str, int] = {}
    for item in snapshots:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "counts": counts,
        "tasks": snapshots,
    }
    atomic_write_json(study_dir / "status_live.json", summary)
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()

