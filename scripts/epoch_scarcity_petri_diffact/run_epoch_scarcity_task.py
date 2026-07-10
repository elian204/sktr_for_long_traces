#!/usr/bin/env python3
"""Run one immutable epoch-scarcity task with dependency and heartbeat checks."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from generate_epoch_scarcity_study import source_provenance


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid JSON: {path}") from exc


def validate_state_dependency(task_id: str, dependency_id: str, state_path: Path) -> None:
    state = load_json(state_path)
    if state.get("status") != "complete" or state.get("returncode") != 0:
        raise RuntimeError(
            f"Task {task_id!r} cannot start before dependency {dependency_id!r} completes; "
            f"state={state_path}, status={state.get('status')!r}, "
            f"returncode={state.get('returncode')!r}"
        )


def validate_dependencies(metadata: Dict[str, Any], task: Dict[str, Any]) -> None:
    tasks_by_id = {str(item["task_id"]): item for item in metadata.get("tasks", [])}
    for dependency_id in task.get("depends_on_task_ids", []):
        dependency = tasks_by_id.get(str(dependency_id))
        if dependency is None:
            raise ValueError(f"Unknown dependency {dependency_id!r} for task {task['task_id']!r}")
        validate_state_dependency(
            str(task["task_id"]),
            str(dependency_id),
            Path(dependency["state_path"]),
        )
    for dependency in task.get("external_dependencies", []):
        validate_state_dependency(
            str(task["task_id"]),
            str(dependency["dependency_id"]),
            Path(dependency["state_path"]),
        )


def validate_source_provenance(metadata: Dict[str, Any]) -> None:
    recorded = metadata.get("source_provenance")
    if not isinstance(recorded, dict) or not recorded.get("source_digest"):
        raise ValueError("Study metadata is missing source_provenance.source_digest")
    current = source_provenance(Path(metadata["diffact_root"]))
    if current["source_digest"] != recorded["source_digest"]:
        recorded_hashes = recorded.get("file_sha256", {})
        current_hashes = current.get("file_sha256", {})
        changed = sorted(
            path
            for path in set(recorded_hashes) | set(current_hashes)
            if recorded_hashes.get(path) != current_hashes.get(path)
        )
        raise RuntimeError(
            "Epoch-scarcity source changed after immutable study generation; "
            f"recorded={recorded['source_digest']}, current={current['source_digest']}, "
            f"changed_files={changed}"
        )
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.heartbeat_seconds < 1:
        raise ValueError("--heartbeat-seconds must be positive")

    study_dir = args.study_dir.resolve()
    metadata = load_json(study_dir / "study_metadata.json")
    matches = [task for task in metadata.get("tasks", []) if task["task_id"] == args.task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one task {args.task_id!r}")
    task = matches[0]
    command = [str(value) for value in task["command"]]
    gpu = int(task["gpu"])
    approved_gpus = tuple(int(value) for value in metadata["gpu_policy"]["physical_gpu_ids"])
    if gpu not in approved_gpus:
        raise ValueError(f"Task uses GPU {gpu}; approved GPUs are {approved_gpus}")
    if bool(task.get("requires_gpu")):
        if "--device" not in command or int(command[command.index("--device") + 1]) != gpu:
            raise ValueError(f"GPU task {args.task_id} is not pinned to physical GPU {gpu}")
    validate_source_provenance(metadata)
    if args.dry_run:
        print(json.dumps({"task": task, "command": command}, indent=2, sort_keys=True))
        return
    validate_dependencies(metadata, task)

    state_path = Path(task["state_path"])
    log_path = Path(task["log_path"])
    lock_path = study_dir / "state" / f"queue_{task['queue']}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        started = utc_now()
        state: Dict[str, Any] = {
            "task_id": args.task_id,
            "task_type": task["task_type"],
            "dataset": task["dataset"],
            "official_fold": task["official_fold"],
            "trajectory_seed": task["trajectory_seed"],
            "checkpoint_epoch_index": task.get("checkpoint_epoch_index"),
            "condition": task.get("condition"),
            "gpu": gpu,
            "queue": task["queue"],
            "status": "running",
            "started_at": started,
            "heartbeat_timestamp": started,
            "completed_at": None,
            "returncode": None,
            "command": command,
            "wrapper_pid": os.getpid(),
            "log_path": str(log_path),
        }
        atomic_json(state_path, state)
        stop = threading.Event()
        guard = threading.Lock()
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n=== {started} task={args.task_id} gpu={gpu} ===\n")
            log_handle.write("COMMAND " + " ".join(command) + "\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=metadata["workspace_root"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            state["child_pid"] = process.pid
            atomic_json(state_path, state)

            def heartbeat() -> None:
                while not stop.wait(args.heartbeat_seconds):
                    with guard:
                        state["heartbeat_timestamp"] = utc_now()
                        atomic_json(state_path, state)

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                returncode = process.wait()
            finally:
                stop.set()
                thread.join(timeout=max(2, args.heartbeat_seconds + 1))
        completed = utc_now()
        with guard:
            state.update(
                {
                    "status": "complete" if returncode == 0 else "failed",
                    "returncode": int(returncode),
                    "completed_at": completed,
                    "heartbeat_timestamp": completed,
                }
            )
            atomic_json(state_path, state)
        if returncode != 0:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
