#!/usr/bin/env python3
"""Run one immutable study task with persistent heartbeat and failure sentinels."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from generate_low_data_study import source_provenance


APPROVED_GPUS = (0, 1, 2, 3)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_task(study_dir: Path, task_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    metadata_path = study_dir / "study_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    matches = [task for task in metadata.get("tasks", []) if task.get("task_id") == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one task {task_id!r} in {metadata_path}")
    return metadata, matches[0]


def validate_completed_dependencies(
    metadata: Dict[str, Any],
    task: Dict[str, Any],
) -> None:
    """Require every declared producer task to have completed successfully."""
    dependency_ids = [str(value) for value in task.get("depends_on_task_ids", [])]
    if not dependency_ids:
        return
    tasks_by_id = {
        str(candidate.get("task_id")): candidate
        for candidate in metadata.get("tasks", [])
    }
    for dependency_id in dependency_ids:
        dependency = tasks_by_id.get(dependency_id)
        if dependency is None:
            raise ValueError(
                f"Task {task['task_id']!r} declares unknown dependency {dependency_id!r}"
            )
        state_path = Path(str(dependency["state_path"]))
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Task {task['task_id']!r} cannot start before dependency "
                f"{dependency_id!r}; missing or unreadable state {state_path}"
            ) from exc
        if state.get("status") != "complete" or state.get("returncode") != 0:
            raise RuntimeError(
                f"Task {task['task_id']!r} cannot start before dependency "
                f"{dependency_id!r} completes successfully; state={state_path}, "
                f"status={state.get('status')!r}, returncode={state.get('returncode')!r}"
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
    metadata, task = load_task(study_dir, args.task_id)
    expected_source = metadata.get("source_provenance", {})
    current_source = source_provenance(Path(metadata["diffact_root"]))
    if current_source["source_digest"] != expected_source.get("source_digest"):
        raise RuntimeError(
            "Study source files changed after immutable metadata generation. "
            "Refusing to run with mixed code; generate a new study directory."
        )
    gpu = int(task["gpu"])
    if gpu not in APPROVED_GPUS:
        raise ValueError(f"Refusing unapproved physical GPU {gpu}; allowed GPUs are {APPROVED_GPUS}")
    command = [str(part) for part in task["command"]]
    requires_gpu = bool(task.get("requires_gpu", True))
    if requires_gpu and (
        "--device" not in command or command[command.index("--device") + 1] != str(gpu)
    ):
        raise ValueError(f"Task command does not pin its assigned physical GPU {gpu}")
    if args.dry_run:
        print(json.dumps({"task_id": args.task_id, "gpu": gpu, "command": command}, indent=2))
        return
    validate_completed_dependencies(metadata, task)

    state_path = Path(task["state_path"])
    log_path = Path(task["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = study_dir / "state" / f"gpu_{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    base_state: Dict[str, Any] = {
        "task_id": args.task_id,
        "dataset": task["dataset"],
        "official_fold": task["official_fold"],
        "subset_seed": task["subset_seed"],
        "diffact_fraction": task["diffact_fraction"],
        "petri_fraction": task["petri_fraction"],
        "condition": task["condition"],
        "run_type": task["run_type"],
        "gpu": gpu,
        "command": command,
        "log_path": str(log_path),
        "wrapper_pid": os.getpid(),
    }

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        started = utc_now()
        state = {
            **base_state,
            "status": "running",
            "started_at": started,
            "heartbeat_timestamp": started,
            "completed_at": None,
            "returncode": None,
        }
        atomic_json(state_path, state)
        stop_heartbeat = threading.Event()
        state_guard = threading.Lock()

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n=== {started} task={args.task_id} gpu={gpu} run_type={task['run_type']} ===\n"
            )
            log_file.write("COMMAND " + " ".join(command) + "\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=metadata["workspace_root"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            state["child_pid"] = process.pid
            atomic_json(state_path, state)

            def heartbeat() -> None:
                while not stop_heartbeat.wait(args.heartbeat_seconds):
                    with state_guard:
                        state["heartbeat_timestamp"] = utc_now()
                        atomic_json(state_path, state)

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                returncode = process.wait()
            finally:
                stop_heartbeat.set()
                heartbeat_thread.join(timeout=max(2, args.heartbeat_seconds + 1))

        completed = utc_now()
        with state_guard:
            state.update(
                {
                    "status": "complete" if returncode == 0 else "failed",
                    "heartbeat_timestamp": completed,
                    "completed_at": completed,
                    "returncode": int(returncode),
                }
            )
            atomic_json(state_path, state)
        if returncode != 0:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
