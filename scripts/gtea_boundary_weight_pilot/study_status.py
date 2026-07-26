#!/usr/bin/env python3
"""Report multi-fold task, queue, tmux, analysis, and GPU status."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from common import FINAL_EPOCH, GPU_IDS, highest_saved_epoch, load_json


def last_logged_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    matches = re.findall(
        r"Epoch\s+(\d+)\s+-\s+Running Loss", path.read_text(errors="replace")
    )
    return int(matches[-1]) if matches else None


def tmux_exists(session: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def gpu_processes(gpu: int) -> list[Dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return [{"error": result.stderr.strip() or result.stdout.strip()}]
    rows: list[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) == 3 and fields[0]:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "used_memory_mib": int(fields[1]),
                    "process": fields[2],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    metadata = load_json(study_dir / "study_metadata.json")
    tasks = load_json(study_dir / "tasks.json")["tasks"]
    rows = []
    for task in tasks:
        state_path = Path(task["state_path"])
        state = load_json(state_path) if state_path.is_file() else {}
        complete_path = Path(task["run_dir"]) / "task_complete.json"
        imported = task.get("execution_mode") == "imported"
        import_path = Path(task["import_manifest_path"]) if imported else None
        status = (
            "imported"
            if imported and import_path is not None and import_path.is_file()
            else "complete"
            if complete_path.is_file()
            else state.get("status", "not_started")
        )
        artifact_run_dir = Path(task.get("artifact_run_dir", task["run_dir"]))
        artifact_model_dir = Path(task.get("artifact_model_dir", task["model_dir"]))
        export_complete_count = len(
            list(
                artifact_run_dir.glob(
                    "exports/epoch_*/sampling_seed_*/export_complete.json"
                )
            )
        )
        rows.append(
            {
                "task_id": task["task_id"],
                "official_fold": task["official_fold"],
                "decoder_boundary_loss": task["decoder_boundary_loss"],
                "training_seed": task["training_seed"],
                "physical_gpu": task["physical_gpu"],
                "execution_mode": task["execution_mode"],
                "status": status,
                "stage": "imported" if imported else state.get("stage"),
                "last_logged_epoch": (
                    FINAL_EPOCH
                    if imported
                    else last_logged_epoch(Path(task["run_dir"]) / "train.log")
                ),
                "highest_checkpoint_epoch": highest_saved_epoch(artifact_model_dir),
                "exports_complete": export_complete_count,
                "exports_expected": (
                    len(task["checkpoint_epochs"]) * len(task["inference_seeds"])
                ),
                "current_export_epoch": state.get("checkpoint_epoch"),
                "current_inference_seed": state.get("inference_seed"),
                "error": state.get("error"),
            }
        )

    analysis_path = study_dir / "analysis" / "analysis_complete.json"
    analysis = load_json(analysis_path) if analysis_path.is_file() else None
    sessions = {
        str(gpu): {
            "name": f"gtea_bweight_multifold_g{gpu}",
            "alive": tmux_exists(f"gtea_bweight_multifold_g{gpu}"),
            "queue_complete": (
                study_dir / "state" / f"gpu_{gpu}_queue_complete"
            ).is_file(),
        }
        for gpu in GPU_IDS
    }
    payload = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "protocol_version": metadata["protocol_version"],
        "tasks": rows,
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("imported", "not_started", "running", "complete", "failed")
        },
        "sessions": sessions,
        "gpu_processes": {str(gpu): gpu_processes(gpu) for gpu in GPU_IDS},
        "analysis": analysis,
    }
    import json

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
