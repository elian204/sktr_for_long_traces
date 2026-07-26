#!/usr/bin/env python3
"""Report the staged onset-head tasks, launch gate, tmux, and GPU status."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import FINAL_EPOCH, PHYSICAL_GPU, highest_saved_epoch, load_json


def last_logged_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    matches = re.findall(
        r"Epoch\s+(\d+)\s+-\s+Running Loss", path.read_text(errors="replace")
    )
    return int(matches[-1]) if matches else None


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
        complete = (Path(task["run_dir"]) / "task_complete.json").is_file()
        rows.append(
            {
                "task_id": task["task_id"],
                "onset_loss_weight": task["class_specific_onset_loss_weight"],
                "status": "complete" if complete else state.get("status", "not_started"),
                "stage": state.get("stage"),
                "last_logged_epoch": last_logged_epoch(
                    Path(task["run_dir"]) / "train.log"
                ),
                "highest_checkpoint_epoch": highest_saved_epoch(
                    Path(task["model_dir"])
                ),
                "final_epoch": FINAL_EPOCH,
                "error": state.get("error"),
            }
        )
    contract = metadata["multifold_launch_dependency"]
    decision_path = Path(contract["decision_path"])
    decision = load_json(decision_path) if decision_path.is_file() else None
    session = "gtea_onset_head_f1_v1_g3"
    tmux_alive = (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(PHYSICAL_GPU),
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "tasks": rows,
        "launch_gate": {
            "decision_path": str(decision_path),
            "decision_exists": decision is not None,
            "onset_head_launch_gate_passes": (
                decision.get("onset_head_launch_gate_passes")
                if decision is not None
                else None
            ),
        },
        "tmux": {"session": session, "alive": tmux_alive},
        "gpu_3_compute_processes": [
            line for line in gpu.stdout.splitlines() if line.strip()
        ],
        "analysis": (
            load_json(study_dir / "analysis" / "analysis_complete.json")
            if (study_dir / "analysis" / "analysis_complete.json").is_file()
            else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
