#!/usr/bin/env python3
"""Report checkpoint, export, dependency-gate, tmux, and GPU-3 status."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from common import (
    FINAL_EPOCH,
    PHYSICAL_GPU,
    atomic_write_json,
    highest_saved_epoch,
    load_json,
)


def last_logged_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    matches = re.findall(r"Epoch\s+(\d+)\s+-\s+Running Loss", path.read_text(errors="replace"))
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


def gpu_processes() -> list[Dict[str, Any]]:
    result = subprocess.run(
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
    rows: list[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) == 3 and fields[0]:
            rows.append(
                {"pid": int(fields[0]), "used_memory_mib": int(fields[1]), "process": fields[2]}
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
        status = "complete" if complete_path.is_file() else state.get("status", "not_started")
        export_complete_count = len(list(Path(task["run_dir"]).glob(
            "exports/epoch_*/sampling_seed_*/export_complete.json"
        )))
        expected_exports = len(task["checkpoint_epochs"]) * len(task["inference_seeds"])
        row = {
            "task_id": task["task_id"],
            "decoder_boundary_loss": task["decoder_boundary_loss"],
            "role": task["role"],
            "status": status,
            "stage": state.get("stage"),
            "last_logged_epoch": last_logged_epoch(Path(task["run_dir"]) / "train.log"),
            "highest_checkpoint_epoch": highest_saved_epoch(Path(task["model_dir"])),
            "final_epoch": FINAL_EPOCH,
            "exports_complete": export_complete_count,
            "exports_expected": expected_exports,
            "current_export_epoch": state.get("checkpoint_epoch"),
            "current_inference_seed": state.get("inference_seed"),
            "error": state.get("error"),
        }
        rows.append(row)

    reconciliation_path = study_dir / "baseline_reconciliation.json"
    reconciliation = load_json(reconciliation_path) if reconciliation_path.is_file() else None
    analysis_path = study_dir / "analysis" / "analysis_complete.json"
    analysis = load_json(analysis_path) if analysis_path.is_file() else None
    session = "gtea_bweight_f1_s0_g3"
    payload = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "protocol_version": metadata["protocol_version"],
        "physical_gpu": PHYSICAL_GPU,
        "tmux_session": session,
        "tmux_alive": tmux_exists(session),
        "gpu_processes": gpu_processes(),
        "baseline_reconciliation": (
            {"available": True, "passed": reconciliation.get("passed")}
            if reconciliation is not None
            else {"available": False, "passed": None}
        ),
        "analysis": analysis,
        "tasks": rows,
    }
    atomic_write_json(study_dir / "status_live.json", payload)
    print(
        f"GTEA boundary-weight pilot | GPU {PHYSICAL_GPU} | "
        f"tmux={'alive' if payload['tmux_alive'] else 'absent'} | "
        f"processes={len(payload['gpu_processes'])}"
    )
    for row in rows:
        print(
            f"  weight={row['decoder_boundary_loss']:<3} role={row['role']:<32} "
            f"status={row['status']:<11} stage={str(row['stage']):<15} "
            f"epoch={row['last_logged_epoch']}/{FINAL_EPOCH} "
            f"checkpoint={row['highest_checkpoint_epoch']} "
            f"exports={row['exports_complete']}/{row['exports_expected']}"
        )
        if row["error"]:
            print(f"    error: {row['error']}")
    gate = payload["baseline_reconciliation"]
    print(f"  baseline reconciliation: {gate['passed'] if gate['available'] else 'pending'}")
    print(f"  analysis: {'complete' if analysis else 'pending'}")


if __name__ == "__main__":
    main()

