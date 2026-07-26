#!/usr/bin/env python3
"""Train one resumable onset-head variant and export its locked checkpoint grid."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from common import (
    FINAL_EPOCH,
    PHYSICAL_GPU,
    SCRIPT_DIR,
    atomic_write_json,
    assert_variant_config,
    checkpoint_path,
    export_dir,
    file_sha256,
    load_json,
    load_task,
    read_bundle,
    verify_export,
    verify_source_digest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stream_command(
    command: List[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: Mapping[str, str],
    append: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    print(f"$ (cd {cwd} && {' '.join(command)})", flush=True)
    with log_path.open(mode, encoding="utf-8") as log:
        if append:
            log.write(f"\n\n=== Resuming at {utc_now()} ===\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
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
        raise RuntimeError(f"Command returned {returncode}; see {log_path}")


def runtime_environment() -> Dict[str, Any]:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,numpy,torch; "
                "print(json.dumps({'numpy':numpy.__version__,'torch':torch.__version__,"
                "'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(PHYSICAL_GPU),
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    import json

    return {
        "recorded_utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": json.loads(probe.stdout),
        "physical_gpu": PHYSICAL_GPU,
        "gpu_probe": gpu,
    }


def validate_completed_task(
    task: Mapping[str, Any], metadata: Mapping[str, Any]
) -> Dict[str, Any] | None:
    complete_path = Path(task["run_dir"]) / "task_complete.json"
    if not complete_path.is_file():
        return None
    complete = load_json(complete_path)
    if complete.get("task_id") != task["task_id"]:
        raise ValueError(f"Completion marker belongs to another task: {complete_path}")
    if complete.get("source_digest") != metadata["source_provenance"]["source_digest"]:
        raise ValueError(f"Completion source digest mismatch: {complete_path}")
    expected_cases = read_bundle(Path(task["test_manifest"]))
    checkpoint_hashes: Dict[str, str] = {}
    for epoch in task["checkpoint_epochs"]:
        checkpoint = checkpoint_path(task, int(epoch))
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[str(epoch)] = file_sha256(checkpoint)
        for inference_seed in task["inference_seeds"]:
            verify_export(
                export_dir(task, int(epoch), int(inference_seed)), expected_cases
            )
    if complete.get("checkpoint_sha256") != checkpoint_hashes:
        raise ValueError(f"Checkpoint hashes changed: {task['task_id']}")
    return complete


def run_task(study_dir: Path, task_id: str) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    diffact_root = Path(metadata["diffact_root"])
    verify_source_digest(metadata, diffact_root)
    task = load_task(study_dir, task_id)
    if int(task["physical_gpu"]) != PHYSICAL_GPU:
        raise ValueError(f"Task is not pinned to physical GPU {PHYSICAL_GPU}")
    config = load_json(Path(task["config_path"]))
    assert_variant_config(config, task, metadata)
    existing = validate_completed_task(task, metadata)
    if existing is not None:
        print(f"COMPLETE {task_id}: validated existing artifacts", flush=True)
        return

    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(task["state_path"])
    status: Dict[str, Any] = {
        "task_id": task_id,
        "status": "running",
        "stage": "initializing",
        "physical_gpu": PHYSICAL_GPU,
        "class_specific_onset_loss_weight": task[
            "class_specific_onset_loss_weight"
        ],
        "started_utc": utc_now(),
        "pid": os.getpid(),
    }
    atomic_write_json(status_path, status)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    try:
        environment_path = run_dir / "runtime_environment.json"
        if not environment_path.is_file():
            atomic_write_json(environment_path, runtime_environment())

        final_checkpoint = Path(task["final_checkpoint"])
        model_dir = Path(task["model_dir"])
        latest = model_dir / "latest.pt"
        if not final_checkpoint.is_file() or final_checkpoint.stat().st_size <= 0:
            orphan_checkpoints = list(model_dir.glob("epoch-*.model"))
            if orphan_checkpoints and not latest.is_file():
                raise RuntimeError(
                    f"Unsafe resume state: checkpoints exist without latest.pt: {model_dir}"
                )
            status.update(
                {
                    "stage": "training",
                    "resuming_from_latest": latest.is_file(),
                    "updated_utc": utc_now(),
                }
            )
            atomic_write_json(status_path, status)
            stream_command(
                [
                    sys.executable,
                    "-u",
                    "main.py",
                    "--config",
                    task["config_path"],
                    "--device",
                    str(PHYSICAL_GPU),
                ],
                cwd=diffact_root,
                log_path=run_dir / "train.log",
                environment=environment,
                append=latest.is_file(),
            )
        if not final_checkpoint.is_file() or final_checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(final_checkpoint)

        checkpoint_hashes: Dict[str, str] = {}
        for epoch in task["checkpoint_epochs"]:
            checkpoint = checkpoint_path(task, int(epoch))
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise FileNotFoundError(checkpoint)
            checkpoint_hashes[str(epoch)] = file_sha256(checkpoint)
        atomic_write_json(
            run_dir / "train_complete.json",
            {
                "task_id": task_id,
                "completed_utc": utc_now(),
                "class_specific_onset_loss_weight": task[
                    "class_specific_onset_loss_weight"
                ],
                "checkpoint_sha256": checkpoint_hashes,
                "checkpoint_selection": "pre_specified_grid_no_test_selection",
                "source_digest": metadata["source_provenance"]["source_digest"],
            },
        )

        expected_cases = read_bundle(Path(task["test_manifest"]))
        exports = []
        for epoch in task["checkpoint_epochs"]:
            for inference_seed in task["inference_seeds"]:
                status.update(
                    {
                        "stage": "activity_and_onset_export",
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(inference_seed),
                        "updated_utc": utc_now(),
                    }
                )
                atomic_write_json(status_path, status)
                output_dir = export_dir(task, int(epoch), int(inference_seed))
                complete_path = output_dir / "export_complete.json"
                if complete_path.is_file():
                    summary = verify_export(output_dir, expected_cases)
                else:
                    stream_command(
                        [
                            sys.executable,
                            "-u",
                            str(SCRIPT_DIR / "export_seeded_outputs.py"),
                            "--study-dir",
                            str(study_dir),
                            "--task-id",
                            task_id,
                            "--checkpoint-epoch",
                            str(epoch),
                            "--inference-seed",
                            str(inference_seed),
                            "--device",
                            str(PHYSICAL_GPU),
                        ],
                        cwd=Path(SCRIPT_DIR.parents[1]),
                        log_path=output_dir / "export.log",
                        environment=environment,
                        append=False,
                    )
                    summary = verify_export(output_dir, expected_cases)
                exports.append(
                    {
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(inference_seed),
                        "output_dir": str(output_dir),
                        "export_complete_sha256": file_sha256(complete_path),
                        "case_count": summary["case_count"],
                        "frame_count": summary["frame_count"],
                    }
                )

        complete = {
            "task_id": task_id,
            "status": "complete",
            "completed_utc": utc_now(),
            "class_specific_onset_loss_weight": task[
                "class_specific_onset_loss_weight"
            ],
            "physical_gpu": PHYSICAL_GPU,
            "checkpoint_sha256": checkpoint_hashes,
            "exports": exports,
            "source_digest": metadata["source_provenance"]["source_digest"],
        }
        atomic_write_json(run_dir / "task_complete.json", complete)
        status.update(
            {
                "status": "complete",
                "stage": "complete",
                "completed_utc": utc_now(),
            }
        )
        atomic_write_json(status_path, status)
        print(f"COMPLETE {task_id}", flush=True)
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
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    run_task(args.study_dir.resolve(), args.task_id)


if __name__ == "__main__":
    main()
