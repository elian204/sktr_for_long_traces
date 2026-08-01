#!/usr/bin/env python3
"""Train and export one approved official-config MS-TCN++ fold."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from phase_b_training_common import (
    DATASETS,
    OFFICIAL_CONFIG,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    verify_manifest,
    verify_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    return parser.parse_args()


def execute(study: Path, dataset: str, fold: int, device: int) -> None:
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    verify_manifest(manifest, full_hash=False)
    preflight = load_json(study / "status" / "preflight_complete.json")
    if preflight["input_manifest_digest"] != manifest["manifest_digest"] or not preflight["full_hash"]:
        raise RuntimeError("Full-hash Phase-B preflight is missing or stale")
    task = next(
        (row for row in config["tasks"] if row["dataset"] == dataset and int(row["fold"]) == fold),
        None,
    )
    if task is None or int(task["device_lane"]) != device:
        raise RuntimeError("Phase-B task/device assignment drift")
    if not config["gpu_training_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("Phase-B GPU training is not approved")
    if config["phase_c_allowed"]:
        raise RuntimeError("Phase C must remain closed during Phase B")
    complete_path = study / "status" / f"{dataset}_fold{fold}_complete.json"
    if complete_path.is_file():
        complete = load_json(complete_path)
        for relative, expected in complete["output_sha256"].items():
            if file_sha256(study / relative) != expected:
                raise RuntimeError(f"Completed Phase-B output drift: {relative}")
        print(f"already complete: {dataset}/fold{fold}")
        return

    runtime = study / "cells" / dataset / f"fold{fold}" / "runtime"
    model_dir = runtime / "models" / dataset / f"split_{fold}"
    export_dir = study / "cells" / dataset / f"fold{fold}" / "export" / "softmax"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    (runtime / "logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    command = [
        "/usr/bin/python",
        "main.py",
        "--action=train",
        f"--dataset={dataset}",
        f"--split={fold}",
        f"--num_epochs={OFFICIAL_CONFIG['num_epochs']}",
        f"--features_dim={OFFICIAL_CONFIG['features_dim']}",
        f"--bz={OFFICIAL_CONFIG['batch_size']}",
        f"--lr={OFFICIAL_CONFIG['learning_rate']}",
        f"--num_f_maps={OFFICIAL_CONFIG['num_f_maps']}",
        f"--num_layers_PG={OFFICIAL_CONFIG['num_layers_PG']}",
        f"--num_layers_R={OFFICIAL_CONFIG['num_layers_R']}",
        f"--num_R={OFFICIAL_CONFIG['num_R']}",
    ]
    subprocess.run(command, cwd=runtime, env=env, check=True)
    checkpoint = model_dir / "epoch-100.model"
    optimizer = model_dir / "epoch-100.opt"
    if not checkpoint.is_file() or not optimizer.is_file():
        raise RuntimeError(f"Final MS-TCN++ checkpoint missing: {dataset}/fold{fold}")
    subprocess.run(
        [
            "/usr/bin/python",
            str(Path(__file__).resolve().parent / "export_mstcn2_official.py"),
            "--runtime",
            str(runtime),
            "--data-root",
            config["data_root"],
            "--dataset",
            dataset,
            "--fold",
            str(fold),
            "--output-dir",
            str(export_dir),
            "--device",
            str(device),
        ],
        env=env,
        check=True,
    )
    outputs = [checkpoint, optimizer, export_dir / "mapping.txt", export_dir / "video_index_map.txt"]
    outputs.extend(sorted(export_dir.glob("*.npy")))
    expected_cases = len(
        [line for line in (Path(config["data_root"]) / dataset / "splits" / f"test.split{fold}.bundle").read_text().splitlines() if line.strip()]
    )
    if len(list(export_dir.glob("*.npy"))) != expected_cases:
        raise RuntimeError("MS-TCN++ export coverage drift")
    completion = {
        "status": "complete",
        "dataset": dataset,
        "fold": fold,
        "device": device,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "test_cases": expected_cases,
        "output_sha256": {
            str(path.relative_to(study)): file_sha256(path) for path in outputs
        },
    }
    completion["completion_digest"] = canonical_digest(completion)
    atomic_write_json(complete_path, completion)
    print(f"complete: {dataset}/fold{fold} cases={expected_cases}")


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    failed = study / "status" / f"{args.dataset}_fold{args.fold}_failed.json"
    try:
        execute(study, args.dataset, args.fold, args.device)
    except Exception as error:
        atomic_write_json(
            failed,
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset,
                "fold": args.fold,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
