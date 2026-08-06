#!/usr/bin/env python3
"""Conditionally retrain one Phase-B cell with a genuinely held-out carve."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from phase_b_selection_common import atomic_write_json, canonical_digest, file_sha256, load_json, verify_manifest, verify_source
from phase_b_training_common import DATASETS, OFFICIAL_CONFIG


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
    paths = verify_manifest(manifest, full_hash=False)
    decision = load_json(study / "selection" / "step0_decision.json")
    if decision["selected_branch"] != "B_RETRAIN_WITH_HELD_OUT_VALIDATION":
        raise RuntimeError("Retraining is forbidden unless Step-0 selects Branch B")
    if not config.get("fable_approval_digest") or not config["conditional_branch_b_training_allowed"]:
        raise RuntimeError("Conditional Branch-B training is not approved")
    if config["phase_c_allowed"] or config["selection_may_read_test_trajectory"]:
        raise RuntimeError("Phase-C/selection firewall drift")
    task = next((row for row in config["tasks"] if row["dataset"] == dataset and int(row["fold"]) == fold), None)
    if task is None or int(task["device_lane"]) != device:
        raise RuntimeError("Branch-B task/device assignment drift")
    complete_path = study / "status" / f"retrain_{dataset}_fold{fold}.json"
    if complete_path.is_file():
        complete = load_json(complete_path)
        for epoch, expected in complete["checkpoint_sha256"].items():
            path = study / "retrain" / dataset / f"fold{fold}" / "runtime" / "models" / dataset / f"split_{fold}" / f"epoch-{epoch}.model"
            if file_sha256(path) != expected:
                raise RuntimeError("Completed Branch-B checkpoint drift")
    else:
        runtime = study / "retrain" / dataset / f"fold{fold}" / "runtime"
        if runtime.exists():
            raise RuntimeError(f"Partial Branch-B runtime exists; diagnose before retry: {runtime}")
        runtime.mkdir(parents=True)
        source_hashes: dict[str, str] = {}
        for name in ("main.py", "model.py", "batch_gen.py", "eval.py", "train.sh"):
            source = paths[f"runtime_source/{name}"]
            shutil.copy2(source, runtime / name)
            source_hashes[name] = file_sha256(runtime / name)
        dataset_root = runtime / "data" / dataset
        dataset_root.mkdir(parents=True)
        source_root = Path(config["data_root"]) / dataset
        os.symlink(source_root / "features", dataset_root / "features", target_is_directory=True)
        os.symlink(source_root / "groundTruth", dataset_root / "groundTruth", target_is_directory=True)
        os.symlink(source_root / "mapping.txt", dataset_root / "mapping.txt")
        splits = dataset_root / "splits"
        splits.mkdir()
        shutil.copy2(study / "carves" / dataset / f"fold{fold}" / "train_remainder.bundle", splits / f"train.split{fold}.bundle")
        shutil.copy2(study / "carves" / dataset / f"fold{fold}" / "carved_validation.bundle", splits / f"test.split{fold}.bundle")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        command = [
            "/usr/bin/python", "main.py", "--action=train", f"--dataset={dataset}", f"--split={fold}",
            f"--num_epochs={OFFICIAL_CONFIG['num_epochs']}", f"--features_dim={OFFICIAL_CONFIG['features_dim']}",
            f"--bz={OFFICIAL_CONFIG['batch_size']}", f"--lr={OFFICIAL_CONFIG['learning_rate']}",
            f"--num_f_maps={OFFICIAL_CONFIG['num_f_maps']}", f"--num_layers_PG={OFFICIAL_CONFIG['num_layers_PG']}",
            f"--num_layers_R={OFFICIAL_CONFIG['num_layers_R']}", f"--num_R={OFFICIAL_CONFIG['num_R']}",
        ]
        subprocess.run(command, cwd=runtime, env=env, check=True)
        model_root = runtime / "models" / dataset / f"split_{fold}"
        checkpoint_hashes: dict[str, str] = {}
        optimizer_hashes: dict[str, str] = {}
        for epoch in range(1, 101):
            checkpoint = model_root / f"epoch-{epoch}.model"
            optimizer = model_root / f"epoch-{epoch}.opt"
            checkpoint_hashes[str(epoch)] = file_sha256(checkpoint)
            optimizer_hashes[str(epoch)] = file_sha256(optimizer)
        complete = {
            "status": "complete",
            "dataset": dataset,
            "fold": fold,
            "device": device,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "train_bundle_sha256": file_sha256(splits / f"train.split{fold}.bundle"),
            "validation_bundle_sha256": file_sha256(splits / f"test.split{fold}.bundle"),
            "official_config": OFFICIAL_CONFIG,
            "runtime_source_sha256": source_hashes,
            "checkpoint_sha256": checkpoint_hashes,
            "optimizer_sha256": optimizer_hashes,
            "test_data_used_for_training_or_selection": False,
        }
        complete["completion_digest"] = canonical_digest(complete)
        atomic_write_json(complete_path, complete)
    subprocess.run(
        [
            "/usr/bin/python", str(Path(__file__).resolve().parent / "run_phase_b_epoch_grid.py"),
            "--study-dir", str(study), "--scope", "retrained_validation", "--dataset", dataset,
            "--fold", str(fold), "--device", str(device),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    failed = study / "status" / f"retrain_{args.dataset}_fold{args.fold}_failed.json"
    try:
        execute(study, args.dataset, args.fold, args.device)
    except Exception as error:
        atomic_write_json(failed, {"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
