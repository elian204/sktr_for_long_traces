#!/usr/bin/env python3
"""Prepare and optionally execute low-data DiffAct training runs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from low_data_common import (
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_EXPERIMENT_DIR,
    FRACTIONS,
    LOW_DATA_PROTOCOL_VERSION,
    SEEDS,
    read_bundle,
    read_case_manifest,
)


OFFICIAL_NUM_EPOCHS = {
    "gtea": 10001,
    "50salads": 5001,
}


def dataset_prefix(dataset: str) -> str:
    return {"50salads": "50salads", "gtea": "GTEA", "breakfast": "Breakfast"}[dataset]


def iter_requested(
    seeds: Iterable[int],
    fractions: Iterable[int],
    max_runs: Optional[int],
) -> Iterable[tuple[int, int]]:
    count = 0
    for seed in seeds:
        for fraction in fractions:
            if max_runs is not None and count >= max_runs:
                return
            count += 1
            yield seed, fraction


def find_existing_checkpoint(run_dir: Path, naming: str) -> Optional[Path]:
    model_dir = run_dir / "training" / naming
    latest = model_dir / "latest.pt"
    return latest if latest.is_file() and latest.stat().st_size > 0 else None


def expected_final_checkpoint(run_dir: Path, naming: str, num_epochs: int) -> Path:
    return run_dir / "training" / naming / f"epoch-{num_epochs - 1}.model"


def completed_checkpoint(run_dir: Path, naming: str, num_epochs: int) -> Optional[Path]:
    expected_checkpoint = expected_final_checkpoint(run_dir, naming, num_epochs)
    complete_path = run_dir / "train_complete.json"
    if complete_path.is_file():
        try:
            payload = json.loads(complete_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        checkpoint = Path(str(payload.get("final_checkpoint", "")))
        if (
            checkpoint == expected_checkpoint
            and checkpoint.is_file()
            and int(payload.get("num_epochs", -1)) == int(num_epochs)
        ):
            return checkpoint

    if expected_checkpoint.is_file():
        return expected_checkpoint
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--fractions", nargs="+", type=int, default=list(FRACTIONS))
    parser.add_argument("--max-runs", type=int, default=None, help="Pilot limiter over seed/fraction pairs.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override DiffAct epochs for a labeled pilot.")
    parser.add_argument("--execute", action="store_true", help="Actually run training. Without this, commands are written only.")
    parser.add_argument("--force", action="store_true", help="Run even if a checkpoint already exists.")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    diffact_root = args.diffact_root.resolve()
    metadata = json.loads((experiment_dir / "experiment_metadata.json").read_text())
    dataset = metadata["dataset"]
    fold = int(metadata["fold"])
    protocol_version = metadata.get("protocol_version")
    if protocol_version not in (None, LOW_DATA_PROTOCOL_VERSION):
        raise ValueError(f"Unsupported low-data protocol version: {protocol_version}")
    prefix = dataset_prefix(dataset)
    baseline_config = diffact_root / "configs" / f"{prefix}-Trained-S{fold}.json"
    if not baseline_config.is_file():
        raise FileNotFoundError(f"Missing baseline config: {baseline_config}")
    baseline = json.loads(baseline_config.read_text())
    official_num_epochs = OFFICIAL_NUM_EPOCHS.get(dataset)
    if official_num_epochs is not None and int(baseline["num_epochs"]) != official_num_epochs:
        raise ValueError(
            f"{baseline_config} has num_epochs={baseline['num_epochs']}; "
            f"the official {dataset} schedule requires {official_num_epochs}"
        )

    official_train_path = Path(metadata["official_train_split_path"])
    official_test_path = Path(metadata["official_test_split_path"])
    official_train_cases = read_bundle(official_train_path)
    official_test_cases = read_bundle(official_test_path)
    official_train_set = set(official_train_cases)
    official_test_set = set(official_test_cases)
    if official_train_set.intersection(official_test_set):
        raise ValueError("Official train and test bundles overlap")

    all_commands: List[str] = []
    prepared = 0
    executed = 0
    skipped = 0

    for seed, fraction in iter_requested(args.seeds, args.fractions, args.max_runs):
        run_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}"
        run_dir.mkdir(parents=True, exist_ok=True)
        naming = f"lowdata_{dataset}_fold{fold}_seed{seed}_frac{fraction}"
        config_path = run_dir / "config.json"
        subset_manifest = (
            experiment_dir
            / "manifests"
            / f"seed_{seed}"
            / f"train_cases_frac_{fraction}.txt"
        )
        train_cases = read_case_manifest(subset_manifest)
        if not train_cases:
            raise ValueError(f"Missing or empty training subset: {subset_manifest}")
        train_case_set = set(train_cases)
        if len(train_case_set) != len(train_cases):
            raise ValueError(f"Training subset contains duplicate cases: {subset_manifest}")
        if not train_case_set.issubset(official_train_set):
            raise ValueError(f"Training subset contains cases outside official train: {subset_manifest}")
        if train_case_set.intersection(official_test_set):
            raise ValueError(f"Training subset contains official test cases: {subset_manifest}")
        actual_train_case_count = len(train_cases)
        if protocol_version == LOW_DATA_PROTOCOL_VERSION:
            recorded_count = metadata.get("expected_subset_case_counts", {}).get(str(fraction))
            if recorded_count is None or int(recorded_count) != actual_train_case_count:
                raise ValueError(
                    f"Training subset count {actual_train_case_count} does not match protocol "
                    f"metadata for fraction {fraction}: {recorded_count}"
                )

        params: Dict[str, Any] = dict(baseline)
        params["naming"] = naming
        params["root_data_dir"] = str(
            experiment_dir / "diffact_dataset_views" / f"seed_{seed}" / f"frac_{fraction}"
        )
        params["split_id"] = 1
        params["result_dir"] = str(run_dir / "training")
        params["random_seed"] = int(seed)
        params["initialization_seed"] = int(seed)
        params["low_data_fraction"] = int(fraction)
        params["low_data_seed"] = int(seed)
        params["low_data_protocol_version"] = protocol_version or "legacy-validation-pilot"
        params["actual_train_case_count"] = actual_train_case_count
        params["training_subset_manifest"] = str(subset_manifest)
        params["official_train_split_path"] = str(official_train_path)
        params["official_train_split_sha256"] = metadata.get("official_train_split_sha256")
        params["official_test_split_path"] = str(official_test_path)
        params["official_test_split_sha256"] = metadata.get("official_test_split_sha256")
        params["evaluate_during_training"] = False
        params["log_train_results"] = False
        params["source_baseline_config"] = str(baseline_config)
        if args.num_epochs is not None:
            params["num_epochs"] = int(args.num_epochs)
            params["pilot_num_epochs_override"] = True
        num_epochs = int(params["num_epochs"])
        log_freq = int(params["log_freq"])
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        if (num_epochs - 1) % log_freq != 0:
            raise ValueError(
                f"Final epoch {num_epochs - 1} is not a checkpoint epoch for log_freq={log_freq}; "
                "choose a pre-specified schedule whose final epoch is checkpointed"
            )
        final_checkpoint = expected_final_checkpoint(run_dir, naming, num_epochs)
        params["official_num_epochs"] = official_num_epochs
        params["pre_specified_final_epoch"] = num_epochs - 1
        params["pre_specified_final_checkpoint"] = str(final_checkpoint)
        params["checkpoint_selection"] = "pre_specified_final_epoch_no_test_selection"

        config_path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
        command = (
            f"cd {shlex.quote(str(diffact_root))} && "
            f"{shlex.quote(sys.executable)} -u main.py "
            f"--config {shlex.quote(str(config_path))} --device {args.device}"
        )
        all_commands.append(command)
        prepared += 1

        checkpoint = completed_checkpoint(run_dir, naming, num_epochs)
        resume_checkpoint = find_existing_checkpoint(run_dir, naming)
        run_meta = {
            "protocol_version": protocol_version or "legacy-validation-pilot",
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "initialization_seed": seed,
            "fraction": fraction,
            "actual_train_case_count": actual_train_case_count,
            "training_subset_manifest": str(subset_manifest),
            "official_train_split_path": str(official_train_path),
            "official_train_split_sha256": metadata.get("official_train_split_sha256"),
            "official_test_split_path": str(official_test_path),
            "official_test_split_sha256": metadata.get("official_test_split_sha256"),
            "command": command,
            "config": str(config_path),
            "log": str(run_dir / "train.log"),
            "baseline_config": str(baseline_config),
            "official_num_epochs": official_num_epochs,
            "num_epochs_override": args.num_epochs,
            "num_epochs": num_epochs,
            "periodic_evaluation": False,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "expected_final_checkpoint": str(final_checkpoint),
            "pre_specified_final_epoch": num_epochs - 1,
            "checkpoint_selection": "pre_specified_final_epoch_no_test_selection",
            "best_checkpoint": None,
            "status": "complete_existing" if checkpoint is not None else "prepared",
        }
        (run_dir / "train_run_metadata.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=True) + "\n"
        )
        if checkpoint is not None and not args.force:
            skipped += 1
            print(f"SKIP seed={seed} frac={fraction}: completed checkpoint {checkpoint}")
            continue

        if not args.execute:
            resume_note = (
                f" (will resume from {resume_checkpoint})"
                if resume_checkpoint is not None
                else ""
            )
            print(f"PREPARED seed={seed} frac={fraction}: {command}{resume_note}")
            continue

        log_path = run_dir / "train.log"
        model_dir = run_dir / "training" / naming
        orphan_epoch_checkpoints = list(model_dir.glob("epoch-*.model"))
        if resume_checkpoint is None and orphan_epoch_checkpoints:
            raise RuntimeError(
                f"Refusing to start fresh in {model_dir}: epoch checkpoints exist but latest.pt "
                "is absent. Move the old run aside or restore its own latest.pt."
            )
        run_meta["status"] = "running"
        (run_dir / "train_run_metadata.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=True) + "\n"
        )
        log_mode = "a" if log_path.exists() and resume_checkpoint is not None else "w"
        with log_path.open(log_mode, encoding="utf-8") as log_f:
            if log_mode == "a":
                log_f.write("\n\n=== Resuming low-data DiffAct training ===\n")
            proc = subprocess.run(
                [sys.executable, "-u", "main.py", "--config", str(config_path), "--device", str(args.device)],
                cwd=str(diffact_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        executed += 1
        if proc.returncode != 0:
            raise RuntimeError(
                f"DiffAct training failed for seed={seed}, fraction={fraction}. "
                f"See {log_path}"
            )
        if not final_checkpoint.is_file():
            raise RuntimeError(
                f"DiffAct training finished but the pre-specified final checkpoint is missing: "
                f"{final_checkpoint}"
            )
        complete_meta = {
            **run_meta,
            "returncode": proc.returncode,
            "completed": True,
            "final_checkpoint": str(final_checkpoint),
            "latest_pt": str(run_dir / "training" / naming / "latest.pt"),
            "status": "complete",
        }
        (run_dir / "train_complete.json").write_text(
            json.dumps(complete_meta, indent=2, sort_keys=True) + "\n"
        )

    commands_path = experiment_dir / "diffact" / "train_commands.sh"
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(all_commands) + "\n")
    commands_path.chmod(0o755)

    print(f"Prepared configs: {prepared}")
    print(f"Executed runs: {executed}")
    print(f"Skipped existing runs: {skipped}")
    print(f"Training commands: {commands_path}")
    if not args.execute:
        print("No training was launched; pass --execute to run the commands.")


if __name__ == "__main__":
    main()
