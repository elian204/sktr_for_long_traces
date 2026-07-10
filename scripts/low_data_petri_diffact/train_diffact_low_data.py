#!/usr/bin/env python3
"""Prepare and optionally execute low-data DiffAct training runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from low_data_common import DEFAULT_DIFFACT_ROOT, DEFAULT_EXPERIMENT_DIR, FRACTIONS, SEEDS


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
    candidates = sorted(model_dir.glob("epoch-*.model"))
    if candidates:
        return candidates[-1]
    latest = model_dir / "latest.pt"
    return latest if latest.is_file() else None


def checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"epoch-(\d+)\.model$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_epoch_checkpoint(run_dir: Path, naming: str) -> Optional[Path]:
    model_dir = run_dir / "training" / naming
    candidates = sorted(model_dir.glob("epoch-*.model"), key=checkpoint_sort_key)
    return candidates[-1] if candidates else None


def expected_final_checkpoint(run_dir: Path, naming: str, num_epochs: int) -> Path:
    return run_dir / "training" / naming / f"epoch-{num_epochs - 1}.model"


def completed_checkpoint(run_dir: Path, naming: str, num_epochs: int) -> Optional[Path]:
    complete_path = run_dir / "train_complete.json"
    if complete_path.is_file():
        try:
            payload = json.loads(complete_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        checkpoint = Path(str(payload.get("final_checkpoint", "")))
        if checkpoint.is_file() and int(payload.get("num_epochs", -1)) == int(num_epochs):
            return checkpoint

    final_checkpoint = expected_final_checkpoint(run_dir, naming, num_epochs)
    if final_checkpoint.is_file():
        return final_checkpoint
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
    prefix = dataset_prefix(dataset)
    baseline_config = diffact_root / "configs" / f"{prefix}-Trained-S{fold}.json"
    if not baseline_config.is_file():
        raise FileNotFoundError(f"Missing baseline config: {baseline_config}")
    baseline = json.loads(baseline_config.read_text())

    all_commands: List[str] = []
    prepared = 0
    executed = 0
    skipped = 0

    for seed, fraction in iter_requested(args.seeds, args.fractions, args.max_runs):
        run_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}"
        run_dir.mkdir(parents=True, exist_ok=True)
        naming = f"lowdata_{dataset}_fold{fold}_seed{seed}_frac{fraction}"
        config_path = run_dir / "config.json"

        params: Dict[str, Any] = dict(baseline)
        params["naming"] = naming
        params["root_data_dir"] = str(
            experiment_dir / "diffact_dataset_views" / f"seed_{seed}" / f"frac_{fraction}"
        )
        params["split_id"] = 1
        params["result_dir"] = str(run_dir / "training")
        params["random_seed"] = int(seed)
        params["low_data_fraction"] = int(fraction)
        params["low_data_seed"] = int(seed)
        params["source_baseline_config"] = str(baseline_config)
        if args.num_epochs is not None:
            params["num_epochs"] = int(args.num_epochs)
            params["pilot_num_epochs_override"] = True
        num_epochs = int(params["num_epochs"])

        config_path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
        command = (
            f"cd {diffact_root} && "
            f"python -u main.py --config {config_path} --device {args.device}"
        )
        all_commands.append(command)
        prepared += 1

        checkpoint = completed_checkpoint(run_dir, naming, num_epochs)
        if checkpoint is not None and not args.force:
            skipped += 1
            print(f"SKIP seed={seed} frac={fraction}: completed checkpoint {checkpoint}")
            continue

        if not args.execute:
            resume_checkpoint = find_existing_checkpoint(run_dir, naming)
            resume_note = (
                f" (will resume from {resume_checkpoint})"
                if resume_checkpoint is not None
                else ""
            )
            print(f"PREPARED seed={seed} frac={fraction}: {command}{resume_note}")
            continue

        log_path = run_dir / "train.log"
        resume_checkpoint = find_existing_checkpoint(run_dir, naming)
        run_meta = {
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "fraction": fraction,
            "command": command,
            "config": str(config_path),
            "log": str(log_path),
            "baseline_config": str(baseline_config),
            "num_epochs_override": args.num_epochs,
            "num_epochs": num_epochs,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "expected_final_checkpoint": str(expected_final_checkpoint(run_dir, naming, num_epochs)),
            "best_checkpoint": None,
        }
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
        final_checkpoint = find_latest_epoch_checkpoint(run_dir, naming)
        if final_checkpoint is None:
            raise RuntimeError(
                f"DiffAct training finished but no epoch checkpoint was found for "
                f"seed={seed}, fraction={fraction} under {run_dir / 'training' / naming}"
            )
        complete_meta = {
            **run_meta,
            "returncode": proc.returncode,
            "completed": True,
            "final_checkpoint": str(final_checkpoint),
            "latest_pt": str(run_dir / "training" / naming / "latest.pt"),
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
