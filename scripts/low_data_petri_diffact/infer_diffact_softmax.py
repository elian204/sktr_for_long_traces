#!/usr/bin/env python3
"""Export validation/test DiffAct softmax bundles for low-data runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from low_data_common import DEFAULT_DIFFACT_ROOT, DEFAULT_EXPERIMENT_DIR, FRACTIONS, SEEDS


def iter_requested(seeds: Iterable[int], fractions: Iterable[int], max_runs: Optional[int]):
    count = 0
    for seed in seeds:
        for fraction in fractions:
            if max_runs is not None and count >= max_runs:
                return
            count += 1
            yield seed, fraction


def checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"epoch-(\d+)\.model$", path.name)
    return int(match.group(1)) if match else -1


def expected_final_checkpoint(run_dir: Path, naming: str, num_epochs: int) -> Path:
    return run_dir / "training" / naming / f"epoch-{num_epochs - 1}.model"


def find_checkpoint(run_dir: Path, naming: str, num_epochs: int) -> Optional[Path]:
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

    model_dir = run_dir / "training" / naming
    release = model_dir / "release.model"
    if release.is_file():
        return release
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--fractions", nargs="+", type=int, default=list(FRACTIONS))
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    diffact_root = args.diffact_root.resolve()
    metadata = json.loads((experiment_dir / "experiment_metadata.json").read_text())
    dataset = metadata["dataset"]
    fold = int(metadata["fold"])

    commands: List[str] = []
    prepared = 0
    executed = 0
    skipped = 0
    missing_checkpoint = 0

    for seed, fraction in iter_requested(args.seeds, args.fractions, args.max_runs):
        run_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}"
        config_path = run_dir / "config.json"
        naming = f"lowdata_{dataset}_fold{fold}_seed{seed}_frac{fraction}"
        if not config_path.is_file():
            print(f"MISSING config seed={seed} frac={fraction}: {config_path}")
            continue
        config = json.loads(config_path.read_text())
        num_epochs = int(config["num_epochs"])
        checkpoint = find_checkpoint(run_dir, naming, num_epochs)
        if checkpoint is None:
            missing_checkpoint += 1
            print(
                f"MISSING completed checkpoint seed={seed} frac={fraction}: "
                f"expected {expected_final_checkpoint(run_dir, naming, num_epochs)} "
                f"or valid train_complete.json"
            )
            continue

        for split in args.splits:
            output_dir = run_dir / f"softmax_{split}"
            align_dir = experiment_dir / "align" / split
            log_path = run_dir / f"export_softmax_{split}.log"
            command = (
                f"cd {diffact_root} && "
                f"python -u export_softmax.py --config {config_path} "
                f"--checkpoint {checkpoint} --align_dir {align_dir} "
                f"--output_dir {output_dir} --device {args.device}"
            )
            commands.append(command)
            prepared += 1

            if output_dir.is_dir() and (output_dir / "video_index_map.txt").is_file() and not args.force:
                skipped += 1
                print(f"SKIP seed={seed} frac={fraction} split={split}: existing {output_dir}")
                continue
            if not args.execute:
                print(f"PREPARED seed={seed} frac={fraction} split={split}: {command}")
                continue

            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "export_softmax.py",
                        "--config",
                        str(config_path),
                        "--checkpoint",
                        str(checkpoint),
                        "--align_dir",
                        str(align_dir),
                        "--output_dir",
                        str(output_dir),
                        "--device",
                        str(args.device),
                    ],
                    cwd=str(diffact_root),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            executed += 1
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Softmax export failed for seed={seed}, fraction={fraction}, split={split}. "
                    f"See {log_path}"
                )

    commands_path = experiment_dir / "diffact" / "infer_commands.sh"
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n")
    commands_path.chmod(0o755)

    print(f"Prepared exports: {prepared}")
    print(f"Executed exports: {executed}")
    print(f"Skipped existing exports: {skipped}")
    print(f"Runs missing checkpoints: {missing_checkpoint}")
    print(f"Inference commands: {commands_path}")
    if not args.execute:
        print("No inference was launched; pass --execute after training checkpoints exist.")


if __name__ == "__main__":
    main()
