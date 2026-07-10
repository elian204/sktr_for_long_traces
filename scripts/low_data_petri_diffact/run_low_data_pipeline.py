#!/usr/bin/env python3
"""Orchestrate low-data DiffAct training, export, Petri postprocessing, and aggregation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from low_data_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_EXPERIMENT_DIR,
    FRACTIONS,
    SEEDS,
)


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_CHOICES = (
    "train",
    "infer",
    "petri_val",
    "petri_test",
    "petri_test_calibrated",
    "aggregate",
)


def iter_requested(
    seeds: Sequence[int],
    fractions: Sequence[int],
    max_runs: Optional[int],
) -> Iterable[Tuple[int, int]]:
    count = 0
    for seed in seeds:
        for fraction in fractions:
            if max_runs is not None and count >= max_runs:
                return
            count += 1
            yield seed, fraction


def run_command(command: List[str], *, execute: bool, cwd: Path, env: Optional[dict] = None) -> int:
    rendered = " ".join(str(x) for x in command)
    if not execute:
        print(f"DRY-RUN {rendered}", flush=True)
        return 0
    print(f"RUN {rendered}", flush=True)
    return subprocess.run(command, cwd=str(cwd), env=env, check=False).returncode


def parse_gpu_rows(text: str) -> List[Tuple[int, int, int, int]]:
    rows: List[Tuple[int, int, int, int]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            util = int(parts[1].split()[0])
            mem_used = int(parts[2].split()[0])
            mem_total = int(parts[3].split()[0])
        except ValueError:
            continue
        rows.append((idx, util, mem_used, mem_total))
    return rows


def query_gpus() -> List[Tuple[int, int, int, int]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return parse_gpu_rows(proc.stdout)


def choose_gpu(
    candidates: Sequence[int],
    *,
    max_utilization: int,
    min_free_memory_mb: int,
) -> Optional[int]:
    candidate_set = set(int(x) for x in candidates)
    viable: List[Tuple[int, int, int]] = []
    for idx, util, mem_used, mem_total in query_gpus():
        if idx not in candidate_set:
            continue
        free = mem_total - mem_used
        if util <= max_utilization and free >= min_free_memory_mb:
            viable.append((util, -free, idx))
    if not viable:
        return None
    viable.sort()
    return viable[0][2]


def wait_for_gpu(
    candidates: Sequence[int],
    *,
    max_utilization: int,
    min_free_memory_mb: int,
    poll_seconds: int,
    max_wait_seconds: Optional[int],
) -> int:
    start = time.monotonic()
    while True:
        selected = choose_gpu(
            candidates,
            max_utilization=max_utilization,
            min_free_memory_mb=min_free_memory_mb,
        )
        if selected is not None:
            return selected
        if max_wait_seconds is not None and time.monotonic() - start >= max_wait_seconds:
            raise TimeoutError(
                "No candidate GPU became available within "
                f"{max_wait_seconds}s (candidates={list(candidates)}, "
                f"max_utilization={max_utilization}, min_free_memory_mb={min_free_memory_mb})."
            )
        rows = query_gpus()
        print(
            "Waiting for GPU availability; current rows="
            f"{rows}, candidates={list(candidates)}",
            flush=True,
        )
        time.sleep(poll_seconds)


def add_common_selection(command: List[str], seed: int, fraction: int) -> List[str]:
    return command + ["--seeds", str(seed), "--fractions", str(fraction)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--fractions", nargs="+", type=int, default=list(FRACTIONS))
    parser.add_argument("--stages", nargs="+", choices=STAGE_CHOICES, default=list(STAGE_CHOICES))
    parser.add_argument("--max-runs", type=int, default=None, help="Limit selected seed/fraction pairs.")
    parser.add_argument("--device", default="auto", help="CUDA device index, -1 for CPU, or auto.")
    parser.add_argument("--gpu-candidates", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--gpu-max-utilization", type=int, default=30)
    parser.add_argument("--gpu-min-free-memory-mb", type=int, default=16000)
    parser.add_argument("--gpu-poll-seconds", type=int, default=120)
    parser.add_argument("--gpu-max-wait-seconds", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None, help="Pilot-only epoch override for training.")
    parser.add_argument(
        "--petri-inner-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dataset-level SKTR parallelization across validation/test cases.",
    )
    parser.add_argument("--petri-workers", type=int, default=7)
    parser.add_argument(
        "--petri-method",
        choices=["petri_transition_viterbi", "petri_conformance"],
        default="petri_conformance",
    )
    parser.add_argument("--petri-chunk-size", type=int, default=11)
    parser.add_argument("--petri-progress-log-interval-chunks", type=int, default=20)
    parser.add_argument(
        "--petri-discovery-representation",
        choices=["frame_events", "run_collapsed_segments"],
        default="frame_events",
    )
    parser.add_argument("--transition-illegal-penalty", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    workspace = SCRIPT_DIR.parents[2]
    metadata_path = experiment_dir / "experiment_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing experiment metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())

    if args.num_epochs is not None and args.num_epochs < 1:
        raise ValueError("--num-epochs must be positive when provided")

    commands_log = experiment_dir / "pipeline_commands.sh"
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    rendered_commands: List[str] = []
    completed_commands = 0

    for seed, fraction in iter_requested(args.seeds, args.fractions, args.max_runs):
        if args.device == "auto":
            if args.wait_for_gpu and args.execute:
                device = wait_for_gpu(
                    args.gpu_candidates,
                    max_utilization=args.gpu_max_utilization,
                    min_free_memory_mb=args.gpu_min_free_memory_mb,
                    poll_seconds=args.gpu_poll_seconds,
                    max_wait_seconds=args.gpu_max_wait_seconds,
                )
            else:
                device = choose_gpu(
                    args.gpu_candidates,
                    max_utilization=args.gpu_max_utilization,
                    min_free_memory_mb=args.gpu_min_free_memory_mb,
                )
                device = args.gpu_candidates[0] if device is None else device
        else:
            device = int(args.device)

        print(f"\n=== seed={seed} fraction={fraction} device={device} ===", flush=True)

        if "train" in args.stages:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "train_diffact_low_data.py"),
                "--experiment-dir",
                str(experiment_dir),
                "--diffact-root",
                str(diffact_root),
                "--device",
                str(device),
            ]
            cmd = add_common_selection(cmd, seed, fraction)
            if args.num_epochs is not None:
                cmd += ["--num-epochs", str(args.num_epochs)]
            if args.force:
                cmd.append("--force")
            if args.execute:
                cmd.append("--execute")
            rendered_commands.append(" ".join(cmd))
            rc = run_command(cmd, execute=args.execute, cwd=workspace)
            if rc != 0:
                raise RuntimeError(f"Training stage failed for seed={seed}, fraction={fraction}")
            completed_commands += int(args.execute)

        if "infer" in args.stages:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "infer_diffact_softmax.py"),
                "--experiment-dir",
                str(experiment_dir),
                "--diffact-root",
                str(diffact_root),
                "--device",
                str(device),
            ]
            cmd = add_common_selection(cmd, seed, fraction)
            if args.force:
                cmd.append("--force")
            if args.execute:
                cmd.append("--execute")
            env = dict(os.environ)
            if device not in (1, 2, -1):
                env["ALLOW_NON_PREFERRED_GPU"] = "1"
            rendered_commands.append(" ".join(cmd))
            rc = run_command(cmd, execute=args.execute, cwd=workspace, env=env)
            if rc != 0:
                raise RuntimeError(f"Inference stage failed for seed={seed}, fraction={fraction}")
            completed_commands += int(args.execute)

        for stage, split, calibrated in (
            ("petri_val", "val", False),
            ("petri_test", "test", False),
            ("petri_test_calibrated", "test", True),
        ):
            if stage not in args.stages:
                continue
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "run_petri_postprocessing_low_data.py"),
                "--experiment-dir",
                str(experiment_dir),
                "--data-root",
                str(data_root),
                "--split",
                split,
                "--workers",
                str(args.petri_workers),
                "--method",
                args.petri_method,
                "--transition-illegal-penalty",
                str(args.transition_illegal_penalty),
                "--chunk-size",
                str(args.petri_chunk_size),
                "--progress-log-interval-chunks",
                str(args.petri_progress_log_interval_chunks),
                "--petri-discovery-representation",
                args.petri_discovery_representation,
            ]
            cmd.append("--inner-parallel" if args.petri_inner_parallel else "--no-inner-parallel")
            cmd = add_common_selection(cmd, seed, fraction)
            if calibrated:
                cmd.append("--calibrated")
            if args.force:
                cmd.append("--force")
            if args.execute:
                cmd.append("--execute")
            rendered_commands.append(" ".join(cmd))
            rc = run_command(cmd, execute=args.execute, cwd=workspace)
            if rc != 0:
                raise RuntimeError(
                    f"Petri {split} stage failed for seed={seed}, fraction={fraction}"
                )
            completed_commands += int(args.execute)

    if "aggregate" in args.stages:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "aggregate_low_data_results.py"),
            "--experiment-dir",
            str(experiment_dir),
        ]
        rendered_commands.append(" ".join(cmd))
        rc = run_command(cmd, execute=args.execute, cwd=workspace)
        if rc != 0:
            raise RuntimeError("Aggregation stage failed")
        completed_commands += int(args.execute)

    commands_log.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(rendered_commands) + "\n",
        encoding="utf-8",
    )
    commands_log.chmod(0o755)

    mode = "executed" if args.execute else "prepared"
    print(f"\nPipeline {mode}. Commands log: {commands_log}")
    print(f"Completed subprocess commands: {completed_commands}")
    print(f"Experiment metadata dataset={metadata.get('dataset')} fold={metadata.get('fold')}")


if __name__ == "__main__":
    main()
