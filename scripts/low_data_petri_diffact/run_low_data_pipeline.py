#!/usr/bin/env python3
"""Orchestrate low-data DiffAct training, export, Petri postprocessing, and aggregation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

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
PRIMARY_STAGES = ("train", "infer", "petri_test", "aggregate")
COUPLED_CONDITION = "honest_low_data_petri"
DIFFACT_LOW_PETRI_FULL_CONDITION = "structural_prior_full_train_petri"
DIFFACT_FULL_PETRI_LOW_CONDITION = "full_diffact_low_data_petri"
ORACLE_TEST_FOLD_CONDITION = "oracle_test_fold_petri"
PRIMARY_PROTOCOL = "primary_no_validation"
LEGACY_PROTOCOL = "legacy_pilot"
PROTOCOL_CHOICES = ("auto", PRIMARY_PROTOCOL, LEGACY_PROTOCOL)
PETRI_DISCOVERY_NESTED = "nested_train_fraction"
PETRI_DISCOVERY_FULL_TRAIN = "full_train"
PETRI_DISCOVERY_OFFICIAL_TEST = "official_test_fold"


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
    rendered = shlex.join([str(x) for x in command])
    if not execute:
        print(f"DRY-RUN {rendered}", flush=True)
        return 0
    print(f"RUN {rendered}", flush=True)
    return subprocess.run(command, cwd=str(cwd), env=env, check=False).returncode


def resolve_protocol(metadata: Mapping[str, Any], requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if metadata.get("primary_no_validation") is True or metadata.get("validation_enabled") is False:
        return PRIMARY_PROTOCOL
    raw_version = metadata.get("protocol_version", metadata.get("protocol", ""))
    normalized = str(raw_version).strip().lower().replace("-", "_")
    try:
        if float(normalized.lstrip("v")) >= 2:
            return PRIMARY_PROTOCOL
    except ValueError:
        pass
    if (
        normalized.endswith("_v2")
        or normalized.startswith("v2_")
        or any(
            token in normalized
            for token in ("no_validation", "approved", "primary_curve")
        )
    ):
        return PRIMARY_PROTOCOL
    return LEGACY_PROTOCOL


def default_petri_discovery_source(condition: str) -> str:
    if condition == COUPLED_CONDITION:
        return PETRI_DISCOVERY_NESTED
    if condition == DIFFACT_LOW_PETRI_FULL_CONDITION:
        return PETRI_DISCOVERY_FULL_TRAIN
    if condition == ORACLE_TEST_FOLD_CONDITION:
        return PETRI_DISCOVERY_OFFICIAL_TEST
    if condition == DIFFACT_FULL_PETRI_LOW_CONDITION:
        return PETRI_DISCOVERY_NESTED
    raise ValueError(f"Unknown condition for Petri discovery source: {condition!r}")


def build_petri_jobs(
    selected_runs: Sequence[Tuple[int, int]],
    *,
    crossed_controls: bool,
    include_bound_decodes: bool = True,
) -> List[Tuple[int, int, Optional[int], str]]:
    """Build Petri decode jobs for the selected DiffAct seed/fraction pairs.

    Primary protocol defaults include:
    - honest coupled decode for every (seed, f)
    - full-train structural prior for every (seed, f) with f < 100
    - oracle test-fold decode for every (seed, f)
    - D100+P25 process-scarcity control when --crossed-controls and both
      fractions 25 and 100 are present (D100+P100 remains the coupled reference)
    """
    jobs: List[Tuple[int, int, Optional[int], str]] = [
        (seed, fraction, fraction, COUPLED_CONDITION)
        for seed, fraction in selected_runs
    ]
    if include_bound_decodes:
        for seed, fraction in selected_runs:
            if fraction < 100:
                jobs.append(
                    (seed, fraction, 100, DIFFACT_LOW_PETRI_FULL_CONDITION)
                )
            jobs.append((seed, fraction, None, ORACLE_TEST_FOLD_CONDITION))
    if crossed_controls:
        selected_by_seed: dict[int, set[int]] = {}
        for seed, fraction in selected_runs:
            selected_by_seed.setdefault(seed, set()).add(fraction)
        incomplete = [
            seed
            for seed, fractions in selected_by_seed.items()
            if not {25, 100}.issubset(fractions)
        ]
        if incomplete:
            raise ValueError(
                "--crossed-controls requires both fractions 25 and 100 for every selected "
                f"seed; missing for seeds={incomplete}"
            )
        for seed in selected_by_seed:
            # D25+P100 is already covered by full-train bounds when enabled.
            if not include_bound_decodes:
                jobs.append((seed, 25, 100, DIFFACT_LOW_PETRI_FULL_CONDITION))
            jobs.append((seed, 100, 25, DIFFACT_FULL_PETRI_LOW_CONDITION))

    deduplicated: List[Tuple[int, int, Optional[int], str]] = []
    seen: set[Tuple[int, int, Optional[int], str]] = set()
    for job in jobs:
        if job not in seen:
            deduplicated.append(job)
            seen.add(job)
    return deduplicated


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
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_CHOICES,
        default=None,
        help=(
            "Defaults to train/infer/petri_test/aggregate for primary no-validation "
            "metadata and all legacy pilot stages otherwise."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit selected seed/fraction pairs.")
    parser.add_argument("--protocol", choices=PROTOCOL_CHOICES, default="auto")
    parser.add_argument(
        "--crossed-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also decode D100+P25 process-scarcity control when fractions 25 and 100 "
            "are both selected. Full-train D_f+P_100 bounds are controlled by "
            "--petri-bound-decodes."
        ),
    )
    parser.add_argument(
        "--petri-bound-decodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include full-train structural prior (D_f+P_100 for f<100) and "
            "oracle_test_fold_petri decodes. Enabled by default for the primary "
            "protocol; study generators that emit bound tasks separately should "
            "pass --no-petri-bound-decodes."
        ),
    )
    parser.add_argument("--device", default="auto", help="CUDA device index, -1 for CPU, or auto.")
    parser.add_argument("--gpu-candidates", nargs="+", type=int, default=[0, 1, 2, 3])
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
    parser.add_argument(
        "--runtime-pilot-case-limit",
        type=int,
        default=None,
        help=(
            "Limit Petri test decoding to the first N manifest-ordered cases. "
            "Partial outputs are isolated and explicitly marked runtime-only."
        ),
    )
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
    protocol = resolve_protocol(metadata, args.protocol)
    stages = list(
        args.stages
        if args.stages is not None
        else (PRIMARY_STAGES if protocol == PRIMARY_PROTOCOL else STAGE_CHOICES)
    )

    if args.num_epochs is not None and args.num_epochs < 1:
        raise ValueError("--num-epochs must be positive when provided")
    if args.runtime_pilot_case_limit is not None and args.runtime_pilot_case_limit < 1:
        raise ValueError("--runtime-pilot-case-limit must be positive when provided")
    validation_stages = {"petri_val", "petri_test_calibrated"}
    if protocol == PRIMARY_PROTOCOL and validation_stages.intersection(stages):
        raise ValueError(
            "Primary no-validation protocol disables petri_val and "
            "petri_test_calibrated; use the fixed petri_test stage"
        )
    if args.runtime_pilot_case_limit is not None:
        if "petri_test" not in stages:
            raise ValueError("--runtime-pilot-case-limit requires the petri_test stage")
        disallowed = {"petri_val", "petri_test_calibrated", "aggregate"}.intersection(stages)
        if disallowed:
            raise ValueError(
                "Runtime-only case-limited pilots cannot run validation, calibrated, "
                f"or aggregate stages; disallowed={sorted(disallowed)}"
            )

    commands_log = experiment_dir / "pipeline_commands.sh"
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    rendered_commands: List[str] = []
    completed_commands = 0
    selected_runs = list(iter_requested(args.seeds, args.fractions, args.max_runs))
    petri_jobs = build_petri_jobs(
        selected_runs,
        crossed_controls=bool(args.crossed_controls),
        include_bound_decodes=bool(args.petri_bound_decodes),
    )

    for seed, fraction in selected_runs:
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

        if "train" in stages:
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
            rendered_commands.append(shlex.join(cmd))
            rc = run_command(cmd, execute=args.execute, cwd=workspace)
            if rc != 0:
                raise RuntimeError(f"Training stage failed for seed={seed}, fraction={fraction}")
            completed_commands += int(args.execute)

        if "infer" in stages:
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
            cmd += ["--splits", "test"] if protocol == PRIMARY_PROTOCOL else ["--splits", "val", "test"]
            if args.force:
                cmd.append("--force")
            if args.execute:
                cmd.append("--execute")
            env = dict(os.environ)
            if device not in (1, 2, -1):
                env["ALLOW_NON_PREFERRED_GPU"] = "1"
            rendered_commands.append(shlex.join(cmd))
            rc = run_command(cmd, execute=args.execute, cwd=workspace, env=env)
            if rc != 0:
                raise RuntimeError(f"Inference stage failed for seed={seed}, fraction={fraction}")
            completed_commands += int(args.execute)

    for seed, diffact_fraction, petri_fraction, condition in petri_jobs:
        discovery_source = default_petri_discovery_source(condition)
        for stage, split, calibrated in (
            ("petri_val", "val", False),
            ("petri_test", "test", False),
            ("petri_test_calibrated", "test", True),
        ):
            if stage not in stages:
                continue
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "run_petri_postprocessing_low_data.py"),
                "--experiment-dir",
                str(experiment_dir),
                "--data-root",
                str(data_root),
                "--diffact-fractions",
                str(diffact_fraction),
            ]
            if petri_fraction is not None:
                cmd += ["--petri-fractions", str(petri_fraction)]
            cmd += [
                "--condition",
                condition,
                "--protocol",
                protocol,
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
                "--petri-discovery-source",
                discovery_source,
            ]
            cmd.append("--inner-parallel" if args.petri_inner_parallel else "--no-inner-parallel")
            cmd += ["--seeds", str(seed)]
            if calibrated:
                cmd.append("--calibrated")
            if args.runtime_pilot_case_limit is not None:
                cmd += ["--runtime-pilot-case-limit", str(args.runtime_pilot_case_limit)]
            if args.force:
                cmd.append("--force")
            if args.execute:
                cmd.append("--execute")
            rendered_commands.append(shlex.join(cmd))
            rc = run_command(cmd, execute=args.execute, cwd=workspace)
            if rc != 0:
                raise RuntimeError(
                    f"Petri {split} stage failed for seed={seed}, "
                    f"DiffAct={diffact_fraction}, Petri={petri_fraction}, "
                    f"condition={condition}"
                )
            completed_commands += int(args.execute)

    if "aggregate" in stages:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "aggregate_low_data_results.py"),
            "--experiment-dir",
            str(experiment_dir),
        ]
        rendered_commands.append(shlex.join(cmd))
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
