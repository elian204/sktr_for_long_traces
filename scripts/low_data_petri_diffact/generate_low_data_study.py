#!/usr/bin/env python3
"""Generate immutable all-fold low-data study metadata and launch wrappers.

This command only writes configuration and shell scripts. It never starts tmux,
training, inference, or decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from low_data_common import DEFAULT_DATA_ROOT, DEFAULT_DIFFACT_ROOT, FRACTIONS, WORKSPACE_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = (
    ("gtea", (1, 2, 3, 4), 10001),
    ("50salads", (1, 2, 3, 4, 5), 5001),
)
DATASET_BY_NAME = {name: (name, folds, epochs) for name, folds, epochs in DATASETS}
APPROVED_GPUS = (0, 1, 2, 3)
COUPLED_CONDITION = "honest_low_data_petri"
STRUCTURAL_PRIOR_CONDITION = "structural_prior_full_train_petri"
PROCESS_SCARCITY_CONDITION = "full_diffact_low_data_petri"
ORACLE_CONDITION = "oracle_test_fold_petri"
PETRI_DISCOVERY_NESTED = "nested_train_fraction"
PETRI_DISCOVERY_FULL_TRAIN = "full_train"
PETRI_DISCOVERY_OFFICIAL_TEST = "official_test_fold"
PROVENANCE_FILES = (
    "src/incremental_softmax_recovery.py",
    "scripts/low_data_petri_diffact/aggregate_low_data_results.py",
    "scripts/low_data_petri_diffact/create_low_data_splits.py",
    "scripts/low_data_petri_diffact/generate_low_data_study.py",
    "scripts/low_data_petri_diffact/infer_diffact_softmax.py",
    "scripts/low_data_petri_diffact/low_data_common.py",
    "scripts/low_data_petri_diffact/low_data_study_status.py",
    "scripts/low_data_petri_diffact/run_low_data_pipeline.py",
    "scripts/low_data_petri_diffact/run_low_data_study_task.py",
    "scripts/low_data_petri_diffact/run_petri_postprocessing_low_data.py",
    "scripts/low_data_petri_diffact/study_aggregation.py",
    "scripts/low_data_petri_diffact/train_diffact_low_data.py",
    "scripts/low_data_petri_diffact/verify_low_data_manifests.py",
)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "repo": str(repo.resolve()),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def source_provenance(diffact_root: Path) -> Dict[str, Any]:
    file_hashes: Dict[str, str] = {}
    for relative in PROVENANCE_FILES:
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required study source file is missing: {path}")
        file_hashes[relative] = file_sha256(path)
    for relative in ("main.py", "export_softmax.py", "dataset.py", "utils.py"):
        path = diffact_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required DiffAct source file is missing: {path}")
        file_hashes[f"baselines/DiffAct/{relative}"] = file_sha256(path)
    return {
        "workspace_git": git_provenance(WORKSPACE_ROOT),
        "diffact_git": git_provenance(diffact_root),
        "file_sha256": file_hashes,
        "source_digest": canonical_digest(file_hashes),
    }


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def render(command: List[str]) -> str:
    return shlex.join(command)


def build_bound_postprocess_specs(
    fractions: Sequence[int],
    *,
    runtime_pilot: bool,
) -> List[Tuple[int, Optional[int], str, str, str]]:
    """Return (diffact_fraction, petri_fraction, condition, discovery_source, task_suffix)."""
    if runtime_pilot:
        return []
    specs: List[Tuple[int, Optional[int], str, str, str]] = []
    fraction_set = set(fractions)
    for fraction in fractions:
        if fraction < 100:
            specs.append(
                (
                    fraction,
                    100,
                    STRUCTURAL_PRIOR_CONDITION,
                    PETRI_DISCOVERY_FULL_TRAIN,
                    f"d{fraction}_p100",
                )
            )
        specs.append(
            (
                fraction,
                None,
                ORACLE_CONDITION,
                PETRI_DISCOVERY_OFFICIAL_TEST,
                f"d{fraction}_ptest_test",
            )
        )
    if {25, 100}.issubset(fraction_set):
        specs.append(
            (
                100,
                25,
                PROCESS_SCARCITY_CONDITION,
                PETRI_DISCOVERY_NESTED,
                "d100_p25",
            )
        )
    return specs


def build_spec(args: argparse.Namespace) -> Dict[str, Any]:
    study_dir = args.study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    seeds = sorted(set(int(seed) for seed in args.seeds))
    requested_fractions = (
        args.fractions
        if args.fractions is not None
        else ([25, 100] if args.runtime_pilot else list(FRACTIONS))
    )
    fractions = sorted(set(int(fraction) for fraction in requested_fractions))
    if not seeds or seeds[0] != 0:
        raise ValueError("The staged study must start with subset seed 0")
    expected_fractions = [25, 100] if args.runtime_pilot else list(FRACTIONS)
    if fractions != expected_fractions:
        if args.runtime_pilot:
            raise ValueError("The approved runtime pilot uses exactly fractions 25 and 100")
        raise ValueError(f"Approved fractions are exactly {list(FRACTIONS)}")
    if args.runtime_pilot:
        selected_datasets = (("gtea", (1,), 10001),)
    elif args.datasets:
        unknown = [name for name in args.datasets if name not in DATASET_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}; choose from {sorted(DATASET_BY_NAME)}")
        selected_datasets = tuple(DATASET_BY_NAME[name] for name in args.datasets)
    else:
        selected_datasets = DATASETS
    run_type = "runtime_pilot" if args.runtime_pilot else "final"
    bound_specs = build_bound_postprocess_specs(fractions, runtime_pilot=args.runtime_pilot)

    experiments: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    task_index = 0
    for dataset, folds, published_epochs in selected_datasets:
        for fold in folds:
            experiment_dir = study_dir / "experiments" / dataset / f"fold_{fold}"
            experiments.append(
                {
                    "dataset": dataset,
                    "official_fold": fold,
                    "experiment_dir": str(experiment_dir),
                    "published_num_epochs": published_epochs,
                }
            )
            for seed in seeds:
                honest_task_ids: Dict[int, str] = {}
                honest_task_gpus: Dict[int, int] = {}
                for fraction in fractions:
                    gpu = APPROVED_GPUS[task_index % len(APPROVED_GPUS)]
                    task_id = f"{dataset}_fold{fold}_seed{seed}_d{fraction}_p{fraction}"
                    honest_task_ids[fraction] = task_id
                    honest_task_gpus[fraction] = gpu
                    log_path = study_dir / "logs" / f"{task_id}.log"
                    command = [
                        sys.executable,
                        str(SCRIPT_DIR / "run_low_data_pipeline.py"),
                        "--experiment-dir",
                        str(experiment_dir),
                        "--data-root",
                        str(data_root),
                        "--diffact-root",
                        str(diffact_root),
                        "--seeds",
                        str(seed),
                        "--fractions",
                        str(fraction),
                        "--protocol",
                        "primary_no_validation",
                        "--stages",
                        "train",
                        "infer",
                        "petri_test",
                        "--no-petri-bound-decodes",
                        "--device",
                        str(gpu),
                        "--petri-method",
                        "petri_conformance",
                        "--petri-chunk-size",
                        "11",
                        "--petri-workers",
                        str(args.petri_workers),
                        "--petri-discovery-representation",
                        "frame_events",
                    ]
                    if args.runtime_pilot:
                        command += [
                            "--runtime-pilot-case-limit",
                            str(args.runtime_pilot_case_limit),
                        ]
                    command.append("--execute")
                    tasks.append(
                        {
                            "task_id": task_id,
                            "dataset": dataset,
                            "official_fold": fold,
                            "subset_seed": seed,
                            "diffact_fraction": fraction,
                            "petri_fraction": fraction,
                            "condition": COUPLED_CONDITION,
                            "petri_discovery_source": PETRI_DISCOVERY_NESTED,
                            "oracle_upper_bound": False,
                            "run_type": run_type,
                            "expected_num_epochs": published_epochs,
                            "gpu": gpu,
                            "queue": f"gpu{gpu}",
                            "requires_gpu": True,
                            "depends_on_task_ids": [],
                            "experiment_dir": str(experiment_dir),
                            "command": command,
                            "log_path": str(log_path),
                            "state_path": str(study_dir / "state" / f"{task_id}.json"),
                        }
                    )
                    task_index += 1
                for (
                    diffact_fraction,
                    petri_fraction,
                    condition,
                    discovery_source,
                    task_suffix,
                ) in bound_specs:
                    # Keep each overlay behind the honest task that produces its
                    # DiffAct softmax. This removes cross-queue timing assumptions.
                    gpu = honest_task_gpus[diffact_fraction]
                    producer_task_id = honest_task_ids[diffact_fraction]
                    task_id = f"{dataset}_fold{fold}_seed{seed}_{task_suffix}"
                    log_path = study_dir / "logs" / f"{task_id}.log"
                    command = [
                        sys.executable,
                        str(SCRIPT_DIR / "run_petri_postprocessing_low_data.py"),
                        "--experiment-dir",
                        str(experiment_dir),
                        "--data-root",
                        str(data_root),
                        "--seeds",
                        str(seed),
                        "--diffact-fractions",
                        str(diffact_fraction),
                    ]
                    if petri_fraction is not None:
                        command += ["--petri-fractions", str(petri_fraction)]
                    command += [
                        "--condition",
                        condition,
                        "--protocol",
                        "primary_no_validation",
                        "--split",
                        "test",
                        "--method",
                        "petri_conformance",
                        "--chunk-size",
                        "11",
                        "--workers",
                        str(args.petri_workers),
                        "--petri-discovery-representation",
                        "frame_events",
                        "--petri-discovery-source",
                        discovery_source,
                        "--inner-parallel",
                        "--execute",
                    ]
                    tasks.append(
                        {
                            "task_id": task_id,
                            "dataset": dataset,
                            "official_fold": fold,
                            "subset_seed": seed,
                            "diffact_fraction": diffact_fraction,
                            "petri_fraction": petri_fraction,
                            "condition": condition,
                            "petri_discovery_source": discovery_source,
                            "oracle_upper_bound": condition == ORACLE_CONDITION,
                            "run_type": "final",
                            "expected_num_epochs": published_epochs,
                            "gpu": gpu,
                            "queue": f"gpu{gpu}",
                            "requires_gpu": False,
                            "depends_on_task_ids": [producer_task_id],
                            "experiment_dir": str(experiment_dir),
                            "command": command,
                            "log_path": str(log_path),
                            "state_path": str(study_dir / "state" / f"{task_id}.json"),
                        }
                    )
                    task_index += 1

    return {
        "study_id": args.study_id,
        "workspace_root": str(WORKSPACE_ROOT.resolve()),
        "study_dir": str(study_dir),
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "source_provenance": source_provenance(diffact_root),
        "datasets": [
            {
                "dataset": dataset,
                "official_folds": list(folds),
                "published_num_epochs": epochs,
            }
            for dataset, folds, epochs in selected_datasets
        ],
        "seeds": seeds,
        "fractions": fractions,
        "run_type": run_type,
        "runtime_pilot_case_limit": (
            args.runtime_pilot_case_limit if args.runtime_pilot else None
        ),
        "validation_policy": "No validation carve in the approved primary curve.",
        "conditions": [
            "honest_low_data_petri (DiffAct_f + Petri_f; primary realistic curve)",
            "structural_prior_full_train_petri (DiffAct_f + Petri from train_pool; f<100)",
            "oracle_test_fold_petri (DiffAct_f + Petri from official test GT; all f)",
            "full_diffact_low_data_petri (D100+P25 process-scarcity control only)",
        ],
        "bound_decode_matrix": {
            "honest_low_data_petri": "nested train_cases_frac_f",
            "structural_prior_full_train_petri": "train_pool_cases.txt for every DiffAct f<100",
            "oracle_test_fold_petri": "test_cases.txt for every DiffAct f including 100",
            "full_diffact_low_data_petri": "D100+P25 only",
            "dedup_rule": "D100+P100 remains honest only; no structural duplicate",
        },
        "decoder": {
            "method": "petri_conformance",
            "chunk_size": 11,
            "conditioning_state_mode": "topm",
            "conditioning_top_m": 1,
            "candidate_top_k": 3,
            "restrict_log_moves": True,
            "restrict_model_moves_to_tau": True,
            "max_consecutive_tau_moves": 8,
        },
        "gpu_policy": {
            "physical_gpu_ids": list(APPROVED_GPUS),
            "one_queue_per_gpu": True,
            "fallback_gpu_ids": [],
            "wait_instead_of_fallback": False,
            "exclusive_access_assumed": False,
            "shared_with_other_users": True,
            "launch_policy": "shared_concurrent_use_allowed",
        },
        "experiments": experiments,
        "tasks": tasks,
    }


def write_metadata(study_dir: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    path = study_dir / "study_metadata.json"
    digest = canonical_digest(spec)
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("spec_sha256") != digest:
            raise FileExistsError(
                f"Refusing to mutate immutable study metadata at {path}. "
                "Choose a new --study-dir or restore the original specification."
            )
        return existing
    payload = {
        "schema_version": 1,
        "immutable": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": digest,
        **spec,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_prepare_script(metadata: Dict[str, Any], study_dir: Path) -> Path:
    source_check = [
        sys.executable,
        str(SCRIPT_DIR / "run_low_data_study_task.py"),
        "--study-dir",
        str(study_dir),
        "--task-id",
        str(metadata["tasks"][0]["task_id"]),
        "--dry-run",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Fail before writing manifests if the immutable study source changed.",
        f"{render(source_check)} > /dev/null",
        "",
    ]
    for experiment in metadata["experiments"]:
        experiment_dir = experiment["experiment_dir"]
        dataset = experiment["dataset"]
        fold = experiment["official_fold"]
        log_path = study_dir / "logs" / f"prepare_{dataset}_fold{fold}.log"
        create = [
            sys.executable,
            str(SCRIPT_DIR / "create_low_data_splits.py"),
            "--dataset",
            dataset,
            "--fold",
            str(fold),
            "--data-root",
            metadata["data_root"],
            "--diffact-root",
            metadata["diffact_root"],
            "--experiment-dir",
            experiment_dir,
            "--seeds",
            *[str(seed) for seed in metadata["seeds"]],
            "--fractions",
            *[str(fraction) for fraction in metadata["fractions"]],
        ]
        verify = [
            sys.executable,
            str(SCRIPT_DIR / "verify_low_data_manifests.py"),
            "--experiment-dir",
            experiment_dir,
        ]
        lines.extend(
            [
                f"mkdir -p {shlex.quote(str(log_path.parent))}",
                f"{render(create)} > {shlex.quote(str(log_path))} 2>&1",
                f"{render(verify)} >> {shlex.quote(str(log_path))} 2>&1",
                "",
            ]
        )
    path = study_dir / "prepare_study.sh"
    write_executable(path, "\n".join(lines))
    return path


def write_queue_scripts(metadata: Dict[str, Any], study_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for gpu in APPROVED_GPUS:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Physical GPU {gpu} only. This queue never falls back to another GPU.",
        ]
        for task in metadata["tasks"]:
            if int(task["gpu"]) != gpu:
                continue
            wrapper_command = [
                sys.executable,
                str(SCRIPT_DIR / "run_low_data_study_task.py"),
                "--study-dir",
                str(study_dir),
                "--task-id",
                task["task_id"],
            ]
            lines.append(render(wrapper_command))
        path = study_dir / f"queue_gpu{gpu}.sh"
        write_executable(path, "\n".join(lines) + "\n")
        paths.append(path)
    return paths


def write_aggregate_script(study_dir: Path) -> Path:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "aggregate_low_data_results.py"),
        "--study-dir",
        str(study_dir),
    ]
    log_path = study_dir / "logs" / "aggregate_study.log"
    path = study_dir / "aggregate_study.sh"
    write_executable(
        path,
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        f"{render(command)} > {shlex.quote(str(log_path))} 2>&1\n",
    )
    return path


def write_tmux_commands(metadata: Dict[str, Any], study_dir: Path) -> tuple[Path, List[str]]:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(metadata["study_id"])).strip("_")
    commands = []
    for gpu in APPROVED_GPUS:
        session = shlex.quote(f"{slug}_gpu{gpu}")
        queue_command = shlex.quote(f"bash {study_dir / f'queue_gpu{gpu}.sh'}")
        commands.append(f"tmux new-session -d -s {session} {queue_command}")
    path = study_dir / "tmux_commands.sh"
    write_executable(
        path,
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        "# Running this file starts one detached queue per approved physical GPU.\n"
        + "\n".join(commands)
        + "\n",
    )
    return path, commands


def write_instructions(
    metadata: Dict[str, Any],
    study_dir: Path,
    tmux_commands: List[str],
) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(metadata["study_id"])).strip("_")
    status_command = render(
        [
            sys.executable,
            str(SCRIPT_DIR / "low_data_study_status.py"),
            "--study-dir",
            str(study_dir),
        ]
    )
    lines = [
        "# Low-data all-fold study",
        "",
        "No jobs were launched by the generator.",
        "",
        "1. Prepare and verify manifests:",
        f"   `{study_dir / 'prepare_study.sh'}`",
        "2. Review the immutable metadata and queue scripts.",
        "3. Start the four detached queues only when ready:",
        *[f"   `{command}`" for command in tmux_commands],
        "4. Attach to a queue (detach with Ctrl+b, d):",
        *[f"   `tmux attach -t {slug}_gpu{gpu}`" for gpu in APPROVED_GPUS],
        "5. Refresh status:",
        f"   `{status_command}`",
        "6. After all final artifacts complete, aggregate:",
        f"   `{study_dir / 'aggregate_study.sh'}`",
        "",
        f"Per-task logs: `{study_dir / 'logs'}`",
        f"Persistent heartbeat/sentinel files: `{study_dir / 'state'}`",
        "",
        "GPU policy: shared physical GPUs 0-3, one serial queue per GPU, no fallback or exclusivity assumption.",
        "The underlying stages skip validated completed artifacts and SKTR case outputs are resumable.",
    ]
    path = study_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "low_data_decoding_study",
    )
    parser.add_argument("--study-id", default="low_data_gtea_50salads_seed0")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--fractions", nargs="+", type=int, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_BY_NAME),
        default=None,
        help="Subset of approved datasets (default: gtea and 50salads). Ignored for --runtime-pilot.",
    )
    parser.add_argument("--petri-workers", type=int, default=7)
    parser.add_argument(
        "--runtime-pilot",
        action="store_true",
        help="Generate only GTEA fold 1, seed 0, fractions 25/100 with case-limited SKTR.",
    )
    parser.add_argument("--runtime-pilot-case-limit", type=int, default=1)
    args = parser.parse_args()
    if args.petri_workers < 1:
        raise ValueError("--petri-workers must be positive")
    if args.runtime_pilot_case_limit < 1:
        raise ValueError("--runtime-pilot-case-limit must be positive")

    study_dir = args.study_dir.resolve()
    for directory in ("logs", "state", "aggregation", "experiments"):
        (study_dir / directory).mkdir(parents=True, exist_ok=True)
    spec = build_spec(args)
    metadata = write_metadata(study_dir, spec)
    prepare_path = write_prepare_script(metadata, study_dir)
    queue_paths = write_queue_scripts(metadata, study_dir)
    aggregate_path = write_aggregate_script(study_dir)
    tmux_path, tmux_commands = write_tmux_commands(metadata, study_dir)
    instructions_path = write_instructions(metadata, study_dir, tmux_commands)

    print("Generated configuration only; no jobs or tmux sessions were launched.")
    print(f"Immutable metadata: {study_dir / 'study_metadata.json'}")
    print(f"Prepare: {prepare_path}")
    for path in queue_paths:
        print(f"Queue: {path}")
    print(f"Tmux commands: {tmux_path}")
    print(f"Aggregation: {aggregate_path}")
    print(f"Instructions: {instructions_path}")
    for command in tmux_commands:
        print(f"EXAMPLE (not run): {command}")


if __name__ == "__main__":
    main()
