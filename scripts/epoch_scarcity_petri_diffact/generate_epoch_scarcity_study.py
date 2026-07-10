#!/usr/bin/env python3
"""Generate immutable GTEA epoch-scarcity metadata and launch wrappers."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from epoch_scarcity_common import (
    CANONICAL_DECODER,
    DEFAULT_DATA_ROOT,
    DISCOVERY_FULL_TRAIN,
    DISCOVERY_OFFICIAL_TEST,
    ORACLE_CONDITION,
    PRIMARY_CONDITION,
    PROTOCOL_VERSION,
    STUDY_TYPE,
    WORKSPACE_ROOT,
    canonical_digest,
    dataset_spec,
    file_sha256,
    read_json,
    require_unique,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROVENANCE_FILES = (
    "src/calibration.py",
    "src/classes.py",
    "src/conformance_checking.py",
    "src/data_processing.py",
    "src/evaluation.py",
    "src/incremental_softmax_recovery.py",
    "src/petri_model.py",
    "src/utils.py",
    "scripts/epoch_scarcity_petri_diffact/epoch_scarcity_common.py",
    "scripts/epoch_scarcity_petri_diffact/prepare_epoch_scarcity.py",
    "scripts/epoch_scarcity_petri_diffact/import_d100_trajectory.py",
    "scripts/epoch_scarcity_petri_diffact/export_epoch_checkpoint.py",
    "scripts/epoch_scarcity_petri_diffact/run_epoch_petri_bounds.py",
    "scripts/epoch_scarcity_petri_diffact/run_epoch_scarcity_task.py",
    "scripts/epoch_scarcity_petri_diffact/generate_epoch_scarcity_study.py",
    "scripts/epoch_scarcity_petri_diffact/epoch_scarcity_status.py",
    "scripts/epoch_scarcity_petri_diffact/epoch_scarcity_aggregation.py",
    "scripts/low_data_petri_diffact/low_data_common.py",
    "scripts/low_data_petri_diffact/run_petri_postprocessing_low_data.py",
)


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def source_provenance(diffact_root: Path) -> Dict[str, Any]:
    hashes: Dict[str, str] = {}
    for relative in PROVENANCE_FILES:
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required epoch-scarcity source: {path}")
        hashes[relative] = file_sha256(path)
    for relative in (
        "main.py",
        "export_softmax.py",
        "dataset.py",
        "model.py",
        "default_configs.py",
        "utils.py",
    ):
        path = diffact_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing DiffAct source: {path}")
        hashes[f"baselines/DiffAct/{relative}"] = file_sha256(path)
    return {
        "source_digest": canonical_digest(hashes),
        "file_sha256": hashes,
        "workspace_git_head": git_value(WORKSPACE_ROOT, "rev-parse", "HEAD"),
        "workspace_git_branch": git_value(WORKSPACE_ROOT, "branch", "--show-current"),
        "diffact_git_head": git_value(diffact_root, "rev-parse", "HEAD"),
        "diffact_git_branch": git_value(diffact_root, "branch", "--show-current"),
        "diffact_git_status": git_value(diffact_root, "status", "--short").splitlines(),
    }


def render(command: Sequence[Any]) -> str:
    return shlex.join([str(value) for value in command])


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def source_d100_task(source_metadata: Dict[str, Any], fold: int, seed: int) -> Dict[str, Any]:
    matches = [
        task
        for task in source_metadata.get("tasks", [])
        if task.get("dataset") == "gtea"
        and int(task.get("official_fold", -1)) == fold
        and int(task.get("subset_seed", -1)) == seed
        and int(task.get("diffact_fraction", -1)) == 100
        and task.get("condition") == "honest_low_data_petri"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one source D100 task for fold={fold}, seed={seed}")
    return matches[0]


def task_record(
    *,
    study_dir: Path,
    task_id: str,
    task_type: str,
    fold: int,
    seed: int,
    gpu: int,
    command: List[str],
    requires_gpu: bool,
    depends_on: Sequence[str],
    checkpoint_epoch: int | None = None,
    condition: str | None = None,
    external_dependencies: Sequence[Dict[str, str]] = (),
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "dataset": "gtea",
        "official_fold": fold,
        "trajectory_seed": seed,
        "checkpoint_epoch_index": checkpoint_epoch,
        "condition": condition,
        "gpu": gpu,
        "queue": f"gpu{gpu}",
        "requires_gpu": requires_gpu,
        "depends_on_task_ids": list(depends_on),
        "external_dependencies": list(external_dependencies),
        "command": command,
        "state_path": str(study_dir / "state" / f"{task_id}.json"),
        "log_path": str(study_dir / "logs" / f"{task_id}.log"),
    }


def build_spec(args: argparse.Namespace) -> Dict[str, Any]:
    study_dir = args.study_dir.resolve()
    source_study_dir = args.source_study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    seed = int(args.seed)
    gpus = [int(value) for value in args.gpus]
    require_unique(gpus, "GPUs")
    if any(gpu < 0 for gpu in gpus):
        raise ValueError(f"Physical GPUs must be non-negative: {gpus}")
    folds = list(dataset_spec("gtea")["folds"])
    if len(gpus) > len(folds):
        raise ValueError("The GTEA first wave does not need more GPUs than folds")
    source_metadata = read_json(source_study_dir / "study_metadata.json")
    if not source_metadata.get("immutable"):
        raise ValueError("Source data-scarcity study must be immutable")
    if sorted({task.get("dataset") for task in source_metadata.get("tasks", [])}) != ["gtea"]:
        raise ValueError("The first-wave source study must be GTEA-only")

    experiments: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    spec = dataset_spec("gtea")
    for fold_index, fold in enumerate(folds):
        gpu = gpus[fold_index % len(gpus)]
        experiment_dir = study_dir / "experiments" / "gtea" / f"fold_{fold}"
        source_task = source_d100_task(source_metadata, fold, seed)
        experiments.append(
            {
                "dataset": "gtea",
                "official_fold": fold,
                "trajectory_seed": seed,
                "experiment_dir": str(experiment_dir),
                "source_d100_task_id": source_task["task_id"],
                "source_d100_state_path": source_task["state_path"],
            }
        )
        import_id = f"gtea_fold{fold}_seed{seed}_import_d100"
        import_command = [
            sys.executable,
            str(SCRIPT_DIR / "import_d100_trajectory.py"),
            "--experiment-dir",
            str(experiment_dir),
            "--execute",
        ]
        tasks.append(
            task_record(
                study_dir=study_dir,
                task_id=import_id,
                task_type="import_trajectory",
                fold=fold,
                seed=seed,
                gpu=gpu,
                command=import_command,
                requires_gpu=False,
                depends_on=[],
                external_dependencies=[
                    {
                        "dependency_id": source_task["task_id"],
                        "state_path": source_task["state_path"],
                    }
                ],
            )
        )
        export_ids: List[str] = []
        for epoch in spec["checkpoint_epochs"]:
            task_id = f"gtea_fold{fold}_seed{seed}_epoch{epoch}_export"
            export_ids.append(task_id)
            command = [
                sys.executable,
                str(SCRIPT_DIR / "export_epoch_checkpoint.py"),
                "--experiment-dir",
                str(experiment_dir),
                "--diffact-root",
                str(diffact_root),
                "--seed",
                str(seed),
                "--checkpoint-epoch",
                str(epoch),
                "--device",
                str(gpu),
                "--execute",
            ]
            tasks.append(
                task_record(
                    study_dir=study_dir,
                    task_id=task_id,
                    task_type="export_checkpoint",
                    fold=fold,
                    seed=seed,
                    gpu=gpu,
                    command=command,
                    requires_gpu=True,
                    depends_on=[import_id],
                    checkpoint_epoch=epoch,
                )
            )
        for condition, discovery_source in (
            (PRIMARY_CONDITION, DISCOVERY_FULL_TRAIN),
            (ORACLE_CONDITION, DISCOVERY_OFFICIAL_TEST),
        ):
            suffix = "fulltrain" if condition == PRIMARY_CONDITION else "oracle"
            task_id = f"gtea_fold{fold}_seed{seed}_decode_{suffix}_grid"
            command = [
                sys.executable,
                str(SCRIPT_DIR / "run_epoch_petri_bounds.py"),
                "--experiment-dir",
                str(experiment_dir),
                "--seed",
                str(seed),
                "--condition",
                condition,
                "--petri-discovery-source",
                discovery_source,
                "--workers",
                str(args.petri_workers),
                "--execute",
            ]
            tasks.append(
                task_record(
                    study_dir=study_dir,
                    task_id=task_id,
                    task_type="decode_checkpoint_grid",
                    fold=fold,
                    seed=seed,
                    gpu=gpu,
                    command=command,
                    requires_gpu=False,
                    depends_on=export_ids,
                    condition=condition,
                )
            )

    return {
        "study_id": args.study_id,
        "study_type": STUDY_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "workspace_root": str(WORKSPACE_ROOT),
        "study_dir": str(study_dir),
        "source_study_dir": str(source_study_dir),
        "source_study_spec_sha256": source_metadata.get("spec_sha256"),
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "trajectory_seed": seed,
        "datasets": [{"dataset": "gtea", "official_folds": folds}],
        "checkpoint_grid": {"gtea": list(spec["checkpoint_epochs"])},
        "epoch_indexing": "zero_based",
        "schedule_fraction_definition": "100 * epoch_index / final_epoch_index",
        "source_trajectory": "immutable data-scarcity D100 checkpoint trajectory",
        "trajectory_reuse_disclosure": (
            "Phase 2 reuses the same seed-0 D100 trajectory that supplies the Phase 1 D100 "
            "endpoint. It is a learning-trajectory analysis, not an independent retraining "
            "replicate, and must be reported as such."
        ),
        "conditions": {
            PRIMARY_CONDITION: {
                "role": "primary practical condition",
                "petri_discovery_source": DISCOVERY_FULL_TRAIN,
                "is_oracle_condition": False,
            },
            ORACLE_CONDITION: {
                "role": "leaky theoretic ceiling",
                "petri_discovery_source": DISCOVERY_OFFICIAL_TEST,
                "is_oracle_condition": True,
            },
        },
        "decoder": CANONICAL_DECODER,
        "primary_endpoint": {
            "method": "sktr",
            "comparator": "official_diffact",
            "metric": "f1@25",
            "condition": PRIMARY_CONDITION,
            "paired_unit": "test_case",
        },
        "secondary_fold_global_endpoint": {
            "method": "sktr",
            "comparator": "official_diffact",
            "metric": "f1@25",
            "condition": PRIMARY_CONDITION,
            "paired_unit": "official_fold_global_tas",
            "role": "published-style absolute metric comparability; not the primary endpoint",
        },
        "export_integrity_policy": (
            "Every expected raw, canonical, official, mapping, and ground-truth artifact is "
            "SHA-256 recorded after export and revalidated before skip or Petri decode."
        ),
        "launch_policy": (
            "Do not launch concurrently with Phase 1. Wait for the complete GTEA "
            "data-scarcity study, even though task dependencies only require each D100 source."
        ),
        "gpu_policy": {
            "physical_gpu_ids": gpus,
            "one_serial_queue_per_gpu": True,
            "exclusive_access_assumed": False,
            "shared_with_other_users": True,
        },
        "petri_workers": int(args.petri_workers),
        "source_provenance": source_provenance(diffact_root),
        "experiments": experiments,
        "tasks": tasks,
    }


def write_metadata(study_dir: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    path = study_dir / "study_metadata.json"
    digest = canonical_digest(spec)
    if path.is_file():
        existing = read_json(path)
        if existing.get("spec_sha256") != digest:
            raise FileExistsError(f"Refusing to mutate immutable metadata: {path}")
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


def write_scripts(metadata: Dict[str, Any], study_dir: Path) -> None:
    prepare_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for experiment in metadata["experiments"]:
        prepare_lines.append(
            render(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "prepare_epoch_scarcity.py"),
                    "--source-study-dir",
                    metadata["source_study_dir"],
                    "--experiment-dir",
                    experiment["experiment_dir"],
                    "--data-root",
                    metadata["data_root"],
                    "--diffact-root",
                    metadata["diffact_root"],
                    "--dataset",
                    "gtea",
                    "--fold",
                    str(experiment["official_fold"]),
                    "--seed",
                    str(metadata["trajectory_seed"]),
                ]
            )
        )
    write_executable(study_dir / "prepare_study.sh", "\n".join(prepare_lines) + "\n")

    for gpu in metadata["gpu_policy"]["physical_gpu_ids"]:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for task in metadata["tasks"]:
            if int(task["gpu"]) != int(gpu):
                continue
            lines.append(
                render(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "run_epoch_scarcity_task.py"),
                        "--study-dir",
                        str(study_dir),
                        "--task-id",
                        task["task_id"],
                    ]
                )
            )
        write_executable(study_dir / f"queue_gpu{gpu}.sh", "\n".join(lines) + "\n")

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", metadata["study_id"])
    tmux_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for gpu in metadata["gpu_policy"]["physical_gpu_ids"]:
        session_name = f"{slug}_gpu{gpu}"
        queue_command = f"bash {study_dir / f'queue_gpu{gpu}.sh'}"
        tmux_lines.append(
            f"tmux new-session -d -s {shlex.quote(session_name)} "
            f"{shlex.quote(queue_command)}"
        )
    write_executable(study_dir / "tmux_commands.sh", "\n".join(tmux_lines) + "\n")
    write_executable(
        study_dir / "aggregate_study.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + render(
            [
                sys.executable,
                str(SCRIPT_DIR / "epoch_scarcity_aggregation.py"),
                "--study-dir",
                str(study_dir),
            ]
        )
        + "\n",
    )
    write_executable(
        study_dir / "study_status.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + render(
            [
                sys.executable,
                str(SCRIPT_DIR / "epoch_scarcity_status.py"),
                "--study-dir",
                str(study_dir),
            ]
        )
        + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--study-id", default="epoch_scarcity_gtea_seed0")
    parser.add_argument("--source-study-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--diffact-root",
        type=Path,
        required=True,
        help=(
            "Existing DiffAct checkout to hash and use for checkpoint export. "
            "This is required because the isolated epoch-scarcity worktree does not "
            "duplicate the independently cloned DiffAct repository."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--petri-workers", type=int, default=7)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("The first epoch-scarcity wave is pre-registered for seed 0 only")
    if args.petri_workers < 1:
        raise ValueError("--petri-workers must be positive")
    study_dir = args.study_dir.resolve()
    for name in ("logs", "state", "aggregation", "experiments"):
        (study_dir / name).mkdir(parents=True, exist_ok=True)
    metadata = write_metadata(study_dir, build_spec(args))
    write_scripts(metadata, study_dir)
    print("Generated configuration only; no import, export, decode, or tmux job was launched.")
    print(f"Immutable metadata: {study_dir / 'study_metadata.json'}")
    print(f"Prepare: {study_dir / 'prepare_study.sh'}")
    print(f"Tmux commands: {study_dir / 'tmux_commands.sh'}")


if __name__ == "__main__":
    main()
