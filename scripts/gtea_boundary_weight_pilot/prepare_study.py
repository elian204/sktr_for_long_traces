#!/usr/bin/env python3
"""Stage the immutable GTEA multi-fold boundary-weight confirmation study."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    BASELINE_BOUNDARY_WEIGHT,
    BOUNDARY_MATCH_MAX_DISTANCE,
    BOUNDARY_WEIGHTS,
    CHECKPOINT_EPOCHS,
    DATASET,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_FOLD1_STUDY_DIR,
    DEFAULT_LOCKED_MANIFEST_ROOT,
    DEFAULT_LOCKED_REFERENCE_CONFIG,
    DEFAULT_STUDY_DIR,
    EXPECTED_NUM_EPOCHS,
    EXPECTED_TEST_CASES,
    EXPECTED_TRAIN_CASES,
    FINAL_EPOCH,
    FOLDS,
    GPU_IDS,
    INFERENCE_SEEDS,
    INFERENCE_SEED_STRIDE,
    NEW_TRAINING_FOLDS,
    PRIMARY_BOUNDARY_WEIGHT,
    PROTOCOL_VERSION,
    SCRIPT_DIR,
    SHORT_SEGMENT_MAX_LENGTH,
    TRAINING_SEEDS,
    atomic_write_json,
    build_variant_config,
    canonical_digest,
    checkpoint_path,
    create_alignment_dir,
    create_dataset_view,
    export_dir,
    file_sha256,
    load_json,
    read_bundle,
    source_provenance,
    training_invariant_digest,
    training_invariant_payload,
    variant_id,
    verify_export,
    write_bundle,
    write_lines,
)


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def shell_command(parts: Sequence[Any]) -> str:
    return shlex.join([str(part) for part in parts])


def validate_reference_config(config: Mapping[str, Any]) -> None:
    expectations = {
        "dataset_name": DATASET,
        "boundary_smooth": 1,
        "soft_label": 1.4,
        "num_epochs": EXPECTED_NUM_EPOCHS,
        "log_freq": 100,
        "sample_rate": 1,
        "random_seed": 0,
        "initialization_seed": 0,
        "evaluate_during_training": False,
        "log_train_results": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expectations.items()
        if config.get(key) != expected
    }
    if config.get("postprocess") != {"type": "purge", "value": 3}:
        mismatches["postprocess"] = {
            "expected": {"type": "purge", "value": 3},
            "actual": config.get("postprocess"),
        }
    if float(config["loss_weights"]["decoder_boundary_loss"]) != BASELINE_BOUNDARY_WEIGHT:
        mismatches["loss_weights.decoder_boundary_loss"] = {
            "expected": BASELINE_BOUNDARY_WEIGHT,
            "actual": config["loss_weights"]["decoder_boundary_loss"],
        }
    if float(config["loss_weights"]["encoder_boundary_loss"]) != 0.0:
        mismatches["loss_weights.encoder_boundary_loss"] = {
            "expected": 0.0,
            "actual": config["loss_weights"]["encoder_boundary_loss"],
        }
    expected_conditions = ["full", "zero", "boundary03-", "segment=1", "segment=1"]
    if config["diffusion_params"]["cond_types"] != expected_conditions:
        mismatches["diffusion_params.cond_types"] = {
            "expected": expected_conditions,
            "actual": config["diffusion_params"]["cond_types"],
        }
    if any(epoch % int(config["log_freq"]) for epoch in CHECKPOINT_EPOCHS):
        mismatches["checkpoint_grid"] = {
            "expected": list(CHECKPOINT_EPOCHS),
            "actual_log_freq": config["log_freq"],
        }
    if mismatches:
        raise ValueError(f"Locked D100 reference config violates assumptions: {mismatches}")


def locked_manifest_path(root: Path, fold: int) -> Path:
    return root / f"fold_{fold}" / "manifests" / "seed_0" / "train_cases_frac_100.txt"


def locked_bundle_path(root: Path, fold: int) -> Path:
    return (
        root
        / f"fold_{fold}"
        / "diffact_dataset_views"
        / "seed_0"
        / "frac_100"
        / DATASET
        / "splits"
        / "train.split1.bundle"
    )


def source_completion_path(source_task: Mapping[str, Any]) -> Path:
    if source_task["execution_mode"] == "imported":
        return Path(source_task["import_manifest_path"])
    return Path(source_task["run_dir"]) / "task_complete.json"


def record_fold1_import(
    *,
    study_dir: Path,
    task: Mapping[str, Any],
    source_task: Mapping[str, Any],
    source_study_dir: Path,
    expected_cases: Sequence[str],
    expected_invariant_digest: str,
) -> Dict[str, Any]:
    source_config = Path(source_task["config_path"])
    completion_path = source_completion_path(source_task)
    if not source_config.is_file() or not completion_path.is_file():
        raise FileNotFoundError(
            f"Fold-1 import source is incomplete: {source_config}, {completion_path}"
        )
    source_config_payload = load_json(source_config)
    if training_invariant_digest(source_config_payload) != expected_invariant_digest:
        raise ValueError(f"Fold-1 source violates the frozen training protocol: {source_config}")
    checks = {
        "fold": int(source_task["official_fold"]) == 1,
        "training_seed": int(source_task["training_seed"]) == int(task["training_seed"]),
        "decoder_boundary_loss": float(source_task["decoder_boundary_loss"])
        == float(task["decoder_boundary_loss"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Fold-1 import identity mismatch for {task['task_id']}: {checks}")

    source_completion = load_json(completion_path)
    checkpoint_hashes: Dict[str, str] = {}
    exports: List[Dict[str, Any]] = []
    for epoch in task["checkpoint_epochs"]:
        checkpoint = checkpoint_path(task, int(epoch))
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[str(epoch)] = file_sha256(checkpoint)
        for inference_seed in task["inference_seeds"]:
            output_dir = export_dir(task, int(epoch), int(inference_seed))
            summary = verify_export(output_dir, expected_cases)
            exports.append(
                {
                    "checkpoint_epoch": int(epoch),
                    "inference_seed": int(inference_seed),
                    "output_dir": str(output_dir),
                    "export_complete_sha256": file_sha256(
                        output_dir / "export_complete.json"
                    ),
                    "artifact_digest": canonical_digest(summary["artifact_sha256"]),
                    "case_count": summary["case_count"],
                    "frame_count": summary["frame_count"],
                }
            )
    if source_completion.get("checkpoint_sha256") != checkpoint_hashes:
        raise ValueError(
            f"Fold-1 checkpoint hashes differ from source completion: {task['task_id']}"
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "imported_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"],
        "official_fold": 1,
        "training_seed": task["training_seed"],
        "decoder_boundary_loss": task["decoder_boundary_loss"],
        "source_study_dir": str(source_study_dir),
        "source_study_metadata_sha256": file_sha256(
            source_study_dir / "study_metadata.json"
        ),
        "source_tasks_sha256": file_sha256(source_study_dir / "tasks.json"),
        "source_task_id": source_task["task_id"],
        "source_execution_mode": source_task["execution_mode"],
        "source_run_dir": source_task["artifact_run_dir"],
        "source_config_sha256": file_sha256(source_config),
        "source_completion_path": str(completion_path),
        "source_completion_sha256": file_sha256(completion_path),
        "checkpoint_sha256": checkpoint_hashes,
        "exports": exports,
        "hash_verification": "all checkpoints and all exported inference artifacts",
    }
    atomic_write_json(Path(task["import_manifest_path"]), payload)
    return payload


def task_gpu(fold: int, training_seed: int, weight: float) -> int:
    """Latin-rotate the four seed/weight cells across GPUs by fold."""
    combo = TRAINING_SEEDS.index(training_seed) * len(BOUNDARY_WEIGHTS)
    combo += BOUNDARY_WEIGHTS.index(weight)
    return GPU_IDS[(combo + fold - min(NEW_TRAINING_FOLDS)) % len(GPU_IDS)]


def build_study(args: argparse.Namespace) -> None:
    study_dir = args.study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    fold1_study_dir = args.fold1_study_dir.resolve()
    locked_config_path = args.locked_reference_config.resolve()
    locked_manifest_root = args.locked_manifest_root.resolve()
    if study_dir.exists():
        raise FileExistsError(
            f"Study directory already exists: {study_dir}. Choose a new immutable version."
        )
    for path in (
        locked_config_path,
        fold1_study_dir / "study_metadata.json",
        fold1_study_dir / "tasks.json",
        fold1_study_dir / "analysis" / "analysis_complete.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_config = load_json(locked_config_path)
    validate_reference_config(reference_config)
    fold1_metadata = load_json(fold1_study_dir / "study_metadata.json")
    if fold1_metadata.get("protocol_version") != "gtea-boundary-weight-replication-v2":
        raise ValueError("Fold-1 import source has an unexpected protocol")
    if not load_json(fold1_study_dir / "analysis" / "analysis_complete.json").get(
        "completed"
    ):
        raise ValueError("Fold-1 source analysis is not complete")
    fold1_source_tasks = load_json(fold1_study_dir / "tasks.json")["tasks"]
    if len(fold1_source_tasks) != 9:
        raise ValueError(f"Expected all 9 fold-1 v2 tasks; found {len(fold1_source_tasks)}")

    study_dir.mkdir(parents=True)
    (study_dir / "logs").mkdir()
    fold_assets: Dict[int, Dict[str, Any]] = {}
    for fold in FOLDS:
        train_split = data_root / DATASET / "splits" / f"train.split{fold}.bundle"
        test_split = data_root / DATASET / "splits" / f"test.split{fold}.bundle"
        locked_manifest = locked_manifest_path(locked_manifest_root, fold)
        locked_bundle = locked_bundle_path(locked_manifest_root, fold)
        for path in (train_split, test_split, locked_manifest, locked_bundle):
            if not path.is_file():
                raise FileNotFoundError(path)
        official_train_cases = read_bundle(train_split)
        train_cases = read_bundle(locked_manifest)
        locked_bundle_cases = read_bundle(locked_bundle)
        test_cases = read_bundle(test_split)
        if len(train_cases) != EXPECTED_TRAIN_CASES or len(test_cases) != EXPECTED_TEST_CASES:
            raise ValueError(
                f"Unexpected fold-{fold} counts: train={len(train_cases)}, "
                f"test={len(test_cases)}"
            )
        if train_cases != locked_bundle_cases:
            raise ValueError(
                f"Fold-{fold} locked frac-100 manifest and actual train bundle order differ"
            )
        if set(train_cases) != set(official_train_cases):
            raise ValueError(
                f"Fold-{fold} locked train cases are not set-equal to official train"
            )
        if set(train_cases).intersection(test_cases):
            raise ValueError(f"Fold-{fold} official train/test overlap detected")

        manifest_dir = study_dir / "manifests" / f"fold_{fold}"
        train_manifest = manifest_dir / "train_cases_frac_100.txt"
        test_manifest = manifest_dir / "official_test_cases.txt"
        manifest_dir.mkdir(parents=True)
        train_manifest.write_bytes(locked_manifest.read_bytes())
        write_bundle(test_manifest, test_cases)
        view_root = create_dataset_view(
            study_dir,
            data_root,
            fold,
            train_cases,
            test_cases,
            locked_train_bundle=locked_bundle,
        )
        generated_bundle = (
            view_root / DATASET / "splits" / f"train.split{fold}.bundle"
        )
        if generated_bundle.read_bytes() != locked_bundle.read_bytes():
            raise ValueError(f"Fold-{fold} generated bundle is not byte-aligned")
        align_dir = create_alignment_dir(study_dir, data_root, fold, test_cases)
        fold_assets[fold] = {
            "train_split": train_split,
            "test_split": test_split,
            "locked_manifest": locked_manifest,
            "locked_bundle": locked_bundle,
            "train_manifest": train_manifest,
            "test_manifest": test_manifest,
            "view_root": view_root,
            "generated_bundle": generated_bundle,
            "align_dir": align_dir,
            "train_cases": train_cases,
            "test_cases": test_cases,
        }

    source_task_by_id = {
        str(task["task_id"]): task for task in fold1_source_tasks
    }
    specs: List[Dict[str, Any]] = []
    for source_task in fold1_source_tasks:
        specs.append(
            {
                "fold": 1,
                "training_seed": int(source_task["training_seed"]),
                "weight": float(source_task["decoder_boundary_loss"]),
                "execution_mode": "imported",
                "source_task": source_task,
                "physical_gpu": int(source_task["physical_gpu"]),
            }
        )
    for fold in NEW_TRAINING_FOLDS:
        for training_seed in TRAINING_SEEDS:
            for weight in BOUNDARY_WEIGHTS:
                specs.append(
                    {
                        "fold": fold,
                        "training_seed": training_seed,
                        "weight": weight,
                        "execution_mode": "train",
                        "source_task": None,
                        "physical_gpu": task_gpu(fold, training_seed, weight),
                    }
                )

    variant_configs: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        fold = int(spec["fold"])
        seed = int(spec["training_seed"])
        weight = float(spec["weight"])
        assets = fold_assets[fold]
        task_id = variant_id(fold, weight, seed)
        variant_configs[task_id] = build_variant_config(
            reference_config,
            study_dir=study_dir,
            view_root=assets["view_root"],
            train_manifest=assets["train_manifest"],
            test_manifest=assets["test_manifest"],
            fold=fold,
            weight=weight,
            training_seed=seed,
        )
    invariant_digests = {
        task_id: training_invariant_digest(config)
        for task_id, config in variant_configs.items()
    }
    if len(set(invariant_digests.values())) != 1:
        raise ValueError(f"Tasks differ beyond fold/weight/seed/path: {invariant_digests}")
    invariant_digest = next(iter(invariant_digests.values()))

    tasks: List[Dict[str, Any]] = []
    for spec in specs:
        fold = int(spec["fold"])
        seed = int(spec["training_seed"])
        weight = float(spec["weight"])
        imported = spec["execution_mode"] == "imported"
        task_id = variant_id(fold, weight, seed)
        run_dir = study_dir / "runs" / f"fold_{fold}" / task_id
        config_path = run_dir / "config.json"
        atomic_write_json(config_path, variant_configs[task_id])
        source_task = spec["source_task"]
        artifact_run_dir = (
            Path(source_task["artifact_run_dir"]) if imported else run_dir
        )
        artifact_model_dir = (
            Path(source_task["artifact_model_dir"])
            if imported
            else run_dir / "training" / task_id
        )
        assets = fold_assets[fold]
        included_in_primary = (
            weight in BOUNDARY_WEIGHTS and seed in TRAINING_SEEDS
        )
        task: Dict[str, Any] = {
            "task_id": task_id,
            "task_type": (
                "hash_verified_fold1_v2_import"
                if imported
                else "train_and_export_multifold_variant"
            ),
            "execution_mode": spec["execution_mode"],
            "dataset": DATASET,
            "official_fold": fold,
            "training_seed": seed,
            "decoder_boundary_loss": weight,
            "included_in_primary_grid": included_in_primary,
            "role": (
                "paired_baseline"
                if weight == BASELINE_BOUNDARY_WEIGHT
                else "pre_registered_primary"
                if weight == PRIMARY_BOUNDARY_WEIGHT
                else "fold1_exploratory_import"
            ),
            "physical_gpu": int(spec["physical_gpu"]),
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "model_dir": str(run_dir / "training" / task_id),
            "artifact_run_dir": str(artifact_run_dir),
            "artifact_model_dir": str(artifact_model_dir),
            "final_checkpoint": str(
                artifact_model_dir / f"epoch-{FINAL_EPOCH}.model"
            ),
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "inference_seeds": list(INFERENCE_SEEDS),
            "train_manifest": str(assets["train_manifest"]),
            "test_manifest": str(assets["test_manifest"]),
            "align_dir": str(assets["align_dir"]),
            "training_invariant_digest": invariant_digest,
            "state_path": str(study_dir / "state" / f"{task_id}.json"),
            "log_path": str(study_dir / "logs" / f"{task_id}.log"),
            "import_manifest_path": (
                str(study_dir / "imports" / task_id / "import_complete.json")
                if imported
                else None
            ),
        }
        task["expected_checkpoints"] = {
            str(epoch): str(checkpoint_path(task, epoch)) for epoch in CHECKPOINT_EPOCHS
        }
        tasks.append(task)

    imports: List[Dict[str, Any]] = []
    for task in tasks:
        if task["execution_mode"] != "imported":
            continue
        source_task = source_task_by_id[task["task_id"]]
        imports.append(
            record_fold1_import(
                study_dir=study_dir,
                task=task,
                source_task=source_task,
                source_study_dir=fold1_study_dir,
                expected_cases=fold_assets[1]["test_cases"],
                expected_invariant_digest=invariant_digest,
            )
        )

    training_tasks = [task for task in tasks if task["execution_mode"] == "train"]
    queue_assignments = {
        str(gpu): [
            task["task_id"]
            for task in training_tasks
            if int(task["physical_gpu"]) == gpu
        ]
        for gpu in GPU_IDS
    }
    if any(len(task_ids) != 3 for task_ids in queue_assignments.values()):
        raise ValueError(f"Expected exactly 3 trainings per GPU: {queue_assignments}")

    provenance = source_provenance(diffact_root)
    bundle_contracts: Dict[str, Any] = {}
    for fold, assets in fold_assets.items():
        bundle_contracts[str(fold)] = {
            "locked_manifest": str(assets["locked_manifest"]),
            "locked_manifest_sha256": file_sha256(assets["locked_manifest"]),
            "locked_actual_diffact_bundle": str(assets["locked_bundle"]),
            "locked_actual_diffact_bundle_sha256": file_sha256(
                assets["locked_bundle"]
            ),
            "generated_manifest": str(assets["train_manifest"]),
            "generated_manifest_sha256": file_sha256(assets["train_manifest"]),
            "generated_diffact_bundle": str(assets["generated_bundle"]),
            "generated_diffact_bundle_sha256": file_sha256(
                assets["generated_bundle"]
            ),
            "official_train_split": str(assets["train_split"]),
            "official_train_split_sha256": file_sha256(assets["train_split"]),
            "official_test_split": str(assets["test_split"]),
            "official_test_split_sha256": file_sha256(assets["test_split"]),
            "case_order": assets["train_cases"],
            "manifest_order_equals_locked_bundle": True,
            "generated_bundle_byte_identical_to_locked_bundle": True,
            "set_equals_official_train": True,
            "official_split_order_intentionally_not_used": True,
        }
    metadata: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "immutable": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "workspace_root": str(SCRIPT_DIR.parents[1]),
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "dataset": DATASET,
        "official_folds": list(FOLDS),
        "new_training_folds": list(NEW_TRAINING_FOLDS),
        "training_seeds": list(TRAINING_SEEDS),
        "gpu_policy": {
            "physical_gpu_ids": list(GPU_IDS),
            "four_independent_queues": True,
            "trainings_per_queue": 3,
            "wait_for_zero_compute_processes": True,
            "required_consecutive_free_checks": 2,
            "free_check_interval_seconds": 60,
            "fail_closed_on_nvidia_smi_error": True,
            "automatic_fallback": False,
            "never_preempt_selector_study": True,
        },
        "queue_assignments": queue_assignments,
        "boundary_weights": list(BOUNDARY_WEIGHTS),
        "baseline_boundary_weight": BASELINE_BOUNDARY_WEIGHT,
        "pre_registered_primary_boundary_weight": PRIMARY_BOUNDARY_WEIGHT,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "inference_seeds": list(INFERENCE_SEEDS),
        "inference_seed_formula": (
            f"model_seed = inference_seed * {INFERENCE_SEED_STRIDE} + video_index"
        ),
        "new_training_count": len(training_tasks),
        "import_count": len(imports),
        "fold1_import_study": str(fold1_study_dir),
        "fold1_import_study_metadata_sha256": file_sha256(
            fold1_study_dir / "study_metadata.json"
        ),
        "fold1_import_tasks_sha256": file_sha256(fold1_study_dir / "tasks.json"),
        "fold1_import_analysis_complete_sha256": file_sha256(
            fold1_study_dir / "analysis" / "analysis_complete.json"
        ),
        "import_contract": {
            "all_fold1_v2_tasks_imported": True,
            "source_task_count": len(fold1_source_tasks),
            "verification": "all checkpoint and export artifact hashes",
            "retrain_imported_variants": False,
        },
        "per_fold_training_order_contract": bundle_contracts,
        "locked_reference_config": str(locked_config_path),
        "locked_reference_config_sha256": file_sha256(locked_config_path),
        "training_invariant_payload": training_invariant_payload(
            variant_configs[variant_id(2, BASELINE_BOUNDARY_WEIGHT, 0)]
        ),
        "training_invariant_digest": invariant_digest,
        "experimental_fields": [
            "official_fold",
            "loss_weights.decoder_boundary_loss",
            "random_seed",
            "initialization_seed",
        ],
        "frozen_fields_statement": (
            "All runtime configs share one training-invariant digest. Only official "
            "fold/data paths, boundary weight, declared training seed, and output "
            "naming/path fields may differ."
        ),
        "checkpoint_selection": "pre_specified_grid_no_test_selection",
        "baseline_reconciliation_gate": None,
        "reconciliation_policy": (
            "No bitwise reconciliation gate. Fold-1 v2 established benign run-to-run "
            "nondeterminism; this study reports cross-seed noise per fold and weight."
        ),
        "diagnostic_contract": {
            "streams": {
                "pre_purge": "argmax of raw decoder probability matrix",
                "post_purge": "official DiffAct purge-3 discrete prediction",
            },
            "boundary_f1_tolerances_frames": [5, 10],
            "boundary_match_max_distance_frames": BOUNDARY_MATCH_MAX_DISTANCE,
            "boundary_matching": ["class_agnostic", "transition_aware"],
            "short_false_segment_max_length": SHORT_SEGMENT_MAX_LENGTH,
            "standard_metrics": ["acc", "edit", "f1@10", "f1@25", "f1@50"],
            "cross_seed_noise": [
                "metric_delta_seed1_minus_seed0",
                "frame_disagreement_fraction",
            ],
        },
        "decision_rule": {
            "variant": PRIMARY_BOUNDARY_WEIGHT,
            "baseline": BASELINE_BOUNDARY_WEIGHT,
            "epoch": FINAL_EPOCH,
            "stream": "post_purge",
            "per_fold_aggregation": (
                "mean_over_2_training_seeds_x_3_inference_seeds"
            ),
            "pooled_aggregation": (
                "equal_weight_mean_over_4_folds_x_2_training_seeds_x_3_inference_seeds"
            ),
            "six_requirements": {
                "delta_boundary_f1_class_agnostic@10": ">0",
                "delta_edit": ">0",
                "delta_f1@25": ">0",
                "delta_acc": ">=0",
                "delta_segment_count_ratio": "<=0.02",
                "delta_boundary_offset_class_agnostic_mean_absolute": "<0",
            },
            "primary_claim_additional_requirements": {
                "pooled_six_requirement_gate": "all pass",
                "positive_delta_edit_folds": ">=3 of 4",
                "positive_delta_f1@25_folds": ">=3 of 4",
            },
            "offset_statistic": "mean absolute boundary offset (v2 rule)",
            "on_pass": "Task 2 onset-head launch may be considered after review",
            "on_fail": "Do not launch Task 2 without a new explicit scientific decision",
        },
        "source_provenance": provenance,
        "tasks": [task["task_id"] for task in tasks],
        "training_tasks": [task["task_id"] for task in training_tasks],
        "imported_tasks": [
            task["task_id"] for task in tasks if task["execution_mode"] == "imported"
        ],
    }
    metadata["spec_sha256"] = canonical_digest(
        {key: value for key, value in metadata.items() if key != "spec_sha256"}
    )
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    atomic_write_json(study_dir / "tasks.json", {"tasks": tasks})

    python = sys.executable
    sessions: Dict[int, str] = {}
    for gpu in GPU_IDS:
        queue_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"export CUDA_VISIBLE_DEVICES={gpu}",
        ]
        for task_id in queue_assignments[str(gpu)]:
            queue_lines.append(
                shell_command(
                    [
                        python,
                        "-u",
                        SCRIPT_DIR / "run_variant.py",
                        "--study-dir",
                        study_dir,
                        "--task-id",
                        task_id,
                    ]
                )
            )
        queue_lines.append(
            f"touch {shlex.quote(str(study_dir / 'state' / f'gpu_{gpu}_queue_complete'))}"
        )
        queue_path = study_dir / "queues" / f"gpu_{gpu}.sh"
        write_executable(queue_path, "\n".join(queue_lines) + "\n")

        waiter_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"gpu={gpu}",
            "free_checks=0",
            "interval=${FREE_CHECK_INTERVAL_SECONDS:-60}",
            'if ! [[ "$interval" =~ ^[0-9]+$ ]] || (( interval < 1 )); then',
            '  echo "FREE_CHECK_INTERVAL_SECONDS must be a positive integer" >&2',
            "  exit 2",
            "fi",
            "while (( free_checks < 2 )); do",
            "  if ! busy=$(nvidia-smi -i \"$gpu\" "
            "--query-compute-apps=pid,used_memory,process_name "
            "--format=csv,noheader,nounits 2>&1); then",
            '    echo "nvidia-smi failed for GPU $gpu; refusing to infer availability" >&2',
            '    echo "$busy" >&2',
            "    exit 3",
            "  fi",
            "  busy=$(printf '%s\\n' \"$busy\" | sed '/^[[:space:]]*$/d')",
            '  if [[ -z "$busy" ]]; then',
            "    ((free_checks+=1))",
            '    echo "GPU $gpu free check $free_checks/2"',
            "  else",
            "    free_checks=0",
            '    echo "GPU $gpu busy; selector/other work is left untouched:"',
            '    echo "$busy"',
            "  fi",
            "  if (( free_checks < 2 )); then sleep \"$interval\"; fi",
            "done",
            f"exec {shlex.quote(str(queue_path))}",
        ]
        write_executable(
            study_dir / "waiters" / f"gpu_{gpu}_wait_then_run.sh",
            "\n".join(waiter_lines) + "\n",
        )
        sessions[gpu] = f"gtea_bweight_multifold_g{gpu}"

    launch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(study_dir / 'logs'))}",
    ]
    for gpu, session in sessions.items():
        launch_lines.extend(
            [
                f"if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then",
                f"  echo {shlex.quote('Session already exists: ' + session)} >&2",
                "  exit 4",
                "fi",
            ]
        )
    for gpu, session in sessions.items():
        waiter = study_dir / "waiters" / f"gpu_{gpu}_wait_then_run.sh"
        tmux_log = study_dir / "logs" / f"gpu_{gpu}_queue.tmux.log"
        tmux_command = (
            f"{shlex.quote(str(waiter))} 2>&1 | tee -a {shlex.quote(str(tmux_log))}"
        )
        launch_lines.append(
            f"tmux new-session -d -s {shlex.quote(session)} "
            f"{shlex.quote(tmux_command)}"
        )
        launch_lines.append(
            f"echo {shlex.quote(f'Armed {session}: GPU {gpu}, two free checks required')}"
        )
    write_executable(study_dir / "launch_tmux.sh", "\n".join(launch_lines) + "\n")
    write_executable(
        study_dir / "status.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shell_command([python, SCRIPT_DIR / "study_status.py", "--study-dir", study_dir])
        + "\n",
    )
    write_executable(
        study_dir / "analyze.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shell_command([python, "-u", SCRIPT_DIR / "analyze.py", "--study-dir", study_dir])
        + "\n",
    )
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable generated GTEA multi-fold boundary-weight confirmation.",
            f"Protocol: {PROTOCOL_VERSION}",
            "Do not edit configs, manifests, tasks, imports, queues, or source after generation.",
            "Generate a new versioned study for any change.",
            "prepare_study.py staged the study and launched nothing.",
        ],
    )
    print(f"Prepared immutable review study: {study_dir}")
    print(f"Fold-1 imports: {len(imports)}; net new trainings: {len(training_tasks)}")
    print(f"Queue assignments: {queue_assignments}")
    print(f"Training invariant digest: {invariant_digest}")
    print(f"Source digest: {provenance['source_digest']}")
    print("Nothing was launched. Fable spot-check is required before launch_tmux.sh.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument(
        "--fold1-study-dir", type=Path, default=DEFAULT_FOLD1_STUDY_DIR
    )
    parser.add_argument(
        "--locked-reference-config",
        type=Path,
        default=DEFAULT_LOCKED_REFERENCE_CONFIG,
    )
    parser.add_argument(
        "--locked-manifest-root",
        type=Path,
        default=DEFAULT_LOCKED_MANIFEST_ROOT,
    )
    build_study(parser.parse_args())


if __name__ == "__main__":
    main()
