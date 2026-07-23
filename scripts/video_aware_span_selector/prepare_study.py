#!/usr/bin/env python3
"""Generate the immutable, non-launched four-fold Breakfast selector study."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    BASE_NUMERIC_FEATURES,
    DATASET,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_OUTER_EXPORT_ROOT,
    DEFAULT_STUDY_DIR,
    DEFAULT_V1_STUDY_DIR,
    FINAL_EPOCH,
    FORBIDDEN_PRIMARY_FEATURE_PREFIXES,
    FRAME_BUDGETS,
    INNER_FOLDS,
    OUTER_FOLDS,
    PROTOCOL_VERSION,
    REPAIR_CANDIDATE_FIELDS,
    REPAIR_CANDIDATE_TOP_K,
    SCRIPT_DIR,
    SEED,
    SHAPE_FEATURES,
    atomic_write_json,
    canonical_digest,
    create_alignment_dir,
    create_dataset_view,
    file_sha256,
    fold_summary,
    load_case_infos,
    load_json,
    make_subject_disjoint_inner_folds,
    official_splits,
    read_bundle,
    repair_candidate_columns,
    source_provenance,
    write_bundle,
    write_lines,
)


GPU_ASSIGNMENT = (0, 1, 2)
FREE_CHECK_INTERVAL_SECONDS = 60
REQUIRED_CONSECUTIVE_FREE_CHECKS = 2


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def shell_command(parts: Sequence[Any]) -> str:
    return shlex.join([str(part) for part in parts])


def training_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "dataset_name",
        "batch_size",
        "boundary_smooth",
        "class_weighting",
        "decoder_params",
        "diffusion_params",
        "encoder_params",
        "learning_rate",
        "log_freq",
        "num_epochs",
        "postprocess",
        "sample_rate",
        "set_sampling_seed",
        "soft_label",
        "temporal_aug",
        "weight_decay",
        "loss_weights",
        "random_seed",
        "initialization_seed",
        "evaluate_during_training",
        "log_train_results",
    )
    return {key: config[key] for key in keys}


def export_rows(export_dir: Path) -> Dict[str, int]:
    rows: Dict[str, int] = {}
    for raw in (export_dir / "video_index_map.txt").read_text().splitlines():
        if not raw.strip():
            continue
        fields = raw.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed export row in {export_dir}: {raw!r}")
        index, case = int(fields[0]), fields[1].removesuffix(".txt")
        if case in rows:
            raise ValueError(f"Duplicate export case {case}: {export_dir}")
        rows[case] = index
    return rows


def validate_outer_export(
    export_dir: Path, outer_test: Sequence[str], outer_fold: int
) -> Dict[str, Any]:
    map_path = export_dir / "video_index_map.txt"
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    mapped = export_rows(export_dir)
    missing = sorted(set(outer_test) - set(mapped))
    if missing:
        raise ValueError(f"Fold {outer_fold} outer export misses cases: {missing[:5]}")
    for case in outer_test:
        index = mapped[case]
        for suffix in ("_raw.npy", ".npy", "_pred.npy"):
            path = export_dir / f"{index}{suffix}"
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
    for name in ("mapping.txt", "ground_truth.csv"):
        if not (export_dir / name).is_file():
            raise FileNotFoundError(export_dir / name)
    return {
        "path": str(export_dir),
        "outer_fold": outer_fold,
        "outer_test_video_count": len(outer_test),
        "export_map_video_count": len(mapped),
        "video_index_map_sha256": file_sha256(map_path),
        "mapping_sha256": file_sha256(export_dir / "mapping.txt"),
        "ground_truth_sha256": file_sha256(export_dir / "ground_truth.csv"),
        "all_outer_test_artifacts_present": True,
        "stream_semantics": {
            "raw": "pre-postprocessing decoder probabilities ({index}_raw.npy)",
            "canonical": "median-smoothed normalized probabilities ({index}.npy)",
            "official": "DiffAct final discrete predictions ({index}_pred.npy)",
        },
    }


def import_artifact_hashes(
    source_run: Path, heldout_cases: Sequence[str]
) -> Dict[str, str]:
    output_dir = source_run / "softmax_heldout"
    mapped = export_rows(output_dir)
    if [case for case, _ in sorted(mapped.items(), key=lambda item: item[1])] != list(
        heldout_cases
    ):
        raise ValueError(f"Fold-1 import export order differs from heldout manifest: {source_run}")
    paths = [
        source_run / "config.json",
        source_run / "task_complete.json",
        output_dir / "video_index_map.txt",
        output_dir / "mapping.txt",
        output_dir / "ground_truth.csv",
    ]
    for case in heldout_cases:
        index = mapped[case]
        paths.extend(
            [
                output_dir / f"{index}_raw.npy",
                output_dir / f"{index}.npy",
                output_dir / f"{index}_pred.npy",
            ]
        )
    hashes: Dict[str, str] = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        hashes[path.relative_to(source_run).as_posix()] = file_sha256(path)
    return hashes


def record_fold1_import(
    *,
    study_dir: Path,
    task: Mapping[str, Any],
    source_run: Path,
    source_study: Path,
    expected_training_signature: Mapping[str, Any],
) -> Dict[str, Any]:
    source_config = load_json(source_run / "config.json")
    if training_signature(source_config) != dict(expected_training_signature):
        raise ValueError(f"Fold-1 import training config changed: {source_run}")
    heldout = read_bundle(Path(task["heldout_manifest"]))
    artifact_hashes = import_artifact_hashes(source_run, heldout)
    checkpoint = Path(task["final_checkpoint"])
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "imported_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"],
        "outer_fold": 1,
        "inner_fold": task["inner_fold"],
        "source_study": str(source_study),
        "source_study_metadata_sha256": file_sha256(
            source_study / "study_metadata.json"
        ),
        "source_run": str(source_run),
        "checkpoint_sha256": file_sha256(checkpoint),
        "artifact_sha256": artifact_hashes,
        "artifact_digest": canonical_digest(artifact_hashes),
        "heldout_case_count": len(heldout),
        "hash_verification": "checkpoint plus every heldout export artifact",
    }
    atomic_write_json(Path(task["import_manifest_path"]), payload)
    return payload


def build_study(args: argparse.Namespace) -> None:
    study_dir = args.study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    v1_study_dir = args.v1_study_dir.resolve()
    outer_export_root = args.outer_export_root.resolve()
    if study_dir.exists():
        raise FileExistsError(
            f"Study directory already exists: {study_dir}. Choose a new immutable version."
        )
    baseline_config_path = diffact_root / "configs" / "Breakfast-Trained-S1.json"
    for path in (baseline_config_path, v1_study_dir / "study_metadata.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = load_json(baseline_config_path)
    if int(baseline["num_epochs"]) != FINAL_EPOCH + 1:
        raise ValueError("Breakfast schedule does not save the locked final epoch")
    if FINAL_EPOCH % int(baseline["log_freq"]):
        raise ValueError("Final epoch is outside the checkpoint cadence")

    v1_metadata = load_json(v1_study_dir / "study_metadata.json")
    if v1_metadata.get("protocol_version") != "breakfast-video-aware-selector-oof-v1":
        raise ValueError("Fold-1 import source has an unexpected protocol")

    study_dir.mkdir(parents=True)
    (study_dir / "logs").mkdir()
    manifests_root = study_dir / "manifests"
    folds_metadata: Dict[str, Dict[str, Any]] = {}
    tasks: List[Dict[str, Any]] = []

    for outer_fold in OUTER_FOLDS:
        outer_train, outer_test = official_splits(data_root, outer_fold)
        if len(outer_train) + len(outer_test) != 1712:
            raise ValueError(
                f"Unexpected fold-{outer_fold} cardinality: "
                f"train={len(outer_train)}, test={len(outer_test)}"
            )
        infos = load_case_infos(data_root, outer_train)
        train_participants = {info.participant for info in infos}
        test_participants = {
            info.participant for info in load_case_infos(data_root, outer_test)
        }
        if len(train_participants) != 39 or len(test_participants) != 13:
            raise ValueError(
                f"Unexpected fold-{outer_fold} participant counts: "
                f"train={len(train_participants)}, test={len(test_participants)}"
            )
        inner_folds = make_subject_disjoint_inner_folds(infos, INNER_FOLDS, SEED)
        fold_manifest_root = manifests_root / f"outer_fold_{outer_fold}"
        outer_train_manifest = fold_manifest_root / "outer_train_cases.txt"
        outer_test_manifest = fold_manifest_root / "outer_test_cases.txt"
        write_bundle(outer_train_manifest, outer_train)
        write_bundle(outer_test_manifest, outer_test)
        outer_export = validate_outer_export(
            outer_export_root / f"fold_{outer_fold}", outer_test, outer_fold
        )
        inner_metadata: Dict[str, Dict[str, Any]] = {}
        all_train_set = set(outer_train)

        for inner_fold in range(1, INNER_FOLDS + 1):
            heldout_cases = inner_folds[inner_fold]
            heldout_set = set(heldout_cases)
            train_cases = sorted(all_train_set - heldout_set)
            train_people = {
                info.participant for info in infos if info.case_id in train_cases
            }
            heldout_people = {
                info.participant for info in infos if info.case_id in heldout_set
            }
            if train_people.intersection(heldout_people):
                raise ValueError(
                    f"Outer {outer_fold}/inner {inner_fold} leaks participants"
                )
            if len(train_people) != 26 or len(heldout_people) != 13:
                raise ValueError(
                    f"Outer {outer_fold}/inner {inner_fold} expected 26/13 participants"
                )
            inner_manifest_root = fold_manifest_root / f"inner_fold_{inner_fold}"
            train_manifest = inner_manifest_root / "train_cases.txt"
            heldout_manifest = inner_manifest_root / "heldout_cases.txt"
            write_bundle(train_manifest, train_cases)
            write_bundle(heldout_manifest, heldout_cases)
            view_root = (
                study_dir
                / "diffact_dataset_views"
                / f"outer_fold_{outer_fold}"
                / f"inner_fold_{inner_fold}"
            )
            create_dataset_view(view_root, data_root, train_cases, heldout_cases)
            align_dir = (
                study_dir / "align" / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}"
            )
            create_alignment_dir(align_dir, data_root, heldout_cases)

            task_id = f"breakfast_outer{outer_fold}_inner{inner_fold}_seed{SEED}"
            naming = f"{task_id}_oof"
            run_dir = study_dir / "runs" / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}"
            config = dict(baseline)
            config.update(
                {
                    "naming": naming,
                    "root_data_dir": str(view_root),
                    "split_id": 1,
                    "result_dir": str(run_dir / "training"),
                    "random_seed": SEED,
                    "initialization_seed": SEED,
                    "evaluate_during_training": False,
                    "log_train_results": False,
                    "selector_protocol_version": PROTOCOL_VERSION,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "training_subset_manifest": str(train_manifest),
                    "heldout_oof_manifest": str(heldout_manifest),
                    "checkpoint_selection": (
                        "pre_specified_final_epoch_no_heldout_selection"
                    ),
                    "pre_specified_final_epoch": FINAL_EPOCH,
                }
            )
            config_path = run_dir / "config.json"
            atomic_write_json(config_path, config)
            imported = outer_fold == 1
            source_run = (
                v1_study_dir / "runs" / f"inner_fold_{inner_fold}"
                if imported
                else run_dir
            )
            source_naming = (
                f"breakfast_outer1_inner{inner_fold}_seed{SEED}_oof"
                if imported
                else naming
            )
            task: Dict[str, Any] = {
                "task_id": task_id,
                "execution_mode": "imported" if imported else "train",
                "dataset": DATASET,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "seed": SEED,
                "gpu": GPU_ASSIGNMENT[inner_fold - 1],
                "config_path": str(config_path),
                "train_manifest": str(train_manifest),
                "heldout_manifest": str(heldout_manifest),
                "align_dir": str(align_dir),
                "run_dir": str(run_dir),
                "artifact_run_dir": str(source_run),
                "naming": naming,
                "artifact_naming": source_naming,
                "final_checkpoint": str(
                    source_run / "training" / source_naming / f"epoch-{FINAL_EPOCH}.model"
                ),
                "softmax_output_dir": str(source_run / "softmax_heldout"),
                "train_summary": fold_summary(infos, train_cases),
                "heldout_summary": fold_summary(infos, heldout_cases),
                "import_manifest_path": (
                    str(
                        study_dir
                        / "imports"
                        / f"outer_fold_{outer_fold}"
                        / f"inner_fold_{inner_fold}"
                        / "import_complete.json"
                    )
                    if imported
                    else None
                ),
            }
            tasks.append(task)
            inner_metadata[str(inner_fold)] = {
                "train_manifest": str(train_manifest),
                "heldout_manifest": str(heldout_manifest),
                "train_summary": task["train_summary"],
                "heldout_summary": task["heldout_summary"],
                "train_manifest_sha256": file_sha256(train_manifest),
                "heldout_manifest_sha256": file_sha256(heldout_manifest),
            }
        folds_metadata[str(outer_fold)] = {
            "outer_train_manifest": str(outer_train_manifest),
            "outer_test_manifest": str(outer_test_manifest),
            "outer_train_manifest_sha256": file_sha256(outer_train_manifest),
            "outer_test_manifest_sha256": file_sha256(outer_test_manifest),
            "outer_train_case_count": len(outer_train),
            "outer_test_case_count": len(outer_test),
            "inner_folds": inner_metadata,
            "outer_release_export": outer_export,
        }

    # Fold 1 must reproduce the exact deterministic manifests that generated v1.
    for inner_fold in range(1, INNER_FOLDS + 1):
        new_inner = folds_metadata["1"]["inner_folds"][str(inner_fold)]
        old_inner = v1_metadata["inner_folds"][str(inner_fold)]
        for kind in ("train", "heldout"):
            new_path = Path(new_inner[f"{kind}_manifest"])
            old_path = Path(old_inner[f"{kind}_manifest"])
            if new_path.read_bytes() != old_path.read_bytes():
                raise ValueError(
                    f"Fold-1 {kind} manifest changed for inner fold {inner_fold}"
                )

    imports: List[Dict[str, Any]] = []
    expected_signature = training_signature(load_json(Path(tasks[0]["config_path"])))
    for task in tasks:
        if task["execution_mode"] == "imported":
            imports.append(
                record_fold1_import(
                    study_dir=study_dir,
                    task=task,
                    source_run=Path(task["artifact_run_dir"]),
                    source_study=v1_study_dir,
                    expected_training_signature=training_signature(
                        load_json(Path(task["config_path"]))
                    ),
                )
            )

    selector_config = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_variant": "base",
        "variants": {
            "base": {
                "feature_columns": list(BASE_NUMERIC_FEATURES),
                "pre_registered_primary": True,
            },
            "base_plus_metadata": {
                "base_feature_columns": list(BASE_NUMERIC_FEATURES),
                "additional_feature_prefixes": ["task__", "camera__"],
                "analysis_time_ablation_only": True,
            },
            "base_plus_shape": {
                "base_feature_columns": list(BASE_NUMERIC_FEATURES),
                "additional_feature_columns": list(SHAPE_FEATURES),
                "analysis_time_exploratory_only": True,
            },
        },
        "forbidden_primary_feature_prefixes": list(
            FORBIDDEN_PRIMARY_FEATURE_PREFIXES
        ),
        "forbidden_variants": ["base_plus_dfg", "base_plus_prefix_petri"],
        "fail_closed_feature_assertion": True,
    }
    selector_config_path = study_dir / "selector_config.json"
    atomic_write_json(selector_config_path, selector_config)

    repair_corpus_schema = {
        "protocol_version": PROTOCOL_VERSION,
        "row_scope": "one official-mode predicted segment from an OOF video",
        "case_record_count": sum(
            int(fold["outer_train_case_count"])
            for fold in folds_metadata.values()
        ),
        "candidate_pool": {
            "top_k": REPAIR_CANDIDATE_TOP_K,
            "source": "segment-mean raw pre-purge probability matrix",
            "fields_per_rank": list(REPAIR_CANDIDATE_FIELDS),
            "columns": list(repair_candidate_columns()),
            "probabilities_sorted_descending": True,
        },
        "target_columns": [
            "correct_label",
            "correct_label_name",
            "correct_label_fraction",
            "correct_label_probability_rank",
            "correct_label_in_top5",
            "error_fraction",
            "is_fully_correct",
        ],
        "selector_score_columns": [
            "base_score",
            "base_plus_metadata_score",
            "base_plus_shape_score",
        ],
        "primary_feature_columns": list(BASE_NUMERIC_FEATURES),
        "exploratory_shape_feature_columns": list(SHAPE_FEATURES),
        "abstention_rule": (
            "A future repair head abstains by default when the desired label is "
            "outside the saved top-5 candidate pool."
        ),
    }
    repair_corpus_schema_path = study_dir / "repair_corpus_schema.json"
    atomic_write_json(repair_corpus_schema_path, repair_corpus_schema)

    provenance = source_provenance(diffact_root)
    training_tasks = [task for task in tasks if task["execution_mode"] == "train"]
    expected_oof_case_records = sum(
        int(fold["outer_train_case_count"]) for fold in folds_metadata.values()
    )
    metadata: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "study_dir": str(study_dir),
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "dataset": DATASET,
        "outer_folds": list(OUTER_FOLDS),
        "seed": SEED,
        "inner_fold_count": INNER_FOLDS,
        "subject_disjoint_inner_folds": True,
        "folds": folds_metadata,
        "baseline_config": str(baseline_config_path),
        "baseline_config_sha256": file_sha256(baseline_config_path),
        "training_signature": expected_signature,
        "training_signature_sha256": canonical_digest(expected_signature),
        "pre_specified_final_epoch": FINAL_EPOCH,
        "checkpoint_selection": "pre_specified_final_epoch_no_test_selection",
        "v1_import_study": str(v1_study_dir),
        "v1_import_study_metadata_sha256": file_sha256(
            v1_study_dir / "study_metadata.json"
        ),
        "import_count": len(imports),
        "new_training_count": len(training_tasks),
        "import_contract": (
            "Fold-1 checkpoint and every heldout export artifact are hash-verified; "
            "fold 1 is not retrained."
        ),
        "selector_config": str(selector_config_path),
        "selector_config_sha256": file_sha256(selector_config_path),
        "repair_corpus_schema": str(repair_corpus_schema_path),
        "repair_corpus_schema_sha256": file_sha256(repair_corpus_schema_path),
        "primary_feature_columns": list(BASE_NUMERIC_FEATURES),
        "primary_feature_count": len(BASE_NUMERIC_FEATURES),
        "forbidden_primary_feature_prefixes": list(
            FORBIDDEN_PRIMARY_FEATURE_PREFIXES
        ),
        "feature_provenance_rule": (
            "Each OOF video's duration/rarity features use only its matching inner-train "
            "manifest; outer-test features use only that fold's official outer train."
        ),
        "variants": ["base", "base_plus_metadata", "base_plus_shape"],
        "process_feature_policy": "No DFG or Petri variants; rejected in fold-1 pilot.",
        "frame_budgets": list(FRAME_BUDGETS),
        "success_rule": {
            "mode": "official",
            "variant": "base",
            "budget": 0.05,
            "all_outer_folds": {
                "minimum_error_precision_pct": 85.0,
                "minimum_error_recall_pct": 15.0,
                "maximum_recall_range_across_folds_pp": 8.0,
            },
            "pooled": {
                "minimum_error_precision_pct": 85.0,
                "minimum_error_recall_pct": 18.0,
            },
            "on_pass": "green_light_deployable_repair_head",
        },
        "shape_variant_comparison_rule": {
            "mode": "official",
            "budget": 0.05,
            "variant": "base_plus_shape",
            "reference": "base",
            "minimum_pooled_error_recall_gain_pp": 0.5,
            "minimum_outer_folds_with_positive_error_recall_gain": 3,
            "minimum_pooled_oracle_acc_gain_difference_pp": 0.0,
            "minimum_pooled_oracle_f1_at_25_gain_difference_pp": 0.0,
            "on_pass": "retain_shape_features_for_repair_head_localizer",
            "on_fail": "drop_shape_features_and_keep_15_feature_base",
            "analysis_time_exploratory_only": True,
        },
        "repair_corpus_contract": {
            "scope": "oof_validation",
            "mode": "official",
            "expected_case_records": expected_oof_case_records,
            "unit": "predicted segment",
            "includes_correct_label": True,
            "includes_selector_score": True,
            "candidate_pool_size": REPAIR_CANDIDATE_TOP_K,
            "candidate_pool_source": (
                "segment-mean raw pre-purge probability matrix"
            ),
            "candidate_columns_per_rank": list(REPAIR_CANDIDATE_FIELDS),
            "includes_correct_label_probability_rank": True,
            "abstention_is_default_when_correct_label_outside_top5": True,
            "outer_fold_1_feasibility_measurement": {
                "flagged_wrong_frames_gt_is_rank_2_pct": 26.3,
                "flagged_wrong_frames_gt_in_top_3_pct": 39.0,
                "flagged_wrong_frames_gt_in_top_5_pct": 62.3,
                "flagged_wrong_frames_gt_probability_rank_median": 4.0,
                "majority_wrong_flagged_segments_runner_up_matches_gt_majority_pct": 18.8,
                "approximate_gt_outside_top_5_pct": 37.7,
            },
        },
        "oracle_outputs_are_diagnostic_only": True,
        "gpu_policy": {
            "physical_gpu_ids": list(GPU_ASSIGNMENT),
            "one_inner_fold_queue_per_gpu": True,
            "wait_for_occupancy": True,
            "required_consecutive_free_checks": REQUIRED_CONSECUTIVE_FREE_CHECKS,
            "free_check_interval_seconds": FREE_CHECK_INTERVAL_SECONDS,
            "automatic_fallback": False,
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
    for gpu in GPU_ASSIGNMENT:
        gpu_tasks = [
            task
            for task in training_tasks
            if int(task["gpu"]) == gpu
        ]
        queue_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"export CUDA_VISIBLE_DEVICES={gpu}",
        ]
        for task in gpu_tasks:
            queue_lines.append(
                shell_command(
                    [
                        python,
                        "-u",
                        SCRIPT_DIR / "run_oof_task.py",
                        "--study-dir",
                        study_dir,
                        "--task-id",
                        task["task_id"],
                    ]
                )
            )
        queue_path = study_dir / "queues" / f"gpu_{gpu}.sh"
        write_executable(queue_path, "\n".join(queue_lines) + "\n")
        waiter_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"gpu={gpu}",
            "consecutive=0",
            f"required={REQUIRED_CONSECUTIVE_FREE_CHECKS}",
            "while (( consecutive < required )); do",
            (
                '  busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory,'
                'process_name --format=csv,noheader,nounits | sed \'/^[[:space:]]*$/d\')'
            ),
            '  if [[ -z "$busy" ]]; then',
            "    consecutive=$((consecutive + 1))",
            '    echo "GPU $gpu free check $consecutive/$required"',
            "  else",
            "    consecutive=0",
            '    echo "GPU $gpu occupied; waiting:"',
            '    echo "$busy"',
            "  fi",
            "  if (( consecutive < required )); then",
            f"    sleep {FREE_CHECK_INTERVAL_SECONDS}",
            "  fi",
            "done",
            'echo "GPU $gpu passed two consecutive free checks; starting queue"',
            f"exec {shlex.quote(str(queue_path))}",
        ]
        write_executable(
            study_dir / "waiters" / f"gpu_{gpu}.sh",
            "\n".join(waiter_lines) + "\n",
        )

    launch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(study_dir / 'logs'))}",
    ]
    for gpu in GPU_ASSIGNMENT:
        session = f"bfast_sel_all_v2_g{gpu}"
        waiter = study_dir / "waiters" / f"gpu_{gpu}.sh"
        log = study_dir / "logs" / f"gpu_{gpu}.tmux.log"
        command = f"{shlex.quote(str(waiter))} 2>&1 | tee -a {shlex.quote(str(log))}"
        launch_lines.extend(
            [
                f"if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then",
                f"  echo 'Session already exists: {session}'",
                "else",
                f"  tmux new-session -d -s {shlex.quote(session)} {shlex.quote(command)}",
                f"  echo 'Started waiter {session} for GPU {gpu}'",
                "fi",
            ]
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
        + shell_command(
            [python, "-u", SCRIPT_DIR / "analyze_selector.py", "--study-dir", study_dir]
        )
        + "\n",
    )
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable generated four-fold Breakfast selector study.",
            f"Protocol: {PROTOCOL_VERSION}",
            "Do not edit configs, manifests, imports, selector_config, repair schema, or tasks.",
            "Generate a new versioned directory for any change.",
            "prepare_study.py did not launch training.",
        ],
    )
    print(f"Prepared immutable study: {study_dir}")
    print(f"Fold-1 imports: {len(imports)}; net new trainings: {len(training_tasks)}")
    print(f"Primary feature count: {len(BASE_NUMERIC_FEATURES)}")
    print(f"Source digest: {provenance['source_digest']}")
    print("Nothing was launched. Review imports, configs, feature assertions, and waiters.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--v1-study-dir", type=Path, default=DEFAULT_V1_STUDY_DIR)
    parser.add_argument(
        "--outer-export-root", type=Path, default=DEFAULT_OUTER_EXPORT_ROOT
    )
    args = parser.parse_args()
    build_study(args)


if __name__ == "__main__":
    main()
