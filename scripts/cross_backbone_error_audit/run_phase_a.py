#!/usr/bin/env python3
"""Run the hash-locked Phase-A inventory and paper reconciliation."""

from __future__ import annotations

import argparse
import fcntl
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from common import (
    BACKBONES,
    DATASETS,
    PUBLISHED_METRICS,
    RECONCILIATION,
    atomic_write_json,
    canonical_digest,
    cell_paths,
    file_sha256,
    load_json,
    normalize_bundle_case,
    parse_mapping,
    parse_video_index,
    read_nonempty_lines,
    verify_source_provenance,
)
from metrics import OVERLAPS, aggregate_standard, f_counts, per_video_metrics


METRICS = ("acc", "edit", "f1@10", "f1@25", "f1@50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    rows = list(manifest["files"])
    if len(rows) != int(manifest["file_count"]):
        raise RuntimeError("Input manifest file-count mismatch")
    seen_roles: set[str] = set()
    compact: list[dict[str, str]] = []
    for row in rows:
        role = str(row["role"])
        if role in seen_roles:
            raise RuntimeError(f"Duplicate manifest role: {role}")
        seen_roles.add(role)
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"Input size changed: {role}")
        observed = file_sha256(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"Input hash changed: {role}")
        compact.append({"role": role, "sha256": observed})
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("Input manifest digest mismatch")


def load_ground_truth(
    data_root: Path, dataset: str, case_name: str, name_to_id: Mapping[str, int], sample_rate: int
) -> np.ndarray:
    candidates = [
        data_root / dataset / "groundTruth" / f"{case_name}.txt",
        data_root / dataset / "groundTruth" / case_name,
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing ground truth for {dataset}/{case_name}")
    labels = read_nonempty_lines(path)
    try:
        values = np.asarray([name_to_id[label] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown ground-truth label in {path}: {error}") from error
    return values[::sample_rate]


def reconciliation_status(deltas: Mapping[str, float]) -> str:
    absolute = np.asarray([abs(float(deltas[metric])) for metric in METRICS])
    if (
        float(absolute.max()) <= RECONCILIATION["pass_max_abs_delta_pp"]
        and float(absolute.mean()) <= RECONCILIATION["pass_mean_abs_delta_pp"]
    ):
        return "PASS"
    if (
        float(absolute.max()) <= RECONCILIATION["notes_max_abs_delta_pp"]
        and float(absolute.mean()) <= RECONCILIATION["notes_mean_abs_delta_pp"]
    ):
        return "PASS_WITH_NOTES"
    return "FAIL"


def execute(study_dir: Path) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source_provenance(metadata["source_provenance"])
    verify_manifest(manifest)
    if config["phase_c_launch_allowed"] or config["gpu_training_allowed"]:
        raise RuntimeError("Phase-A study unexpectedly authorizes later phases")

    data_root = Path(config["data_root"])
    roots = {key: Path(value) for key, value in config["backbone_roots"].items()}
    inventory_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for backbone in BACKBONES:
        for dataset, dataset_config in DATASETS.items():
            sample_rate = int(dataset_config["sample_rate"])
            official_mapping = data_root / dataset / "mapping.txt"
            id_to_name, name_to_id = parse_mapping(official_mapping)
            paths = cell_paths(backbone, dataset, 1, roots)
            stored_id_to_name, _ = parse_mapping(paths["stored_mapping"])
            mapping_match = id_to_name == stored_id_to_name
            if not mapping_match:
                raise RuntimeError(f"Class mapping mismatch for {backbone}/{dataset}")
            video_index = parse_video_index(paths["video_index_map"])
            background_ids = {
                class_id for class_id, label in id_to_name.items() if label == "background"
            }
            dataset_cases: list[tuple[np.ndarray, np.ndarray]] = []
            dataset_fold_metrics: list[dict[str, float]] = []

            for fold in range(1, int(dataset_config["folds"]) + 1):
                cell = cell_paths(backbone, dataset, fold, roots)
                test_bundle = data_root / dataset / "splits" / f"test.split{fold}.bundle"
                test_cases = [normalize_bundle_case(value) for value in read_nonempty_lines(test_bundle)]
                if len(test_cases) != len(set(test_cases)):
                    raise RuntimeError(f"Duplicate test cases in {test_bundle}")
                missing_from_map = sorted(set(test_cases) - set(video_index))
                if missing_from_map:
                    raise RuntimeError(
                        f"Video-index coverage failure for {backbone}/{dataset}/fold{fold}: {missing_from_map[:5]}"
                    )
                cell_cases: list[tuple[np.ndarray, np.ndarray]] = []
                probability_bytes = 0
                max_probability_sum_error = 0.0
                probability_min = float("inf")
                probability_max = float("-inf")

                for case_name in test_cases:
                    case_index = int(video_index[case_name])
                    probability_path = cell["softmax_dir"] / f"{case_index}.npy"
                    probability = np.load(probability_path, allow_pickle=False)
                    if probability.ndim != 2 or probability.shape[0] != len(id_to_name):
                        raise ValueError(
                            f"Invalid probability shape {probability.shape} for {backbone}/{dataset}/{case_name}"
                        )
                    if not np.isfinite(probability).all():
                        raise ValueError(f"Non-finite probability values in {probability_path}")
                    probability_min = min(probability_min, float(probability.min()))
                    probability_max = max(probability_max, float(probability.max()))
                    normalization_error = float(np.max(np.abs(probability.sum(axis=0) - 1.0)))
                    max_probability_sum_error = max(max_probability_sum_error, normalization_error)
                    if probability_min < -1e-6 or probability_max > 1.0 + 1e-6:
                        raise ValueError(f"Probability range failure in {probability_path}")
                    if normalization_error > 1e-4:
                        raise ValueError(f"Probability normalization failure in {probability_path}")
                    target = load_ground_truth(
                        data_root, dataset, case_name, name_to_id, sample_rate
                    )
                    if probability.shape[1] != len(target):
                        raise ValueError(
                            f"Frame mismatch for {backbone}/{dataset}/fold{fold}/{case_name}: "
                            f"probability={probability.shape[1]} target={len(target)}"
                        )
                    prediction = probability.argmax(axis=0).astype(np.int64)
                    row = {
                        "backbone": backbone,
                        "dataset": dataset,
                        "fold": fold,
                        "case_id": case_name,
                        "case_index": case_index,
                        "n_frames": len(target),
                        "n_errors": int(np.sum(prediction != target)),
                    }
                    row.update(per_video_metrics(prediction, target, background_ids))
                    for overlap in OVERLAPS:
                        tp, fp, fn = f_counts(prediction, target, overlap, background_ids)
                        suffix = int(round(overlap * 100))
                        row[f"f1@{suffix}_tp"] = tp
                        row[f"f1@{suffix}_fp"] = fp
                        row[f"f1@{suffix}_fn"] = fn
                    video_rows.append(row)
                    cell_cases.append((prediction, target))
                    dataset_cases.append((prediction, target))
                    probability_bytes += int(probability_path.stat().st_size)

                fold_metric = aggregate_standard(cell_cases, background_ids)
                dataset_fold_metrics.append(fold_metric)
                fold_rows.append(
                    {
                        "backbone": backbone,
                        "dataset": dataset,
                        "fold": fold,
                        "n_cases": len(cell_cases),
                        "n_frames": sum(len(target) for _, target in cell_cases),
                        **fold_metric,
                        **{
                            f"macro_{metric}": float(
                                np.mean(
                                    [
                                        row[metric]
                                        for row in video_rows
                                        if row["backbone"] == backbone
                                        and row["dataset"] == dataset
                                        and row["fold"] == fold
                                    ]
                                )
                            )
                            for metric in METRICS
                        },
                    }
                )
                inventory_rows.append(
                    {
                        "backbone": backbone,
                        "dataset": dataset,
                        "fold": fold,
                        "status": "USABLE",
                        "n_expected_cases": len(test_cases),
                        "n_probability_files_used": len(cell_cases),
                        "n_frames": sum(len(target) for _, target in cell_cases),
                        "probability_bytes": probability_bytes,
                        "num_classes": len(id_to_name),
                        "sample_rate": sample_rate,
                        "mapping_match": mapping_match,
                        "checkpoint_path": str(cell["checkpoint"]),
                        "checkpoint_sha256": file_sha256(cell["checkpoint"]),
                        "probability_min": probability_min,
                        "probability_max": probability_max,
                        "max_probability_sum_error": max_probability_sum_error,
                    }
                )

            pooled = aggregate_standard(dataset_cases, background_ids)
            # The original papers report the arithmetic mean across official folds.
            # Reconciliation must use that convention; pooled metrics remain an
            # audit readout but do not decide comparability.
            observed = {
                metric: float(np.mean([row[metric] for row in dataset_fold_metrics]))
                for metric in METRICS
            }
            published = PUBLISHED_METRICS[(backbone, dataset)]
            deltas = {metric: float(observed[metric] - published[metric]) for metric in METRICS}
            status = reconciliation_status(deltas)
            matching_video_rows = [
                row
                for row in video_rows
                if row["backbone"] == backbone and row["dataset"] == dataset
            ]
            aggregate_rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "n_folds": int(dataset_config["folds"]),
                    "n_cases": len(dataset_cases),
                    "n_frames": sum(len(target) for _, target in dataset_cases),
                    "reconciliation_status": status,
                    "mean_abs_delta_pp": float(np.mean([abs(value) for value in deltas.values()])),
                    "max_abs_delta_pp": float(max(abs(value) for value in deltas.values())),
                    **{f"observed_{metric}": float(observed[metric]) for metric in METRICS},
                    **{f"pooled_{metric}": float(pooled[metric]) for metric in METRICS},
                    **{f"published_{metric}": float(published[metric]) for metric in METRICS},
                    **{f"delta_{metric}": float(deltas[metric]) for metric in METRICS},
                    **{
                        f"macro_{metric}": float(np.mean([row[metric] for row in matching_video_rows]))
                        for metric in METRICS
                    },
                    **{
                        f"frame_weighted_video_{metric}": float(
                            np.average(
                                [row[metric] for row in matching_video_rows],
                                weights=[row["n_frames"] for row in matching_video_rows],
                            )
                        )
                        for metric in METRICS
                    },
                }
            )

    inventory = pd.DataFrame(inventory_rows).sort_values(["backbone", "dataset", "fold"])
    videos = pd.DataFrame(video_rows).sort_values(["backbone", "dataset", "fold", "case_id"])
    folds = pd.DataFrame(fold_rows).sort_values(["backbone", "dataset", "fold"])
    aggregates = pd.DataFrame(aggregate_rows).sort_values(["backbone", "dataset"])
    if len(inventory) != 26 or not (inventory.status == "USABLE").all():
        raise RuntimeError("Expected all 26 legacy backbone/dataset/fold cells to be usable")

    result_dir = study_dir / "results"
    inventory.to_csv(result_dir / "inventory_cells.csv", index=False)
    videos.to_csv(result_dir / "per_video_metrics.csv", index=False)
    folds.to_csv(result_dir / "per_fold_metrics.csv", index=False)
    aggregates.to_csv(result_dir / "paper_reconciliation.csv", index=False)

    status_counts = aggregates.reconciliation_status.value_counts().to_dict()
    failed_pairs = [
        (str(row.backbone), str(row.dataset))
        for row in aggregates.itertuples(index=False)
        if row.reconciliation_status == "FAIL"
    ]
    phase_b_required_cells = [
        {"backbone": backbone, "dataset": dataset, "fold": fold}
        for backbone, dataset in failed_pairs
        for fold in range(1, int(DATASETS[dataset]["folds"]) + 1)
    ]
    phase_b_required = bool(phase_b_required_cells)
    decision = {
        "phase": "A",
        "inventory_cells_expected": 26,
        "inventory_cells_usable": int(len(inventory)),
        "legacy_probability_exports_complete": True,
        "phase_b_training_required": phase_b_required,
        "phase_b_missing_cells": [],
        "phase_b_failed_reconciliation_pairs": [
            {"backbone": backbone, "dataset": dataset}
            for backbone, dataset in failed_pairs
        ],
        "phase_b_required_cells": phase_b_required_cells,
        "phase_b_required_cell_count": len(phase_b_required_cells),
        "phase_b_planning_estimate": config["phase_b_planning_estimate"],
        "comparability_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "phase_c_may_run": False,
        "phase_c_block_reason": (
            "Phase B retraining/reconciliation and Fable review required"
            if phase_b_required
            else "Fable Phase-A review required even when Phase B is empty"
        ),
        "gpu_launched": False,
        "sealed_studies_touched": False,
    }
    decision["decision_digest"] = canonical_digest(decision)
    atomic_write_json(result_dir / "phase_a_decision.json", decision)

    lines = [
        "# Cross-backbone error audit — Phase A findings",
        "",
        f"**Inventory:** {len(inventory)}/26 legacy cells are usable; every official test case has a finite, normalized, frame-aligned probability matrix and a final checkpoint.",
        "",
        f"**Phase B:** {'required because at least one paper reconciliation failed.' if phase_b_required else 'no missing cells and no failed paper reconciliation; no retraining is currently justified.'}",
        "",
        (
            "**Planning estimate:** the 17 failed cells represent approximately "
            f"{config['phase_b_planning_estimate']['active_gpu_hours']:.1f} active GPU-hours "
            "from historical adjacent-checkpoint cadence. Four-GPU ideal wall time is "
            f"approximately {config['phase_b_planning_estimate']['four_gpu_ideal_wall_hours']:.1f} "
            f"hours; budget {config['phase_b_planning_estimate']['buffered_wall_days']} days and "
            f"about {config['phase_b_planning_estimate']['historical_checkpoint_storage_gib']:.1f} "
            "GiB. This is planning-only and must be refreshed in the reviewed Phase-B study."
        ),
        "",
        "## Paper reconciliation",
        "",
        "| Backbone | Dataset | Status | Mean | Max | Acc | Edit | F1@10 | F1@25 | F1@50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in aggregates.iterrows():
        lines.append(
            f"| {row['backbone']} | {row['dataset']} | {row['reconciliation_status']} | "
            f"{row['mean_abs_delta_pp']:.2f} | {row['max_abs_delta_pp']:.2f} | "
            f"{row['observed_acc']:.2f} ({row['delta_acc']:+.2f}) | "
            f"{row['observed_edit']:.2f} ({row['delta_edit']:+.2f}) | "
            f"{row['observed_f1@10']:.2f} ({row['delta_f1@10']:+.2f}) | "
            f"{row['observed_f1@25']:.2f} ({row['delta_f1@25']:+.2f}) | "
            f"{row['observed_f1@50']:.2f} ({row['delta_f1@50']:+.2f}) |"
        )
    lines.extend(
        [
            "",
            "Parentheses are observed minus the exact paper row, in percentage points.",
            "",
            "## Governance",
            "",
            "This review study does not authorize Phase C or GPU training. Fable must review the input hashes, mapping/alignment checks, and reconciliation table first.",
            "",
            "The legacy repositories are dirty because their historical export code, checkpoints, and results are uncommitted. Their Git HEAD, tracked-diff digest, and every consumed input hash are recorded; no file in either repository was modified.",
        ]
    )
    (result_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    verify_manifest(manifest)
    outputs = sorted(path for path in result_dir.iterdir() if path.is_file())
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_digest": manifest["manifest_digest"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "phase_a_decision_digest": decision["decision_digest"],
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    atomic_write_json(result_dir / "phase_a_complete.json", complete)


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    lock_path = study_dir / ".phase_a.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Phase A is already running") from error
        try:
            execute(study_dir)
        except Exception as error:
            atomic_write_json(
                study_dir / "results" / "phase_a_failed.json",
                {
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
