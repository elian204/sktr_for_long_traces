#!/usr/bin/env python3
"""Aggregate the multi-fold confirmation and apply its pre-registered rule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

from common import (
    BASELINE_BOUNDARY_WEIGHT,
    FINAL_EPOCH,
    FOLDS,
    PRIMARY_BOUNDARY_WEIGHT,
    atomic_write_json,
    export_dir,
    load_json,
    read_bundle,
    verify_export,
    verify_source_digest,
)
from metrics import evaluate_export, fixed_broke_ledger
from run_variant import validate_completed_task


METRICS = (
    "acc",
    "edit",
    "f1@10",
    "f1@25",
    "f1@50",
    "boundary_f1_class_agnostic@5",
    "boundary_f1_class_agnostic@10",
    "boundary_f1_transition_aware@5",
    "boundary_f1_transition_aware@10",
    "boundary_offset_class_agnostic_mean_signed",
    "boundary_offset_class_agnostic_median_signed",
    "boundary_offset_class_agnostic_mean_absolute",
    "boundary_offset_class_agnostic_median_absolute",
    "boundary_offset_class_agnostic_p90_absolute",
    "boundary_offset_transition_aware_mean_signed",
    "boundary_offset_transition_aware_median_signed",
    "boundary_offset_transition_aware_mean_absolute",
    "boundary_offset_transition_aware_median_absolute",
    "boundary_offset_transition_aware_p90_absolute",
    "segment_count_ratio",
    "short_false_segment_rate",
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def summarize_inference_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "official_fold",
        "task_id",
        "decoder_boundary_loss",
        "role",
        "training_seed",
        "execution_mode",
        "checkpoint_epoch",
        "stream",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in metrics.groupby(groups, sort=True, dropna=False):
        row = dict(zip(groups, keys))
        row["n_inference_seeds"] = int(len(group))
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def six_checks(means: Mapping[str, float]) -> Dict[str, bool]:
    return {
        "boundary_f1_class_agnostic@10_improves": (
            means["delta_boundary_f1_class_agnostic@10"] > 0
        ),
        "edit_improves": means["delta_edit"] > 0,
        "f1@25_improves": means["delta_f1@25"] > 0,
        "accuracy_non_negative": means["delta_acc"] >= 0,
        "segment_count_ratio_not_inflated": (
            means["delta_segment_count_ratio"] <= 0.02
        ),
        "mean_absolute_boundary_offset_decreases": (
            means["delta_boundary_offset_class_agnostic_mean_absolute"] < 0
        ),
    }


def mean_deltas(frame: pd.DataFrame) -> Dict[str, float]:
    return {
        column: float(pd.to_numeric(frame[column], errors="raise").mean())
        for column in frame.columns
        if column.startswith("delta_")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()

    study_dir = args.study_dir.resolve()
    metadata = load_json(study_dir / "study_metadata.json")
    verify_source_digest(metadata, Path(metadata["diffact_root"]))
    tasks = load_json(study_dir / "tasks.json")["tasks"]
    for task in tasks:
        if validate_completed_task(task, metadata) is None:
            raise RuntimeError(f"Cannot analyze incomplete task: {task['task_id']}")

    data_root = Path(metadata["data_root"])
    case_ids_by_fold = {
        int(fold): read_bundle(
            Path(
                next(
                    task["test_manifest"]
                    for task in tasks
                    if int(task["official_fold"]) == int(fold)
                )
            )
        )
        for fold in metadata["official_folds"]
    }
    metric_rows: List[Dict[str, Any]] = []
    per_case_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[str, int, int, str], Dict[str, Any]] = {}
    ground_truth_by_fold: Dict[int, Dict[str, Any]] = {}
    for task in tasks:
        fold = int(task["official_fold"])
        case_ids = case_ids_by_fold[fold]
        for epoch in task["checkpoint_epochs"]:
            for inference_seed in task["inference_seeds"]:
                output_dir = export_dir(task, int(epoch), int(inference_seed))
                verify_export(output_dir, case_ids)
                summaries, case_metrics, matches, streams, ground_truth = evaluate_export(
                    data_root=data_root,
                    case_ids=case_ids,
                    output_dir=output_dir,
                )
                if fold in ground_truth_by_fold:
                    if set(ground_truth_by_fold[fold]) != set(ground_truth):
                        raise ValueError(f"Ground-truth cases changed within fold {fold}")
                else:
                    ground_truth_by_fold[fold] = ground_truth
                for stream, summary in summaries.items():
                    prefix = {
                        "official_fold": fold,
                        "task_id": task["task_id"],
                        "decoder_boundary_loss": task["decoder_boundary_loss"],
                        "role": task["role"],
                        "training_seed": task["training_seed"],
                        "execution_mode": task["execution_mode"],
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(inference_seed),
                        "stream": stream,
                        "output_dir": str(output_dir),
                    }
                    row = dict(prefix)
                    row.update(summary)
                    metric_rows.append(row)
                    predictions[
                        (task["task_id"], int(epoch), int(inference_seed), stream)
                    ] = streams[stream]
                for row in case_metrics:
                    row.update(
                        {
                            "official_fold": fold,
                            "task_id": task["task_id"],
                            "decoder_boundary_loss": task["decoder_boundary_loss"],
                            "role": task["role"],
                            "training_seed": task["training_seed"],
                            "execution_mode": task["execution_mode"],
                            "checkpoint_epoch": int(epoch),
                            "inference_seed": int(inference_seed),
                        }
                    )
                    per_case_rows.append(row)
                for row in matches:
                    row.update(
                        {
                            "official_fold": fold,
                            "task_id": task["task_id"],
                            "decoder_boundary_loss": task["decoder_boundary_loss"],
                            "role": task["role"],
                            "training_seed": task["training_seed"],
                            "execution_mode": task["execution_mode"],
                            "checkpoint_epoch": int(epoch),
                            "inference_seed": int(inference_seed),
                        }
                    )
                    boundary_rows.append(row)

    analysis_dir = study_dir / "analysis"
    metrics_frame = write_csv(analysis_dir / "fold_metrics.csv", metric_rows)
    write_csv(analysis_dir / "per_case_metrics.csv", per_case_rows)
    write_csv(analysis_dir / "boundary_matches.csv", boundary_rows)
    summary_frame = summarize_inference_seeds(metrics_frame)
    summary_frame.to_csv(analysis_dir / "sampling_seed_summary.csv", index=False)
    summary_frame.to_csv(analysis_dir / "checkpoint_curves.csv", index=False)

    task_by_fold_seed_weight = {
        (
            int(task["official_fold"]),
            int(task["training_seed"]),
            float(task["decoder_boundary_loss"]),
        ): task
        for task in tasks
    }
    metric_index = {
        (
            row["task_id"],
            int(row["checkpoint_epoch"]),
            int(row["inference_seed"]),
            row["stream"],
        ): row
        for row in metric_rows
    }

    paired_rows: List[Dict[str, Any]] = []
    ledger_rows: List[Dict[str, Any]] = []
    for task in tasks:
        fold = int(task["official_fold"])
        weight = float(task["decoder_boundary_loss"])
        seed = int(task["training_seed"])
        if weight == BASELINE_BOUNDARY_WEIGHT:
            continue
        baseline = task_by_fold_seed_weight.get(
            (fold, seed, BASELINE_BOUNDARY_WEIGHT)
        )
        if baseline is None:
            continue
        for epoch in task["checkpoint_epochs"]:
            for inference_seed in task["inference_seeds"]:
                for stream in ("pre_purge", "post_purge"):
                    base_key = (
                        baseline["task_id"],
                        int(epoch),
                        int(inference_seed),
                        stream,
                    )
                    variant_key = (
                        task["task_id"],
                        int(epoch),
                        int(inference_seed),
                        stream,
                    )
                    base_row = metric_index[base_key]
                    variant_row = metric_index[variant_key]
                    paired = {
                        "official_fold": fold,
                        "task_id": task["task_id"],
                        "baseline_task_id": baseline["task_id"],
                        "decoder_boundary_loss": weight,
                        "role": task["role"],
                        "training_seed": seed,
                        "execution_mode": task["execution_mode"],
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(inference_seed),
                        "stream": stream,
                    }
                    for metric in METRICS:
                        paired[f"delta_{metric}"] = (
                            float(variant_row[metric]) - float(base_row[metric])
                        )
                    ledger = fixed_broke_ledger(
                        ground_truth_by_fold[fold],
                        predictions[base_key],
                        predictions[variant_key],
                    )
                    paired.update(ledger)
                    paired_rows.append(paired)
                    ledger_rows.append(
                        {
                            key: value
                            for key, value in paired.items()
                            if key.startswith(
                                (
                                    "fixed",
                                    "broken",
                                    "net_",
                                    "wrong_",
                                    "unchanged",
                                    "prediction_",
                                )
                            )
                            or key
                            in {
                                "official_fold",
                                "task_id",
                                "baseline_task_id",
                                "decoder_boundary_loss",
                                "role",
                                "training_seed",
                                "execution_mode",
                                "checkpoint_epoch",
                                "inference_seed",
                                "stream",
                                "n_frames",
                            }
                        }
                    )
    paired_frame = write_csv(analysis_dir / "paired_deltas.csv", paired_rows)
    write_csv(analysis_dir / "fixed_broke_ledger.csv", ledger_rows)

    cross_seed_rows: List[Dict[str, Any]] = []
    fold_weight_pairs = sorted(
        {
            (fold, weight)
            for fold, seed, weight in task_by_fold_seed_weight
            if (fold, 0, weight) in task_by_fold_seed_weight
            and (fold, 1, weight) in task_by_fold_seed_weight
        }
    )
    for fold, weight in fold_weight_pairs:
        seed0 = task_by_fold_seed_weight[(fold, 0, weight)]
        seed1 = task_by_fold_seed_weight[(fold, 1, weight)]
        for epoch in seed0["checkpoint_epochs"]:
            for inference_seed in seed0["inference_seeds"]:
                for stream in ("pre_purge", "post_purge"):
                    key0 = (seed0["task_id"], int(epoch), int(inference_seed), stream)
                    key1 = (seed1["task_id"], int(epoch), int(inference_seed), stream)
                    row0, row1 = metric_index[key0], metric_index[key1]
                    noise = {
                        "official_fold": fold,
                        "decoder_boundary_loss": weight,
                        "seed0_task_id": seed0["task_id"],
                        "seed1_task_id": seed1["task_id"],
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(inference_seed),
                        "stream": stream,
                    }
                    for metric in METRICS:
                        delta = float(row1[metric]) - float(row0[metric])
                        noise[f"delta_seed1_minus_seed0_{metric}"] = delta
                        noise[f"absolute_delta_{metric}"] = abs(delta)
                    noise.update(
                        fixed_broke_ledger(
                            ground_truth_by_fold[fold],
                            predictions[key0],
                            predictions[key1],
                        )
                    )
                    cross_seed_rows.append(noise)
    cross_seed_frame = write_csv(
        analysis_dir / "cross_seed_noise.csv", cross_seed_rows
    )
    noise_group = [
        "official_fold",
        "decoder_boundary_loss",
        "checkpoint_epoch",
        "stream",
    ]
    numeric_noise_columns = [
        column
        for column in cross_seed_frame.columns
        if column.startswith(("delta_seed1_minus_seed0_", "absolute_delta_"))
        or column in {"prediction_disagreement_pct", "fixed_pct", "broken_pct"}
    ]
    noise_summary = (
        cross_seed_frame.groupby(noise_group, sort=True)[numeric_noise_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    noise_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in noise_summary.columns
    ]
    noise_summary.to_csv(analysis_dir / "cross_seed_noise_summary.csv", index=False)

    primary_rows = paired_frame[
        (paired_frame["decoder_boundary_loss"] == PRIMARY_BOUNDARY_WEIGHT)
        & (paired_frame["checkpoint_epoch"] == FINAL_EPOCH)
        & (paired_frame["stream"] == "post_purge")
    ]
    expected_per_fold = len(metadata["training_seeds"]) * len(
        metadata["inference_seeds"]
    )
    expected_total = len(metadata["official_folds"]) * expected_per_fold
    if len(primary_rows) != expected_total:
        raise ValueError(
            f"Primary expected {expected_total} paired rows; found {len(primary_rows)}"
        )

    per_fold_decisions: Dict[str, Any] = {}
    for fold in FOLDS:
        fold_rows = primary_rows[primary_rows["official_fold"] == fold]
        if len(fold_rows) != expected_per_fold:
            raise ValueError(
                f"Fold {fold} expected {expected_per_fold} primary rows; "
                f"found {len(fold_rows)}"
            )
        means = mean_deltas(fold_rows)
        checks = six_checks(means)
        per_fold_decisions[str(fold)] = {
            "n_paired_rows": len(fold_rows),
            "mean_deltas": means,
            "checks": checks,
            "six_check_gate_passes": all(checks.values()),
        }

    pooled_means = mean_deltas(primary_rows)
    pooled_checks = six_checks(pooled_means)
    positive_edit_folds = sum(
        decision["mean_deltas"]["delta_edit"] > 0
        for decision in per_fold_decisions.values()
    )
    positive_f1_folds = sum(
        decision["mean_deltas"]["delta_f1@25"] > 0
        for decision in per_fold_decisions.values()
    )
    sign_consistency = {
        "positive_delta_edit_folds": positive_edit_folds,
        "positive_delta_f1@25_folds": positive_f1_folds,
        "required_positive_folds": 3,
        "edit_sign_consistent": positive_edit_folds >= 3,
        "f1@25_sign_consistent": positive_f1_folds >= 3,
    }
    primary_claim = (
        all(pooled_checks.values())
        and sign_consistency["edit_sign_consistent"]
        and sign_consistency["f1@25_sign_consistent"]
    )
    completed_utc = datetime.now(timezone.utc).isoformat()
    decision = {
        "completed_utc": completed_utc,
        "primary_claim_supported": primary_claim,
        "onset_head_launch_gate_passes": primary_claim,
        "pre_registered_primary_boundary_weight": PRIMARY_BOUNDARY_WEIGHT,
        "baseline_boundary_weight": BASELINE_BOUNDARY_WEIGHT,
        "checkpoint_epoch": FINAL_EPOCH,
        "stream": "post_purge",
        "aggregation": (
            "equal_weight_mean_over_4_folds_x_2_training_seeds_x_3_inference_seeds"
        ),
        "n_paired_rows": len(primary_rows),
        "per_fold": per_fold_decisions,
        "pooled_mean_deltas": pooled_means,
        "pooled_checks": pooled_checks,
        "sign_consistency": sign_consistency,
        "source_digest": metadata["source_provenance"]["source_digest"],
    }
    atomic_write_json(analysis_dir / "decision.json", decision)
    decision_word = "GO" if primary_claim else "NO-GO"
    findings = [
        "# GTEA multi-fold boundary-weight confirmation",
        "",
        f"**{decision_word}** under the pre-registered multi-fold rule.",
        "",
        "The primary comparison is weight 1.0 versus paired 0.1 at epoch 10000, "
        "official purge-3 output, pooled equally over four folds, two training seeds, "
        "and three inference seeds.",
        "",
        "The primary claim requires all six pooled checks plus positive Edit and "
        "F1@25 deltas in at least three of four folds.",
        "",
        "## Pooled checks",
        "",
    ]
    findings.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}"
        for name, passed in pooled_checks.items()
    )
    findings.extend(
        [
            "",
            "## Sign consistency",
            "",
            f"- Positive Edit folds: {positive_edit_folds}/4 (need at least 3).",
            f"- Positive F1@25 folds: {positive_f1_folds}/4 (need at least 3).",
            "",
            "## Per-fold gates",
            "",
        ]
    )
    for fold, fold_decision in per_fold_decisions.items():
        word = "PASS" if fold_decision["six_check_gate_passes"] else "FAIL"
        findings.append(f"- Fold {fold}: {word} on all six checks.")
    findings.extend(
        [
            "",
            "Task 2 remains launch-gated by this decision and independent review.",
            "",
        ]
    )
    (analysis_dir / "FINDINGS.md").write_text(
        "\n".join(findings), encoding="utf-8"
    )
    atomic_write_json(
        analysis_dir / "analysis_complete.json",
        {
            "completed": True,
            "completed_utc": completed_utc,
            "decision": decision_word,
            "metric_rows": len(metric_rows),
            "per_case_rows": len(per_case_rows),
            "boundary_match_rows": len(boundary_rows),
            "paired_rows": len(paired_rows),
            "cross_seed_rows": len(cross_seed_rows),
            "source_digest": metadata["source_provenance"]["source_digest"],
        },
    )
    print(f"{decision_word}: multi-fold analysis written to {analysis_dir}", flush=True)


if __name__ == "__main__":
    main()
