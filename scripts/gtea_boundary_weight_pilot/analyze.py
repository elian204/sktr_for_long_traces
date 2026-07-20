#!/usr/bin/env python3
"""Aggregate the boundary-weight checkpoint grid and apply the locked decision rule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from common import (
    FINAL_EPOCH,
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


STANDARD_AND_DIAGNOSTIC_METRICS = (
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


def sampling_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "task_id",
        "decoder_boundary_loss",
        "role",
        "training_seed",
        "checkpoint_epoch",
        "stream",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in metrics.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_inference_seeds"] = int(len(group))
        for metric in STANDARD_AND_DIAGNOSTIC_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()

    study_dir = args.study_dir.resolve()
    metadata = load_json(study_dir / "study_metadata.json")
    diffact_root = Path(metadata["diffact_root"])
    verify_source_digest(metadata, diffact_root)
    reconciliation_path = study_dir / "baseline_reconciliation.json"
    reconciliation = load_json(reconciliation_path)
    if not reconciliation.get("passed"):
        raise RuntimeError("Baseline reconciliation gate has not passed")

    tasks = load_json(study_dir / "tasks.json")["tasks"]
    for task in tasks:
        if validate_completed_task(task, metadata) is None:
            raise RuntimeError(f"Cannot analyze incomplete task: {task['task_id']}")
    case_ids = read_bundle(Path(tasks[0]["test_manifest"]))
    data_root = Path(metadata["data_root"])

    metric_rows: List[Dict[str, Any]] = []
    per_case_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[str, int, int, str], Dict[str, Any]] = {}
    ground_truth: Dict[str, Any] | None = None
    for task in tasks:
        for epoch in task["checkpoint_epochs"]:
            for inference_seed in task["inference_seeds"]:
                output_dir = export_dir(task, int(epoch), int(inference_seed))
                verify_export(output_dir, case_ids)
                summaries, case_metrics, matches, streams, current_ground_truth = evaluate_export(
                    data_root=data_root,
                    case_ids=case_ids,
                    output_dir=output_dir,
                )
                if ground_truth is None:
                    ground_truth = current_ground_truth
                for stream, summary in summaries.items():
                    prefix = {
                        "task_id": task["task_id"],
                        "decoder_boundary_loss": task["decoder_boundary_loss"],
                        "role": task["role"],
                        "training_seed": task["training_seed"],
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(inference_seed),
                        "stream": stream,
                        "output_dir": str(output_dir),
                    }
                    row = dict(prefix)
                    row.update(summary)
                    metric_rows.append(row)
                    predictions[(task["task_id"], int(epoch), int(inference_seed), stream)] = streams[
                        stream
                    ]
                for row in case_metrics:
                    row.update(
                        {
                            "task_id": task["task_id"],
                            "decoder_boundary_loss": task["decoder_boundary_loss"],
                            "role": task["role"],
                            "training_seed": task["training_seed"],
                            "checkpoint_epoch": int(epoch),
                            "inference_seed": int(inference_seed),
                        }
                    )
                    per_case_rows.append(row)
                for row in matches:
                    row.update(
                        {
                            "task_id": task["task_id"],
                            "decoder_boundary_loss": task["decoder_boundary_loss"],
                            "role": task["role"],
                            "training_seed": task["training_seed"],
                            "checkpoint_epoch": int(epoch),
                            "inference_seed": int(inference_seed),
                        }
                    )
                    boundary_rows.append(row)
    assert ground_truth is not None

    analysis_dir = study_dir / "analysis"
    metrics_frame = write_csv(analysis_dir / "fold_metrics.csv", metric_rows)
    write_csv(analysis_dir / "per_case_metrics.csv", per_case_rows)
    write_csv(analysis_dir / "boundary_matches.csv", boundary_rows)
    summary_frame = sampling_summary(metrics_frame)
    summary_frame.to_csv(analysis_dir / "sampling_seed_summary.csv", index=False)
    summary_frame.to_csv(analysis_dir / "checkpoint_curves.csv", index=False)

    baseline_task = next(task for task in tasks if float(task["decoder_boundary_loss"]) == 0.1)
    paired_rows: List[Dict[str, Any]] = []
    ledger_rows: List[Dict[str, Any]] = []
    metric_index = {
        (
            row["task_id"],
            int(row["checkpoint_epoch"]),
            int(row["inference_seed"]),
            row["stream"],
        ): row
        for row in metric_rows
    }
    for task in tasks:
        if task["task_id"] == baseline_task["task_id"]:
            continue
        for epoch in task["checkpoint_epochs"]:
            for inference_seed in task["inference_seeds"]:
                for stream in ("pre_purge", "post_purge"):
                    base_key = (
                        baseline_task["task_id"],
                        int(epoch),
                        int(inference_seed),
                        stream,
                    )
                    variant_key = (task["task_id"], int(epoch), int(inference_seed), stream)
                    base_row = metric_index[base_key]
                    variant_row = metric_index[variant_key]
                    paired = {
                        "task_id": task["task_id"],
                        "baseline_task_id": baseline_task["task_id"],
                        "decoder_boundary_loss": task["decoder_boundary_loss"],
                        "role": task["role"],
                        "training_seed": task["training_seed"],
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(inference_seed),
                        "stream": stream,
                    }
                    for metric in STANDARD_AND_DIAGNOSTIC_METRICS:
                        paired[f"delta_{metric}"] = (
                            float(variant_row[metric]) - float(base_row[metric])
                            if variant_row[metric] is not None and base_row[metric] is not None
                            else None
                        )
                    ledger = fixed_broke_ledger(
                        ground_truth,
                        predictions[base_key],
                        predictions[variant_key],
                    )
                    paired.update(ledger)
                    paired_rows.append(paired)
                    ledger_rows.append(
                        {
                            key: value
                            for key, value in paired.items()
                            if key.startswith("fixed")
                            or key.startswith("broken")
                            or key.startswith("net_")
                            or key.startswith("wrong_")
                            or key.startswith("unchanged")
                            or key.startswith("prediction_")
                            or key
                            in {
                                "task_id",
                                "baseline_task_id",
                                "decoder_boundary_loss",
                                "role",
                                "training_seed",
                                "checkpoint_epoch",
                                "inference_seed",
                                "stream",
                                "n_frames",
                            }
                        }
                    )
    paired_frame = write_csv(analysis_dir / "paired_deltas.csv", paired_rows)
    write_csv(analysis_dir / "fixed_broke_ledger.csv", ledger_rows)

    primary_rows = paired_frame[
        (paired_frame["decoder_boundary_loss"] == PRIMARY_BOUNDARY_WEIGHT)
        & (paired_frame["checkpoint_epoch"] == FINAL_EPOCH)
        & (paired_frame["stream"] == "post_purge")
    ]
    if len(primary_rows) != len(metadata["inference_seeds"]):
        raise ValueError(
            f"Primary decision expected {len(metadata['inference_seeds'])} seeds; "
            f"found {len(primary_rows)}"
        )
    means = {
        column: float(pd.to_numeric(primary_rows[column], errors="raise").mean())
        for column in primary_rows.columns
        if column.startswith("delta_")
    }
    checks = {
        "boundary_f1_class_agnostic@10_improves": (
            means["delta_boundary_f1_class_agnostic@10"] > 0
        ),
        "edit_improves": means["delta_edit"] > 0,
        "f1@25_improves": means["delta_f1@25"] > 0,
        "accuracy_non_negative": means["delta_acc"] >= 0,
        "segment_count_ratio_not_inflated": means["delta_segment_count_ratio"] <= 0.02,
        "median_absolute_boundary_offset_decreases": (
            means["delta_boundary_offset_class_agnostic_median_absolute"] < 0
        ),
    }
    decision = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "advance_to_class_specific_onset_head": all(checks.values()),
        "pre_registered_primary_boundary_weight": PRIMARY_BOUNDARY_WEIGHT,
        "baseline_boundary_weight": 0.1,
        "checkpoint_epoch": FINAL_EPOCH,
        "stream": "post_purge",
        "aggregation": "mean_over_inference_seeds",
        "n_inference_seeds": len(primary_rows),
        "mean_deltas": means,
        "checks": checks,
        "baseline_reconciliation": str(reconciliation_path),
        "source_digest": metadata["source_provenance"]["source_digest"],
    }
    atomic_write_json(analysis_dir / "decision.json", decision)
    decision_word = "GO" if decision["advance_to_class_specific_onset_head"] else "NO-GO"
    findings = [
        "# GTEA boundary-weight pilot decision",
        "",
        f"**{decision_word}** for advancing this ladder to the class-specific onset head.",
        "",
        "The decision uses the pre-registered weight 0.5 versus the fresh in-harness 0.1 "
        "baseline at epoch 10000, official purge-3 output, averaged over inference seeds.",
        "",
        "## Gate checks",
        "",
    ]
    findings.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in checks.items()
    )
    findings.extend(
        [
            "",
            "This seven-video fold-1 pilot is not a final multi-fold efficacy claim.",
            "See `decision.json`, `paired_deltas.csv`, and `checkpoint_curves.csv` for values.",
            "",
        ]
    )
    (analysis_dir / "FINDINGS.md").write_text("\n".join(findings), encoding="utf-8")
    atomic_write_json(
        analysis_dir / "analysis_complete.json",
        {
            "completed": True,
            "completed_utc": decision["completed_utc"],
            "decision": decision_word,
            "metric_rows": len(metric_rows),
            "per_case_rows": len(per_case_rows),
            "boundary_match_rows": len(boundary_rows),
            "paired_rows": len(paired_rows),
            "source_digest": metadata["source_provenance"]["source_digest"],
        },
    )
    print(f"{decision_word}: analysis written to {analysis_dir}", flush=True)


if __name__ == "__main__":
    main()

