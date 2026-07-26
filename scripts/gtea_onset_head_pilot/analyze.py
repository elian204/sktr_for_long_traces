#!/usr/bin/env python3
"""Analyze activity metrics and class-specific onset quality for the pilot."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from common import (
    BASELINE_ONSET_LOSS_WEIGHT,
    DATASET,
    FINAL_EPOCH,
    PRIMARY_ONSET_LOSS_WEIGHT,
    WORKSPACE_ROOT,
    atomic_write_json,
    export_dir,
    load_json,
    load_mapping,
    read_bundle,
    read_lines,
    verify_export,
    verify_source_digest,
)
from run_variant import validate_completed_task

METRICS_DIR = WORKSPACE_ROOT / "scripts" / "gtea_boundary_weight_pilot"
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))
from metrics import evaluate_export, fixed_broke_ledger  # noqa: E402


ACTIVITY_METRICS = (
    "acc",
    "edit",
    "f1@10",
    "f1@25",
    "f1@50",
    "boundary_f1_class_agnostic@5",
    "boundary_f1_class_agnostic@10",
    "boundary_offset_class_agnostic_mean_absolute",
    "boundary_offset_class_agnostic_p90_absolute",
    "segment_count_ratio",
    "short_false_segment_rate",
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def gt_starts(labels: np.ndarray) -> List[Tuple[int, int]]:
    positions = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    return [(int(position), int(labels[position])) for position in positions]


def predicted_peaks(curves: np.ndarray) -> List[Tuple[int, int, float]]:
    peaks: List[Tuple[int, int, float]] = []
    for class_id, curve in enumerate(curves):
        indices, properties = find_peaks(curve, height=0.5, distance=3)
        if curve[0] >= 0.5 and (len(curve) == 1 or curve[0] >= curve[1]):
            indices = np.r_[0, indices]
            heights = np.r_[curve[0], properties["peak_heights"]]
        else:
            heights = properties["peak_heights"]
        peaks.extend(
            (int(index), class_id, float(height))
            for index, height in zip(indices, heights)
        )
    return peaks


def one_to_one_peak_counts(
    truth: Sequence[Tuple[int, int]],
    predicted: Sequence[Tuple[int, int, float]],
    tolerance: int,
) -> Tuple[int, int, int]:
    candidates = sorted(
        (
            abs(gt_position - pred_position),
            gt_index,
            pred_index,
        )
        for gt_index, (gt_position, gt_class) in enumerate(truth)
        for pred_index, (pred_position, pred_class, _) in enumerate(predicted)
        if gt_class == pred_class and abs(gt_position - pred_position) <= tolerance
    )
    used_gt, used_pred = set(), set()
    for _, gt_index, pred_index in candidates:
        if gt_index not in used_gt and pred_index not in used_pred:
            used_gt.add(gt_index)
            used_pred.add(pred_index)
    tp = len(used_gt)
    return tp, len(predicted) - tp, len(truth) - tp


def onset_case_metrics(
    labels: np.ndarray, curves: np.ndarray, max_local_distance: int = 50
) -> Dict[str, Any]:
    truth = gt_starts(labels)
    peaks = predicted_peaks(curves)
    offsets = []
    for position, class_id in truth:
        start = max(0, position - max_local_distance)
        end = min(curves.shape[1], position + max_local_distance + 1)
        local_peak = start + int(np.argmax(curves[class_id, start:end]))
        offsets.append(local_peak - position)
    result: Dict[str, Any] = {
        "onset_count": len(truth),
        "predicted_peak_count": len(peaks),
        "onset_peak_offset_mean_signed": float(np.mean(offsets)),
        "onset_peak_offset_mean_absolute": float(np.mean(np.abs(offsets))),
        "onset_peak_offset_median_absolute": float(np.median(np.abs(offsets))),
        "onset_peak_offset_p90_absolute": float(
            np.percentile(np.abs(offsets), 90)
        ),
    }
    for tolerance in (5, 10):
        tp, fp, fn = one_to_one_peak_counts(truth, peaks, tolerance)
        denominator = 2 * tp + fp + fn
        result[f"onset_peak_f1@{tolerance}"] = (
            100.0 * 2 * tp / denominator if denominator else 100.0
        )
        result[f"onset_peak_tp@{tolerance}"] = tp
        result[f"onset_peak_fp@{tolerance}"] = fp
        result[f"onset_peak_fn@{tolerance}"] = fn
    return result


def aggregate_onset(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    onset_count = sum(int(row["onset_count"]) for row in rows)
    result = {
        key: float(
            np.average(
                [float(row[key]) for row in rows],
                weights=[int(row["onset_count"]) for row in rows],
            )
        )
        for key in (
            "onset_peak_offset_mean_signed",
            "onset_peak_offset_mean_absolute",
            "onset_peak_offset_median_absolute",
            "onset_peak_offset_p90_absolute",
        )
    }
    result["onset_count"] = onset_count
    result["predicted_peak_count"] = sum(
        int(row["predicted_peak_count"]) for row in rows
    )
    for tolerance in (5, 10):
        tp = sum(int(row[f"onset_peak_tp@{tolerance}"]) for row in rows)
        fp = sum(int(row[f"onset_peak_fp@{tolerance}"]) for row in rows)
        fn = sum(int(row[f"onset_peak_fn@{tolerance}"]) for row in rows)
        denominator = 2 * tp + fp + fn
        result[f"onset_peak_f1@{tolerance}"] = (
            100.0 * 2 * tp / denominator if denominator else 100.0
        )
    return result


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
            raise RuntimeError(f"Incomplete task: {task['task_id']}")

    cases = read_bundle(Path(tasks[0]["test_manifest"]))
    data_root = Path(metadata["data_root"])
    label_to_index, _ = load_mapping(data_root / DATASET / "mapping.txt")
    labels_by_case = {
        case: np.asarray(
            [
                label_to_index[label]
                for label in read_lines(
                    data_root / DATASET / "groundTruth" / f"{case}.txt"
                )
            ],
            dtype=np.int64,
        )
        for case in cases
    }
    activity_rows: List[Dict[str, Any]] = []
    onset_rows: List[Dict[str, Any]] = []
    onset_case_rows: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[str, int, int, str], Dict[str, np.ndarray]] = {}
    ground_truth = None
    onset_hash_contract: Dict[Tuple[str, int], Dict[str, str]] = {}
    for task in tasks:
        for epoch in task["checkpoint_epochs"]:
            for inference_seed in task["inference_seeds"]:
                output_dir = export_dir(task, int(epoch), int(inference_seed))
                verify_export(output_dir, cases)
                complete = load_json(output_dir / "export_complete.json")
                onset_hashes = complete["onset_artifact_sha256"]
                hash_key = (task["task_id"], int(epoch))
                if hash_key in onset_hash_contract:
                    if onset_hash_contract[hash_key] != onset_hashes:
                        raise ValueError(
                            "Deterministic onset curves changed across inference seeds: "
                            f"{hash_key}"
                        )
                else:
                    onset_hash_contract[hash_key] = onset_hashes
                summaries, _, _, streams, current_gt = evaluate_export(
                    data_root=data_root,
                    case_ids=cases,
                    output_dir=output_dir,
                )
                ground_truth = current_gt if ground_truth is None else ground_truth
                prefix = {
                    "task_id": task["task_id"],
                    "class_specific_onset_loss_weight": task[
                        "class_specific_onset_loss_weight"
                    ],
                    "role": task["role"],
                    "checkpoint_epoch": int(epoch),
                    "inference_seed": int(inference_seed),
                }
                for stream, summary in summaries.items():
                    row = dict(prefix)
                    row["stream"] = stream
                    row.update(summary)
                    activity_rows.append(row)
                    predictions[
                        (task["task_id"], int(epoch), int(inference_seed), stream)
                    ] = streams[stream]
                case_metrics = []
                for index, case in enumerate(cases):
                    curves = np.load(output_dir / f"{index}_onset.npy")
                    row = {
                        **prefix,
                        "case_id": case,
                        **onset_case_metrics(labels_by_case[case], curves),
                    }
                    onset_case_rows.append(row)
                    case_metrics.append(row)
                onset_row = dict(prefix)
                onset_row.update(aggregate_onset(case_metrics))
                onset_rows.append(onset_row)
    assert ground_truth is not None

    analysis_dir = study_dir / "analysis"
    activity_frame = write_csv(
        analysis_dir / "activity_metrics.csv", activity_rows
    )
    onset_frame = write_csv(analysis_dir / "onset_metrics.csv", onset_rows)
    write_csv(analysis_dir / "onset_per_case.csv", onset_case_rows)

    tasks_by_weight = {
        float(task["class_specific_onset_loss_weight"]): task for task in tasks
    }
    activity_index = {
        (
            row["task_id"],
            int(row["checkpoint_epoch"]),
            int(row["inference_seed"]),
            row["stream"],
        ): row
        for row in activity_rows
    }
    onset_index = {
        (
            row["task_id"],
            int(row["checkpoint_epoch"]),
            int(row["inference_seed"]),
        ): row
        for row in onset_rows
    }
    paired_rows: List[Dict[str, Any]] = []
    baseline = tasks_by_weight[BASELINE_ONSET_LOSS_WEIGHT]
    for weight, task in tasks_by_weight.items():
        if weight == BASELINE_ONSET_LOSS_WEIGHT:
            continue
        for epoch in task["checkpoint_epochs"]:
            for seed in task["inference_seeds"]:
                for stream in ("pre_purge", "post_purge"):
                    base_key = (baseline["task_id"], int(epoch), int(seed), stream)
                    task_key = (task["task_id"], int(epoch), int(seed), stream)
                    row = {
                        "task_id": task["task_id"],
                        "baseline_task_id": baseline["task_id"],
                        "class_specific_onset_loss_weight": weight,
                        "checkpoint_epoch": int(epoch),
                        "inference_seed": int(seed),
                        "stream": stream,
                    }
                    for metric in ACTIVITY_METRICS:
                        row[f"delta_{metric}"] = float(
                            activity_index[task_key][metric]
                        ) - float(activity_index[base_key][metric])
                    row.update(
                        fixed_broke_ledger(
                            ground_truth,
                            predictions[base_key],
                            predictions[task_key],
                        )
                    )
                    base_onset = onset_index[
                        (baseline["task_id"], int(epoch), int(seed))
                    ]
                    task_onset = onset_index[
                        (task["task_id"], int(epoch), int(seed))
                    ]
                    for metric in (
                        "onset_peak_offset_mean_absolute",
                        "onset_peak_offset_median_absolute",
                        "onset_peak_offset_p90_absolute",
                        "onset_peak_f1@5",
                        "onset_peak_f1@10",
                    ):
                        row[f"delta_{metric}"] = float(task_onset[metric]) - float(
                            base_onset[metric]
                        )
                    paired_rows.append(row)
    paired_frame = write_csv(analysis_dir / "paired_deltas.csv", paired_rows)

    primary = paired_frame[
        (
            paired_frame["class_specific_onset_loss_weight"]
            == PRIMARY_ONSET_LOSS_WEIGHT
        )
        & (paired_frame["checkpoint_epoch"] == FINAL_EPOCH)
        & (paired_frame["stream"] == "post_purge")
    ]
    if len(primary) != len(metadata["inference_seeds"]):
        raise ValueError(f"Unexpected primary row count: {len(primary)}")
    means = {
        column: float(pd.to_numeric(primary[column], errors="raise").mean())
        for column in primary.columns
        if column.startswith("delta_")
    }
    checks = {
        "accuracy_non_negative": means["delta_acc"] >= 0,
        "edit_improves": means["delta_edit"] > 0,
        "f1@25_improves": means["delta_f1@25"] > 0,
        "segment_count_ratio_not_inflated": (
            means["delta_segment_count_ratio"] <= 0.02
        ),
        "onset_mean_absolute_peak_displacement_decreases": (
            means["delta_onset_peak_offset_mean_absolute"] < 0
        ),
    }
    completed_utc = datetime.now(timezone.utc).isoformat()
    decision = {
        "completed_utc": completed_utc,
        "primary_onset_weight": PRIMARY_ONSET_LOSS_WEIGHT,
        "baseline_onset_weight": BASELINE_ONSET_LOSS_WEIGHT,
        "checkpoint_epoch": FINAL_EPOCH,
        "stream": "post_purge",
        "mean_deltas": means,
        "checks": checks,
        "pilot_gate_passes": all(checks.values()),
        "source_digest": metadata["source_provenance"]["source_digest"],
    }
    atomic_write_json(analysis_dir / "decision.json", decision)
    word = "GO" if all(checks.values()) else "NO-GO"
    findings = [
        "# GTEA class-specific onset-head pilot",
        "",
        f"**{word}** under the staged onset-pilot rule.",
        "",
        "Primary: onset loss 0.3 versus onset-disabled 0.0, both with "
        "decoder boundary loss 1.0, epoch 10000, post-purge.",
        "",
    ]
    findings.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}"
        for name, passed in checks.items()
    )
    (analysis_dir / "FINDINGS.md").write_text(
        "\n".join(findings) + "\n", encoding="utf-8"
    )
    atomic_write_json(
        analysis_dir / "analysis_complete.json",
        {
            "completed": True,
            "completed_utc": completed_utc,
            "decision": word,
            "activity_rows": len(activity_frame),
            "onset_rows": len(onset_frame),
            "paired_rows": len(paired_frame),
            "source_digest": metadata["source_provenance"]["source_digest"],
        },
    )
    print(f"{word}: onset-head analysis written to {analysis_dir}", flush=True)


if __name__ == "__main__":
    main()
