#!/usr/bin/env python3
"""Metrics and mechanistic boundary diagnostics for the GTEA pilot."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from common import (
    BOUNDARY_MATCH_MAX_DISTANCE,
    DATASET,
    SHORT_SEGMENT_MAX_LENGTH,
    WORKSPACE_ROOT,
    load_mapping,
    map_rows,
    read_lines,
)


if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
from src.evaluation import compute_tas_metrics_from_sequences  # noqa: E402


@dataclass(frozen=True)
class Segment:
    label: int
    start: int
    end: int


@dataclass(frozen=True)
class Boundary:
    position: int
    left_label: int
    right_label: int


def segments(labels: np.ndarray, background: int | None = None) -> List[Segment]:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional label sequence, got {values.shape}")
    if not len(values):
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    result = [
        Segment(int(values[start]), int(start), int(end))
        for start, end in zip(starts, ends)
    ]
    if background is not None:
        result = [segment for segment in result if segment.label != background]
    return result


def boundaries(labels: np.ndarray) -> List[Boundary]:
    values = np.asarray(labels, dtype=np.int64)
    positions = np.flatnonzero(values[1:] != values[:-1]) + 1
    return [
        Boundary(int(position), int(values[position - 1]), int(values[position]))
        for position in positions
    ]


def match_boundaries(
    ground_truth: Sequence[Boundary],
    predicted: Sequence[Boundary],
    *,
    max_distance: int,
    transition_aware: bool,
) -> Tuple[List[Tuple[int, int, int]], int, int, int]:
    """Greedy one-to-one matching ordered by absolute temporal distance."""
    candidates: List[Tuple[int, int, int, int]] = []
    for gt_index, gt_boundary in enumerate(ground_truth):
        for pred_index, pred_boundary in enumerate(predicted):
            if transition_aware and (
                gt_boundary.left_label != pred_boundary.left_label
                or gt_boundary.right_label != pred_boundary.right_label
            ):
                continue
            offset = pred_boundary.position - gt_boundary.position
            if abs(offset) <= max_distance:
                candidates.append((abs(offset), gt_index, pred_index, offset))
    candidates.sort()
    used_ground_truth: set[int] = set()
    used_predicted: set[int] = set()
    matches: List[Tuple[int, int, int]] = []
    for _, gt_index, pred_index, offset in candidates:
        if gt_index in used_ground_truth or pred_index in used_predicted:
            continue
        used_ground_truth.add(gt_index)
        used_predicted.add(pred_index)
        matches.append((gt_index, pred_index, offset))
    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(ground_truth) - true_positive
    return matches, true_positive, false_positive, false_negative


def f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 100.0 * 2 * true_positive / denominator if denominator else 100.0


def background_index(mapping_path: Path) -> int | None:
    label_to_index, _ = load_mapping(mapping_path)
    for candidate in ("background", "silence", "sil", "bg"):
        if candidate in label_to_index:
            return int(label_to_index[candidate])
    return None


def ground_truth_indices(data_root: Path, case_id: str, label_to_index: Mapping[str, int]) -> np.ndarray:
    labels = read_lines(data_root / DATASET / "groundTruth" / f"{case_id}.txt")
    missing = sorted(set(labels) - set(label_to_index))
    if missing:
        raise ValueError(f"{case_id}: labels missing from mapping: {missing}")
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int64)


def load_prediction_streams(output_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    streams: Dict[str, Dict[str, np.ndarray]] = {"pre_purge": {}, "post_purge": {}}
    for index, case_id in map_rows(output_dir / "video_index_map.txt"):
        raw = np.load(output_dir / f"{index}_raw.npy")
        official = np.load(output_dir / f"{index}_pred.npy")
        if raw.ndim != 2:
            raise ValueError(f"{case_id}: expected CxT raw probabilities, got {raw.shape}")
        pre = np.argmax(raw, axis=0).astype(np.int64, copy=False)
        post = np.asarray(official, dtype=np.int64)
        if pre.shape != post.shape:
            raise ValueError(f"{case_id}: pre/post prediction length mismatch")
        streams["pre_purge"][case_id] = pre
        streams["post_purge"][case_id] = post
    return streams


def short_false_segments(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    background: int | None,
) -> Tuple[int, int]:
    predicted_segments = segments(prediction, background)
    short_false = 0
    for segment in predicted_segments:
        if segment.end - segment.start > SHORT_SEGMENT_MAX_LENGTH:
            continue
        if not np.any(ground_truth[segment.start : segment.end] == segment.label):
            short_false += 1
    return short_false, len(predicted_segments)


def distribution(values: Sequence[int]) -> Dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "n": 0,
            "mean_signed": None,
            "median_signed": None,
            "mean_absolute": None,
            "median_absolute": None,
            "p90_absolute": None,
        }
    absolute = np.abs(array)
    return {
        "n": int(len(array)),
        "mean_signed": float(array.mean()),
        "median_signed": float(np.median(array)),
        "mean_absolute": float(absolute.mean()),
        "median_absolute": float(np.median(absolute)),
        "p90_absolute": float(np.quantile(absolute, 0.90)),
    }


def evaluate_stream(
    *,
    data_root: Path,
    case_ids: Sequence[str],
    predictions: Mapping[str, np.ndarray],
    stream: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    mapping_path = data_root / DATASET / "mapping.txt"
    label_to_index, _ = load_mapping(mapping_path)
    background = background_index(mapping_path)
    ground_truth_by_case: Dict[str, np.ndarray] = {
        case_id: ground_truth_indices(data_root, case_id, label_to_index) for case_id in case_ids
    }
    for case_id in case_ids:
        if case_id not in predictions:
            raise KeyError(f"Prediction stream misses {case_id}")
        if len(predictions[case_id]) != len(ground_truth_by_case[case_id]):
            raise ValueError(
                f"{case_id}: prediction/GT length mismatch "
                f"{len(predictions[case_id])}/{len(ground_truth_by_case[case_id])}"
            )

    standard = compute_tas_metrics_from_sequences(
        [ground_truth_by_case[case].tolist() for case in case_ids],
        [predictions[case].tolist() for case in case_ids],
        background=background,
    )
    summary: Dict[str, Any] = {
        "stream": stream,
        "acc": float(standard["acc"]),
        "edit": float(standard["edit"]),
        "f1@10": float(standard["f1@10"]),
        "f1@25": float(standard["f1@25"]),
        "f1@50": float(standard["f1@50"]),
    }
    per_case_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    counts = {
        (aware, tolerance): [0, 0, 0]
        for aware in (False, True)
        for tolerance in (5, 10)
    }
    offsets: Dict[bool, List[int]] = {False: [], True: []}
    predicted_segment_total = 0
    ground_truth_segment_total = 0
    short_false_total = 0
    frame_total = 0

    for case_id in case_ids:
        ground_truth = ground_truth_by_case[case_id]
        prediction = np.asarray(predictions[case_id], dtype=np.int64)
        frame_total += len(ground_truth)
        gt_boundaries = boundaries(ground_truth)
        pred_boundaries = boundaries(prediction)
        gt_segments = segments(ground_truth, background)
        pred_segments = segments(prediction, background)
        ground_truth_segment_total += len(gt_segments)
        predicted_segment_total += len(pred_segments)
        short_false, _ = short_false_segments(ground_truth, prediction, background)
        short_false_total += short_false

        case_metrics = compute_tas_metrics_from_sequences(
            [ground_truth.tolist()], [prediction.tolist()], background=background
        )
        case_row: Dict[str, Any] = {
            "case_id": case_id,
            "stream": stream,
            "n_frames": len(ground_truth),
            "acc": float(case_metrics["acc"]),
            "edit": float(case_metrics["edit"]),
            "f1@10": float(case_metrics["f1@10"]),
            "f1@25": float(case_metrics["f1@25"]),
            "f1@50": float(case_metrics["f1@50"]),
            "gt_segments": len(gt_segments),
            "predicted_segments": len(pred_segments),
            "short_false_segments": short_false,
        }
        for aware in (False, True):
            prefix = "transition_aware" if aware else "class_agnostic"
            for tolerance in (5, 10):
                _, true_positive, false_positive, false_negative = match_boundaries(
                    gt_boundaries,
                    pred_boundaries,
                    max_distance=tolerance,
                    transition_aware=aware,
                )
                totals = counts[(aware, tolerance)]
                totals[0] += true_positive
                totals[1] += false_positive
                totals[2] += false_negative
                case_row[f"boundary_f1_{prefix}@{tolerance}"] = f1_from_counts(
                    true_positive, false_positive, false_negative
                )

            matches, _, _, _ = match_boundaries(
                gt_boundaries,
                pred_boundaries,
                max_distance=BOUNDARY_MATCH_MAX_DISTANCE,
                transition_aware=aware,
            )
            for gt_index, pred_index, offset in matches:
                offsets[aware].append(offset)
                gt_boundary = gt_boundaries[gt_index]
                pred_boundary = pred_boundaries[pred_index]
                match_rows.append(
                    {
                        "case_id": case_id,
                        "stream": stream,
                        "matching": prefix,
                        "gt_position": gt_boundary.position,
                        "predicted_position": pred_boundary.position,
                        "signed_offset": offset,
                        "absolute_offset": abs(offset),
                        "gt_left_label": gt_boundary.left_label,
                        "gt_right_label": gt_boundary.right_label,
                        "predicted_left_label": pred_boundary.left_label,
                        "predicted_right_label": pred_boundary.right_label,
                    }
                )
        per_case_rows.append(case_row)

    for aware in (False, True):
        prefix = "transition_aware" if aware else "class_agnostic"
        for tolerance in (5, 10):
            true_positive, false_positive, false_negative = counts[(aware, tolerance)]
            summary[f"boundary_f1_{prefix}@{tolerance}"] = f1_from_counts(
                true_positive, false_positive, false_negative
            )
        for name, value in distribution(offsets[aware]).items():
            summary[f"boundary_offset_{prefix}_{name}"] = value

    summary.update(
        {
            "n_cases": len(case_ids),
            "n_frames": frame_total,
            "gt_segment_count": ground_truth_segment_total,
            "predicted_segment_count": predicted_segment_total,
            "segment_count_ratio": (
                predicted_segment_total / ground_truth_segment_total
                if ground_truth_segment_total
                else None
            ),
            "short_false_segment_count": short_false_total,
            "short_false_segment_rate": (
                short_false_total / predicted_segment_total if predicted_segment_total else 0.0
            ),
        }
    )
    return summary, per_case_rows, match_rows, ground_truth_by_case


def evaluate_export(
    *, data_root: Path, case_ids: Sequence[str], output_dir: Path
) -> Tuple[
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
]:
    streams = load_prediction_streams(output_dir)
    summaries: Dict[str, Dict[str, Any]] = {}
    per_case_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    ground_truth_by_case: Dict[str, np.ndarray] | None = None
    for stream, predictions in streams.items():
        summary, case_rows, boundary_rows, ground_truth = evaluate_stream(
            data_root=data_root,
            case_ids=case_ids,
            predictions=predictions,
            stream=stream,
        )
        summaries[stream] = summary
        per_case_rows.extend(case_rows)
        match_rows.extend(boundary_rows)
        ground_truth_by_case = ground_truth
    assert ground_truth_by_case is not None
    return summaries, per_case_rows, match_rows, streams, ground_truth_by_case


def fixed_broke_ledger(
    ground_truth: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    variant: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    fixed = 0
    broke = 0
    wrong_to_wrong_changed = 0
    unchanged = 0
    total = 0
    for case_id in ground_truth:
        gt = np.asarray(ground_truth[case_id])
        base = np.asarray(baseline[case_id])
        current = np.asarray(variant[case_id])
        if not (gt.shape == base.shape == current.shape):
            raise ValueError(f"{case_id}: ledger input shapes do not agree")
        base_correct = base == gt
        current_correct = current == gt
        fixed += int((~base_correct & current_correct).sum())
        broke += int((base_correct & ~current_correct).sum())
        wrong_to_wrong_changed += int(
            ((~base_correct) & (~current_correct) & (base != current)).sum()
        )
        unchanged += int((base == current).sum())
        total += len(gt)
    return {
        "n_frames": total,
        "fixed_frames": fixed,
        "broken_frames": broke,
        "net_fixed_minus_broken": fixed - broke,
        "wrong_to_wrong_changed_frames": wrong_to_wrong_changed,
        "unchanged_frames": unchanged,
        "fixed_pct": 100.0 * fixed / total if total else 0.0,
        "broken_pct": 100.0 * broke / total if total else 0.0,
        "prediction_disagreement_pct": 100.0 * (total - unchanged) / total if total else 0.0,
    }

