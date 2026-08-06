#!/usr/bin/env python3
"""Pure taxonomy and aggregation helpers for the Phase-C audit."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from metrics import f_counts, f1_from_counts, per_video_metrics
from phase_c_common import ANALYSIS_FPS, PRIMARY_BUCKETS, exact_one_sided_sign_p, holm_adjust


BOUNDARY_WIDTHS = (10, 25, 50)
PRIMARY_BOUNDARY_WIDTH = 25
FRAGMENT_MAX_LEN = 25
LONG_MIN_LEN = 100
LONG_HOMOGENEITY = 0.90


@dataclass(frozen=True)
class DFG:
    starts: frozenset[int]
    ends: frozenset[int]
    edges: frozenset[tuple[int, int]]


def contiguous_spans(mask: Sequence[bool]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, len(mask)))
    return result


def segments(labels: Sequence[int], background: set[int] | None = None) -> list[tuple[int, int, int]]:
    values = [int(value) for value in labels]
    if not values:
        return []
    excluded = background or set()
    result: list[tuple[int, int, int]] = []
    start, current = 0, values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != current:
            if current not in excluded:
                result.append((start, index, current))
            start, current = index, value
    if current not in excluded:
        result.append((start, len(values), current))
    return result


def discover_dfg(traces: Iterable[Sequence[int]], background: set[int] | None = None) -> DFG:
    starts: set[int] = set()
    ends: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for trace in traces:
        labels = [label for _, _, label in segments(trace, background)]
        if not labels:
            continue
        starts.add(labels[0])
        ends.add(labels[-1])
        edges.update(zip(labels[:-1], labels[1:]))
    if not starts or not ends:
        raise RuntimeError("Cannot discover an empty fold-training DFG")
    return DFG(frozenset(starts), frozenset(ends), frozenset(edges))


def boundary_mask(gt: np.ndarray, prediction: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros(len(gt), dtype=bool)
    errors = prediction != gt
    for boundary in np.flatnonzero(gt[1:] != gt[:-1]) + 1:
        left, right = int(gt[boundary - 1]), int(gt[boundary])
        start, end = max(0, int(boundary) - width), min(len(gt), int(boundary) + width + 1)
        local = errors[start:end] & np.isin(prediction[start:end], [left, right])
        result[start:end] |= local
    return result


def fragmentation_mask(gt: np.ndarray, prediction: np.ndarray, max_len: int = FRAGMENT_MAX_LEN) -> np.ndarray:
    result = np.zeros(len(gt), dtype=bool)
    for start, end, label in segments(gt):
        wrong = prediction[start:end] != label
        for relative_start, relative_end in contiguous_spans(wrong):
            span_start, span_end = start + relative_start, start + relative_end
            if span_end - span_start > max_len or span_start <= start or span_end >= end:
                continue
            if prediction[span_start - 1] == label and prediction[span_end] == label:
                result[span_start:span_end] = True
    return result


def illegal_order_mask(
    prediction: np.ndarray, dfg: DFG, background: set[int] | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    predicted_segments = segments(prediction, background)
    result = np.zeros(len(prediction), dtype=bool)
    illegal_internal = 0
    illegal_start = 0
    illegal_end = 0
    for index, (start, end, label) in enumerate(predicted_segments):
        illegal_destination = False
        if index == 0:
            illegal_destination = label not in dfg.starts
            illegal_start += int(illegal_destination)
        else:
            prior = predicted_segments[index - 1][2]
            illegal_destination = (prior, label) not in dfg.edges
            illegal_internal += int(illegal_destination)
        if illegal_destination:
            result[start:end] = True
    if predicted_segments and predicted_segments[-1][2] not in dfg.ends:
        start, end, _ = predicted_segments[-1]
        result[start:end] = True
        illegal_end = 1
    return result, {
        "predicted_segments": len(predicted_segments),
        "predicted_internal_transitions": max(0, len(predicted_segments) - 1),
        "illegal_internal_transitions": illegal_internal,
        "illegal_start_events": illegal_start,
        "illegal_end_events": illegal_end,
    }


def wrong_label_span_mask(gt: np.ndarray, prediction: np.ndarray, background: set[int] | None = None) -> np.ndarray:
    result = np.zeros(len(gt), dtype=bool)
    for start, end, label in segments(prediction, background):
        counts = Counter(int(value) for value in gt[start:end])
        majority, count = counts.most_common(1)[0]
        if count > (end - start) / 2 and majority != label:
            result[start:end] = prediction[start:end] != gt[start:end]
    return result


def long_substitution_mask(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.zeros(len(gt), dtype=bool)
    for start, end in contiguous_spans(prediction != gt):
        if end - start < LONG_MIN_LEN:
            continue
        modal_count = Counter(int(value) for value in prediction[start:end]).most_common(1)[0][1]
        if modal_count / (end - start) >= LONG_HOMOGENEITY:
            result[start:end] = True
    return result


def candidate_ranks(probability: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if probability.ndim != 2 or probability.shape[1] != len(gt):
        raise ValueError("Probability/GT shape mismatch")
    gt_probability = probability[gt, np.arange(len(gt))]
    return 1 + np.sum(probability > gt_probability[None, :], axis=0)


def histogram_quantile(histogram: Mapping[int, int], quantile: float) -> float:
    total = sum(int(count) for count in histogram.values())
    if total <= 0:
        return np.nan
    position = (total - 1) * float(quantile)
    lower, upper = int(np.floor(position)), int(np.ceil(position))

    def value_at(index: int) -> int:
        cumulative = 0
        for value, count in sorted(histogram.items()):
            cumulative += int(count)
            if index < cumulative:
                return int(value)
        raise AssertionError("Histogram quantile index overflow")

    low_value, high_value = value_at(lower), value_at(upper)
    return float(low_value + (position - lower) * (high_value - low_value))


def analyze_case(
    *,
    gt: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    dfg: DFG,
    background: set[int],
) -> dict[str, Any]:
    gt = np.asarray(gt, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    probability = np.asarray(probability)
    if gt.ndim != 1 or prediction.shape != gt.shape or probability.shape[1] != len(gt):
        raise ValueError("Phase-C case alignment mismatch")
    errors = prediction != gt
    boundary_masks = {width: boundary_mask(gt, prediction, width) for width in BOUNDARY_WIDTHS}
    fragmentation = fragmentation_mask(gt, prediction)
    illegal, order_counts = illegal_order_mask(prediction, dfg, background)
    categories = np.full(len(gt), "correct", dtype=object)
    unassigned = errors.copy()
    for name, mask in (
        ("boundary_offset", boundary_masks[PRIMARY_BOUNDARY_WIDTH]),
        ("fragmentation", fragmentation),
        ("illegal_order", illegal),
    ):
        selected = unassigned & mask
        categories[selected] = name
        unassigned[selected] = False
    categories[unassigned] = "legal_substitution"
    counts = {name: int(np.sum(categories == name)) for name in PRIMARY_BUCKETS}
    if sum(counts.values()) != int(errors.sum()):
        raise AssertionError("Exclusive Phase-C taxonomy does not partition all errors")
    gt_labels = set(int(value) for value in gt)
    present_nonadjacent = errors & (categories == "legal_substitution") & np.isin(prediction, list(gt_labels))
    absent_confusion = errors & (categories == "legal_substitution") & ~np.isin(prediction, list(gt_labels))
    if int(present_nonadjacent.sum() + absent_confusion.sum()) != counts["legal_substitution"]:
        raise AssertionError("Legal-substitution subtypes do not partition the legal bucket")
    wrong_label_span = wrong_label_span_mask(gt, prediction, background)
    long_substitution = long_substitution_mask(gt, prediction)
    ranks = candidate_ranks(probability, gt)
    error_ranks = ranks[errors]
    duration_minutes = len(gt) / (ANALYSIS_FPS * 60.0)
    gt_segment_count = len(segments(gt, background))
    error_spans = contiguous_spans(errors)
    row: dict[str, Any] = {
        "n_frames": len(gt),
        "n_errors": int(errors.sum()),
        "n_error_spans": len(error_spans),
        "n_gt_segments": gt_segment_count,
        "duration_minutes": duration_minutes,
        "correct_frames": int((~errors).sum()),
        **per_video_metrics(prediction, gt, background),
        **order_counts,
        "illegal_internal_transition_rate": order_counts["illegal_internal_transitions"] / max(order_counts["predicted_internal_transitions"], 1),
        "over_segmentation_ratio": order_counts["predicted_segments"] / max(gt_segment_count, 1),
        "wrong_label_span_frames": int(wrong_label_span.sum()),
        "long_substitution_frames": int(long_substitution.sum()),
        "legal_present_nonadjacent_frames": int(present_nonadjacent.sum()),
        "legal_absent_confusion_frames": int(absent_confusion.sum()),
        "boundary_w10_frames": int(boundary_masks[10].sum()),
        "boundary_w25_frames": int(boundary_masks[25].sum()),
        "boundary_w50_frames": int(boundary_masks[50].sum()),
        "candidate_rank_observations": int(len(error_ranks)),
        "candidate_gt_top2_frames": int(np.sum(error_ranks <= 2)),
        "candidate_gt_top3_frames": int(np.sum(error_ranks <= 3)),
        "candidate_gt_top5_frames": int(np.sum(error_ranks <= 5)),
        "candidate_gt_rank_median": float(np.median(error_ranks)) if len(error_ranks) else np.nan,
        "candidate_gt_rank_p90": float(np.quantile(error_ranks, 0.90)) if len(error_ranks) else np.nan,
        "candidate_rank_histogram_json": json.dumps(
            {str(int(rank)): int(count) for rank, count in sorted(Counter(int(value) for value in error_ranks).items())},
            sort_keys=True,
        ),
    }
    for overlap in (0.10, 0.25, 0.50):
        suffix = int(round(100 * overlap))
        tp, fp, fn = f_counts(prediction, gt, overlap, background)
        row[f"f1@{suffix}_tp"] = tp
        row[f"f1@{suffix}_fp"] = fp
        row[f"f1@{suffix}_fn"] = fn
    for name in PRIMARY_BUCKETS:
        mask = categories == name
        frame_count = counts[name]
        span_count = len(contiguous_spans(mask))
        row[f"{name}_frames"] = frame_count
        row[f"{name}_spans"] = span_count
        row[f"{name}_share"] = frame_count / int(errors.sum()) if errors.any() else np.nan
        row[f"{name}_frames_per_gt_segment"] = frame_count / max(gt_segment_count, 1)
        row[f"{name}_frames_per_minute"] = frame_count / max(duration_minutes, np.finfo(float).eps)
        row[f"{name}_spans_per_gt_segment"] = span_count / max(gt_segment_count, 1)
        row[f"{name}_spans_per_minute"] = span_count / max(duration_minutes, np.finfo(float).eps)
    row["total_error_frames_per_gt_segment"] = int(errors.sum()) / max(gt_segment_count, 1)
    row["total_error_frames_per_minute"] = int(errors.sum()) / max(duration_minutes, np.finfo(float).eps)
    row["total_error_spans_per_gt_segment"] = len(error_spans) / max(gt_segment_count, 1)
    row["total_error_spans_per_minute"] = len(error_spans) / max(duration_minutes, np.finfo(float).eps)
    return row


def aggregate_cases(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        common = {
            **identity,
            "n_cases": len(group),
            "n_frames": int(group["n_frames"].sum()),
            "n_errors": int(group["n_errors"].sum()),
            "n_gt_segments": int(group["n_gt_segments"].sum()),
            "duration_minutes": float(group["duration_minutes"].sum()),
        }
        weighted = {**common, "aggregation": "frame_weighted"}
        macro = {**common, "aggregation": "per_video_macro"}
        weighted["acc"] = 100.0 * float(group["correct_frames"].sum()) / max(int(group["n_frames"].sum()), 1)
        weighted["edit"] = float(group["edit"].mean())
        for suffix in (10, 25, 50):
            weighted[f"f1@{suffix}"] = f1_from_counts(
                [(int(group[f"f1@{suffix}_tp"].sum()), int(group[f"f1@{suffix}_fp"].sum()), int(group[f"f1@{suffix}_fn"].sum()))]
            )
        for metric in ("acc", "edit", "f1@10", "f1@25", "f1@50"):
            macro[metric] = float(group[metric].mean())
        nonzero = group[group["n_errors"] > 0]
        weighted["error_share_case_count"] = int(len(nonzero))
        macro["error_share_case_count"] = int(len(nonzero))
        for name in PRIMARY_BUCKETS:
            frames = int(group[f"{name}_frames"].sum())
            spans = int(group[f"{name}_spans"].sum())
            weighted[f"{name}_frames"] = frames
            weighted[f"{name}_spans"] = spans
            weighted[f"{name}_share"] = frames / max(int(group["n_errors"].sum()), 1)
            weighted[f"{name}_frames_per_gt_segment"] = frames / max(int(group["n_gt_segments"].sum()), 1)
            weighted[f"{name}_frames_per_minute"] = frames / max(float(group["duration_minutes"].sum()), np.finfo(float).eps)
            weighted[f"{name}_spans_per_gt_segment"] = spans / max(int(group["n_gt_segments"].sum()), 1)
            weighted[f"{name}_spans_per_minute"] = spans / max(float(group["duration_minutes"].sum()), np.finfo(float).eps)
            macro[f"{name}_frames"] = frames
            macro[f"{name}_spans"] = spans
            macro[f"{name}_share"] = float(nonzero[f"{name}_share"].mean()) if len(nonzero) else np.nan
            for suffix in ("frames_per_gt_segment", "frames_per_minute", "spans_per_gt_segment", "spans_per_minute"):
                macro[f"{name}_{suffix}"] = float(group[f"{name}_{suffix}"].mean())
        for target in (weighted, macro):
            target["over_segmentation_ratio"] = (
                float(group["predicted_segments"].sum()) / max(int(group["n_gt_segments"].sum()), 1)
                if target["aggregation"] == "frame_weighted" else float(group["over_segmentation_ratio"].mean())
            )
            target["illegal_internal_transition_rate"] = (
                float(group["illegal_internal_transitions"].sum()) / max(int(group["predicted_internal_transitions"].sum()), 1)
                if target["aggregation"] == "frame_weighted" else float(group["illegal_internal_transition_rate"].mean())
            )
            for column in ("wrong_label_span_frames", "long_substitution_frames", "legal_present_nonadjacent_frames", "legal_absent_confusion_frames", "boundary_w10_frames", "boundary_w25_frames", "boundary_w50_frames"):
                target[column] = int(group[column].sum())
                target[column.replace("_frames", "_share")] = int(group[column].sum()) / max(int(group["n_errors"].sum()), 1)
            observations = int(group["candidate_rank_observations"].sum())
            target["candidate_rank_observations"] = observations
            for k in (2, 3, 5):
                count = int(group[f"candidate_gt_top{k}_frames"].sum())
                target[f"candidate_gt_top{k}_frames"] = count
                target[f"candidate_gt_top{k}_coverage"] = count / max(observations, 1)
            if target["aggregation"] == "frame_weighted":
                histogram: Counter[int] = Counter()
                for value in group["candidate_rank_histogram_json"]:
                    histogram.update({int(rank): int(count) for rank, count in json.loads(value).items()})
                target["candidate_gt_rank_median"] = histogram_quantile(histogram, 0.50)
                target["candidate_gt_rank_p90"] = histogram_quantile(histogram, 0.90)
            else:
                target["candidate_gt_rank_median"] = float(group["candidate_gt_rank_median"].mean())
                target["candidate_gt_rank_p90"] = float(group["candidate_gt_rank_p90"].mean())
        rows.extend([weighted, macro])
    return pd.DataFrame(rows)


def generation_hypothesis_test(per_fold: pd.DataFrame) -> pd.DataFrame:
    selected = per_fold[
        (per_fold["analysis_role"] == "primary") & (per_fold["aggregation"] == "frame_weighted")
    ].copy()
    endpoints = {
        "fragmentation_frames_per_minute": "decrease",
        "illegal_order_frames_per_minute": "decrease",
        "legal_substitution_share": "increase",
    }
    raw_p: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    for endpoint, direction in endpoints.items():
        pivot = selected.pivot(index=["dataset", "fold"], columns="backbone", values=endpoint).dropna()
        delta = pivot["diffact"] - pivot["mstcn2"]
        expected = delta < 0 if direction == "decrease" else delta > 0
        non_tie = delta != 0
        successes, trials = int(expected[non_tie].sum()), int(non_tie.sum())
        p_value = exact_one_sided_sign_p(successes, trials)
        raw_p[endpoint] = p_value
        records.append(
            {
                "endpoint": endpoint,
                "expected_direction": direction,
                "folds": len(pivot),
                "non_tie_folds": trials,
                "expected_direction_folds": successes,
                "diffact_minus_mstcn2_mean_fold_delta": float(delta.mean()),
                "raw_one_sided_sign_p": p_value,
            }
        )
    adjusted = holm_adjust(raw_p)
    for record in records:
        endpoint = record["endpoint"]
        delta = float(record["diffact_minus_mstcn2_mean_fold_delta"])
        expected_delta = delta < 0 if record["expected_direction"] == "decrease" else delta > 0
        record["holm_p"] = adjusted[endpoint]
        record["expected_pooled_direction"] = expected_delta
        record["endpoint_support"] = bool(expected_delta and adjusted[endpoint] <= 0.05)
    all_support = all(bool(record["endpoint_support"]) for record in records)
    for record in records:
        record["overall_pre_registered_support"] = all_support
    return pd.DataFrame(records)
