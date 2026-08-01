#!/usr/bin/env python3
"""Dependency-light temporal action segmentation metrics."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


OVERLAPS = (0.10, 0.25, 0.50)


def segments(labels: Sequence[int], background_ids: set[int] | None = None) -> list[tuple[int, int, int]]:
    background = background_ids or set()
    values = [int(value) for value in labels]
    if not values:
        return []
    result: list[tuple[int, int, int]] = []
    start = 0
    current = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != current:
            if current not in background:
                result.append((start, index, current))
            start = index
            current = value
    if current not in background:
        result.append((start, len(values), current))
    return result


def edit_score(prediction: Sequence[int], target: Sequence[int], background_ids: set[int] | None = None) -> float:
    predicted = [label for _, _, label in segments(prediction, background_ids)]
    truth = [label for _, _, label in segments(target, background_ids)]
    if not predicted and not truth:
        return 100.0
    rows, cols = len(predicted) + 1, len(truth) + 1
    distance = np.zeros((rows, cols), dtype=np.int64)
    distance[:, 0] = np.arange(rows)
    distance[0, :] = np.arange(cols)
    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if predicted[row - 1] == truth[col - 1] else 1
            distance[row, col] = min(
                distance[row - 1, col] + 1,
                distance[row, col - 1] + 1,
                distance[row - 1, col - 1] + cost,
            )
    return 100.0 * (1.0 - float(distance[-1, -1]) / max(len(predicted), len(truth), 1))


def f_counts(
    prediction: Sequence[int],
    target: Sequence[int],
    overlap: float,
    background_ids: set[int] | None = None,
) -> tuple[int, int, int]:
    predicted = segments(prediction, background_ids)
    truth = segments(target, background_ids)
    hits = np.zeros(len(truth), dtype=bool)
    true_positive = 0
    false_positive = 0
    for pred_start, pred_end, pred_label in predicted:
        if not truth:
            false_positive += 1
            continue
        intersections = np.asarray(
            [max(0, min(pred_end, end) - max(pred_start, start)) for start, end, _ in truth],
            dtype=np.float64,
        )
        unions = np.asarray(
            [max(pred_end, end) - min(pred_start, start) for start, end, _ in truth],
            dtype=np.float64,
        )
        labels = np.asarray([pred_label == label for _, _, label in truth], dtype=np.float64)
        iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0) * labels
        best = int(np.argmax(iou))
        if iou[best] >= overlap and not hits[best]:
            true_positive += 1
            hits[best] = True
        else:
            false_positive += 1
    false_negative = int(len(truth) - hits.sum())
    return true_positive, false_positive, false_negative


def f1_from_counts(counts: Iterable[tuple[int, int, int]]) -> float:
    array = np.asarray(list(counts), dtype=np.int64)
    if array.size == 0:
        return 0.0
    true_positive, false_positive, false_negative = array.sum(axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    return 100.0 * (2 * true_positive / denominator) if denominator else 100.0


def per_video_metrics(
    prediction: Sequence[int], target: Sequence[int], background_ids: set[int] | None = None
) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.int64)
    truth = np.asarray(target, dtype=np.int64)
    if predicted.shape != truth.shape or predicted.ndim != 1:
        raise ValueError(f"Prediction/target mismatch: {predicted.shape} vs {truth.shape}")
    row = {
        "acc": 100.0 * float(np.mean(predicted == truth)),
        "edit": edit_score(predicted, truth, background_ids),
    }
    for overlap in OVERLAPS:
        suffix = int(round(overlap * 100))
        row[f"f1@{suffix}"] = f1_from_counts(
            [f_counts(predicted, truth, overlap, background_ids)]
        )
    return row

def aggregate_standard(
    cases: Sequence[tuple[np.ndarray, np.ndarray]], background_ids: set[int] | None = None
) -> dict[str, float]:
    total_frames = sum(len(target) for _, target in cases)
    correct = sum(int(np.sum(prediction == target)) for prediction, target in cases)
    row = {
        "acc": 100.0 * correct / max(total_frames, 1),
        "edit": float(np.mean([edit_score(prediction, target, background_ids) for prediction, target in cases])),
    }
    for overlap in OVERLAPS:
        suffix = int(round(overlap * 100))
        row[f"f1@{suffix}"] = f1_from_counts(
            [f_counts(prediction, target, overlap, background_ids) for prediction, target in cases]
        )
    return row
