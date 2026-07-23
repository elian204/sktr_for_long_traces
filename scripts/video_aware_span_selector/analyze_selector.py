#!/usr/bin/env python3
"""Leakage-safe four-fold Breakfast base-selector scale-up."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pm4py
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (
    BASE_NUMERIC_FEATURES,
    DATASET,
    DEFAULT_STUDY_DIR,
    FRAME_BUDGETS,
    FORBIDDEN_PRIMARY_FEATURE_PREFIXES,
    INNER_FOLDS,
    OUTER_FOLDS,
    PROTOCOL_VERSION,
    REPAIR_CANDIDATE_FIELDS,
    REPAIR_CANDIDATE_TOP_K,
    SHAPE_FEATURES,
    WORKSPACE_ROOT,
    atomic_write_json,
    file_sha256,
    ground_truth_path,
    load_json,
    normalize_case_id,
    parse_case,
    read_bundle,
    read_lines,
    repair_candidate_columns,
    verify_source_digest,
)


if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation import _edit_score_asformer, _segmental_f1_counts_asformer  # noqa: E402


RANDOM_STATE = 20260717
LONG_MIN_LEN = 100
LONG_HOMOGENEITY = 0.90
MODES = ("official", "raw_argmax")
VARIANTS = ("base", "base_plus_metadata", "base_plus_shape")
DFG_FEATURES = (
    "dfg_start_violation",
    "dfg_incoming_unseen",
    "dfg_outgoing_unseen",
    "dfg_segment_severity",
    "dfg_cumulative_violations",
    "dfg_cumulative_rate",
    "dfg_incoming_surprisal",
    "dfg_outgoing_surprisal",
)
PETRI_FEATURES = (
    "petri_prefix_cost_before",
    "petri_prefix_cost_after",
    "petri_prefix_cost_increment",
    "petri_prefix_cost_rate",
    "petri_prefix_violation",
    "petri_case_final_prefix_cost",
    "petri_case_final_prefix_cost_rate",
)


@dataclass
class RouteStats:
    task: str
    labels: List[str]
    class_frame_counts: Counter[str]
    class_segment_counts: Counter[str]
    pair_counts: Counter[Tuple[str, str]]
    outgoing_counts: Counter[str]
    starts: Counter[str]
    raw_duration: Dict[str, np.ndarray]
    norm_duration: Dict[str, np.ndarray]
    global_raw_duration: np.ndarray
    global_norm_duration: np.ndarray
    total_frames: int
    total_segments: int
    total_cases: int
    traces: List[List[str]]


@dataclass
class PetriRuntime:
    task: str
    places: Tuple[Any, ...]
    transitions: Tuple[Any, ...]
    pre: Dict[Any, Tuple[int, ...]]
    post: Dict[Any, Tuple[int, ...]]
    initial: Tuple[int, ...]
    model_path: Path


@dataclass
class CaseData:
    outer_fold: int
    scope: str
    mode: str
    inner_fold: int | None
    case_id: str
    source_case_id: str
    participant: str
    task: str
    gt: np.ndarray
    pred: np.ndarray
    long_substitution: np.ndarray
    segment_ids: np.ndarray

    @property
    def key(self) -> Tuple[int, str, str, int | None, str]:
        return (self.outer_fold, self.scope, self.mode, self.inner_fold, self.case_id)


def stable_uint64(*parts: Any) -> np.uint64:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return np.uint64(int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little"))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def collapse_segments(labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not labels:
        return []
    result: List[Tuple[int, int, str]] = []
    start = 0
    current = str(labels[0])
    for index, value in enumerate(labels[1:], start=1):
        label = str(value)
        if label != current:
            result.append((start, index, current))
            start = index
            current = label
    result.append((start, len(labels), current))
    return result


def contiguous_spans(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
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


def long_substitution_mask(
    gt: Sequence[str],
    pred: Sequence[str],
    min_len: int = LONG_MIN_LEN,
    homogeneity: float = LONG_HOMOGENEITY,
) -> np.ndarray:
    error = np.asarray([str(left) != str(right) for left, right in zip(gt, pred)], dtype=bool)
    result = np.zeros(len(error), dtype=bool)
    for start, end in contiguous_spans(error):
        if end - start < min_len:
            continue
        most_common = Counter(str(value) for value in pred[start:end]).most_common(1)
        if most_common and most_common[0][1] / (end - start) >= homogeneity:
            result[start:end] = True
    return result


def parse_mapping(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    for line in read_lines(path):
        index, label = line.split(maxsplit=1)
        id_to_name[int(index)] = label
    if not id_to_name or sorted(id_to_name) != list(range(len(id_to_name))):
        raise ValueError(f"Mapping IDs must be contiguous from zero: {path}")
    return id_to_name, {name: index for index, name in id_to_name.items()}


def gt_indices(data_root: Path, case_id: str, name_to_id: Mapping[str, int]) -> np.ndarray:
    labels = read_lines(ground_truth_path(data_root, case_id))
    missing = sorted(set(labels) - set(name_to_id))
    if missing:
        raise ValueError(f"{case_id}: GT labels absent from mapping: {missing[:5]}")
    return np.asarray([name_to_id[label] for label in labels], dtype=np.int32)


def probability_features(probabilities: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(pred):
        raise ValueError(f"Probability shape {probs.shape} does not match T={len(pred)}")
    if not np.isfinite(probs).all() or np.min(probs) < -1e-7:
        raise ValueError("Probabilities contain invalid values")
    column_sums = probs.sum(axis=0)
    if not np.allclose(column_sums, 1.0, atol=2e-4):
        raise ValueError(
            f"Probability columns are not normalized; max error={np.max(abs(column_sums - 1))}"
        )
    top1 = probs.max(axis=0)
    top2 = (
        np.partition(probs, -2, axis=0)[-2]
        if probs.shape[0] > 1
        else np.zeros(len(pred), dtype=float)
    )
    pred_indices = np.asarray(pred, dtype=int)
    pred_probability = probs[pred_indices, np.arange(len(pred))]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=0)
    entropy /= max(math.log(probs.shape[0]), 1.0)
    return {
        "pmax": top1,
        "margin": top1 - top2,
        "pred_probability": pred_probability,
        "entropy": entropy,
        "override_gap": top1 - pred_probability,
    }


def segment_candidate_pool(
    probabilities: np.ndarray,
    start: int,
    end: int,
    id_to_name: Mapping[int, str],
    top_k: int = REPAIR_CANDIDATE_TOP_K,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Rank labels by their mean raw probability within one predicted span."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or not (0 <= start < end <= probs.shape[1]):
        raise ValueError(
            f"Invalid segment bounds [{start}, {end}) for probability shape {probs.shape}"
        )
    if top_k <= 0 or top_k > probs.shape[0]:
        raise ValueError(f"top_k must be in 1..{probs.shape[0]}, got {top_k}")
    mean_probability = probs[:, start:end].mean(axis=1)
    class_ids = np.arange(probs.shape[0], dtype=np.int32)
    # Descending probability with class ID as a deterministic tie breaker.
    order = np.lexsort((class_ids, -mean_probability))
    row: Dict[str, Any] = {}
    for rank, class_id in enumerate(order[:top_k], start=1):
        class_id = int(class_id)
        row[f"candidate_rank_{rank}_class_id"] = class_id
        row[f"candidate_rank_{rank}_label"] = id_to_name[class_id]
        row[f"candidate_rank_{rank}_mean_probability"] = float(
            mean_probability[class_id]
        )
    return row, order


def segment_probability_shape_features(
    probabilities: np.ndarray,
    raw_argmax: np.ndarray,
    start: int,
    end: int,
) -> Dict[str, float]:
    """Compute the exploratory probability-shape features from raw outputs."""
    probs = np.asarray(probabilities, dtype=float)
    raw = np.asarray(raw_argmax, dtype=np.int32)
    if probs.ndim != 2 or raw.shape != (probs.shape[1],):
        raise ValueError(
            f"Raw stream/probability mismatch: probs={probs.shape}, argmax={raw.shape}"
        )
    if not (0 <= start < end <= probs.shape[1]):
        raise ValueError(
            f"Invalid segment bounds [{start}, {end}) for probability shape {probs.shape}"
        )
    span = probs[:, start:end]
    length = end - start
    pmax = span.max(axis=0)
    top2 = (
        np.partition(span, -2, axis=0)[-2]
        if span.shape[0] > 1
        else np.zeros(length, dtype=float)
    )
    margin = pmax - top2
    if length >= 2:
        position = np.linspace(0.0, 1.0, length)
        centered = position - float(np.mean(position))
        confidence_slope = float(
            np.sum(centered * (pmax - float(np.mean(pmax))))
            / np.sum(centered**2)
        )
        flicker_rate = float(np.mean(raw[start + 1 : end] != raw[start : end - 1]))
    else:
        confidence_slope = 0.0
        flicker_rate = 0.0

    edge_width = min(max(1, int(round(0.10 * length))), (length - 1) // 2)
    if edge_width >= 1:
        edge = np.concatenate([margin[:edge_width], margin[-edge_width:]])
        core = margin[edge_width:-edge_width]
        edge_vs_core_margin = float(np.mean(edge) - np.mean(core))
    else:
        edge_vs_core_margin = 0.0

    mean_probability = span.mean(axis=1)
    class_ids = np.arange(span.shape[0], dtype=np.int32)
    segment_order = np.lexsort((class_ids, -mean_probability))
    segment_runner_up = int(segment_order[1]) if len(segment_order) > 1 else int(
        segment_order[0]
    )
    if span.shape[0] > 1:
        frame_class_ids = np.broadcast_to(class_ids[:, None], span.shape)
        frame_order = np.lexsort((frame_class_ids, -span), axis=0)
        frame_runner_up = frame_order[1]
        runner_up_consistency = float(np.mean(frame_runner_up == segment_runner_up))
    else:
        runner_up_consistency = 1.0
    return {
        "confidence_slope": confidence_slope,
        "edge_vs_core_margin": edge_vs_core_margin,
        "flicker_rate": flicker_rate,
        # This is intentionally an explicit alias of the base margin_mean
        # definition requested in the exploratory feature specification.
        "runner_up_gap": float(np.mean(margin)),
        "runner_up_consistency": runner_up_consistency,
    }


def robust_location_scale(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0, 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if mad < 1e-6:
        mad = float(np.std(values))
    return median, max(mad, 0.10)


def duration_abs_z(
    by_class: Mapping[str, np.ndarray],
    global_values: np.ndarray,
    label: str,
    value: float,
) -> float:
    class_values = by_class.get(label, np.asarray([], dtype=float))
    reference = class_values if len(class_values) >= 5 else global_values
    median, scale = robust_location_scale(np.log1p(reference))
    return float(abs((math.log1p(max(float(value), 0.0)) - median) / scale))


def class_rarity(count: int, total: int, n_labels: int) -> float:
    return float(-math.log((count + 1.0) / (total + n_labels)))


def read_route_stats(
    data_root: Path,
    cases: Sequence[str],
    name_to_id: Mapping[str, int],
) -> Dict[str, RouteStats]:
    grouped: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for case in cases:
        labels = gt_indices(data_root, case, name_to_id).astype(str)
        task = parse_case(case, len(labels)).task
        grouped[task].append((case, labels))
    stats_by_task: Dict[str, RouteStats] = {}
    all_labels = [str(index) for index in sorted(name_to_id.values())]
    for task, task_cases in sorted(grouped.items()):
        frame_counts: Counter[str] = Counter()
        segment_counts: Counter[str] = Counter()
        pair_counts: Counter[Tuple[str, str]] = Counter()
        outgoing_counts: Counter[str] = Counter()
        starts: Counter[str] = Counter()
        raw_duration: Dict[str, List[float]] = defaultdict(list)
        norm_duration: Dict[str, List[float]] = defaultdict(list)
        traces: List[List[str]] = []
        total_frames = 0
        total_segments = 0
        for _, labels_array in task_cases:
            labels = labels_array.tolist()
            segments = collapse_segments(labels)
            trace = [label for _, _, label in segments]
            traces.append(trace)
            total_frames += len(labels)
            total_segments += len(segments)
            frame_counts.update(labels)
            segment_counts.update(trace)
            if trace:
                starts[trace[0]] += 1
            for left, right in zip(trace, trace[1:]):
                pair_counts[(left, right)] += 1
                outgoing_counts[left] += 1
            for start, end, label in segments:
                length = end - start
                raw_duration[label].append(float(length))
                norm_duration[label].append(float(length / len(labels)))
        raw_np = {label: np.asarray(values) for label, values in raw_duration.items()}
        norm_np = {label: np.asarray(values) for label, values in norm_duration.items()}
        stats_by_task[task] = RouteStats(
            task=task,
            labels=all_labels,
            class_frame_counts=frame_counts,
            class_segment_counts=segment_counts,
            pair_counts=pair_counts,
            outgoing_counts=outgoing_counts,
            starts=starts,
            raw_duration=raw_np,
            norm_duration=norm_np,
            global_raw_duration=(
                np.concatenate(list(raw_np.values())) if raw_np else np.asarray([1.0])
            ),
            global_norm_duration=(
                np.concatenate(list(norm_np.values())) if norm_np else np.asarray([1.0])
            ),
            total_frames=total_frames,
            total_segments=total_segments,
            total_cases=len(task_cases),
            traces=traces,
        )
    return stats_by_task


def dfg_features(trace: Sequence[str], stats: RouteStats) -> Dict[str, np.ndarray]:
    count = len(trace)
    start_violation = np.zeros(count, dtype=float)
    incoming_unseen = np.zeros(count, dtype=float)
    outgoing_unseen = np.zeros(count, dtype=float)
    cumulative = np.zeros(count, dtype=float)
    incoming_surprisal = np.zeros(count, dtype=float)
    outgoing_surprisal = np.zeros(count, dtype=float)
    if not count:
        return {
            "dfg_start_violation": start_violation,
            "dfg_incoming_unseen": incoming_unseen,
            "dfg_outgoing_unseen": outgoing_unseen,
            "dfg_segment_severity": incoming_unseen,
            "dfg_cumulative_violations": cumulative,
            "dfg_cumulative_rate": cumulative,
            "dfg_incoming_surprisal": incoming_surprisal,
            "dfg_outgoing_surprisal": outgoing_surprisal,
        }
    alpha = 0.5
    n_labels = max(len(stats.labels), 1)
    start_violation[0] = float(stats.starts[str(trace[0])] == 0)
    running = start_violation[0]
    cumulative[0] = running
    for index, (left, right) in enumerate(zip(trace, trace[1:])):
        unseen = float(stats.pair_counts[(str(left), str(right))] == 0)
        outgoing_unseen[index] = unseen
        incoming_unseen[index + 1] = unseen
        probability = (stats.pair_counts[(str(left), str(right))] + alpha) / (
            stats.outgoing_counts[str(left)] + alpha * n_labels
        )
        surprisal = -math.log(probability)
        outgoing_surprisal[index] = surprisal
        incoming_surprisal[index + 1] = surprisal
        running += unseen
        cumulative[index + 1] = running
    severity = np.maximum(start_violation, np.maximum(incoming_unseen, outgoing_unseen))
    return {
        "dfg_start_violation": start_violation,
        "dfg_incoming_unseen": incoming_unseen,
        "dfg_outgoing_unseen": outgoing_unseen,
        "dfg_segment_severity": severity,
        "dfg_cumulative_violations": cumulative,
        "dfg_cumulative_rate": cumulative / np.arange(1, count + 1),
        "dfg_incoming_surprisal": incoming_surprisal,
        "dfg_outgoing_surprisal": outgoing_surprisal,
    }


def build_route_petri(
    task: str,
    traces: Sequence[Sequence[str]],
    model_path: Path,
) -> PetriRuntime:
    rows: List[Dict[str, Any]] = []
    origin = pd.Timestamp("2000-01-01")
    for case_index, trace in enumerate(traces):
        for event_index, label in enumerate(trace):
            rows.append(
                {
                    "case:concept:name": f"train_{case_index}",
                    "concept:name": str(label),
                    "time:timestamp": origin + pd.Timedelta(seconds=event_index),
                }
            )
    if not rows:
        raise ValueError(f"Cannot discover route Petri net without training traces: {task}")
    tree = pm4py.discover_process_tree_inductive(pd.DataFrame(rows))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_pnml(net, initial_marking, final_marking, str(model_path))
    places = tuple(sorted(net.places, key=lambda place: str(place.name)))
    transitions = tuple(sorted(net.transitions, key=lambda transition: str(transition.name)))
    place_index = {place: index for index, place in enumerate(places)}
    pre: Dict[Any, Tuple[int, ...]] = {}
    post: Dict[Any, Tuple[int, ...]] = {}
    for transition in transitions:
        pre_values = [0] * len(places)
        post_values = [0] * len(places)
        for arc in transition.in_arcs:
            pre_values[place_index[arc.source]] += int(arc.weight)
        for arc in transition.out_arcs:
            post_values[place_index[arc.target]] += int(arc.weight)
        pre[transition] = tuple(pre_values)
        post[transition] = tuple(post_values)
    initial = tuple(int(initial_marking.get(place, 0)) for place in places)
    return PetriRuntime(task, places, transitions, pre, post, initial, model_path)


def exact_prefix_costs(
    trace: Sequence[str], runtime: PetriRuntime
) -> Tuple[np.ndarray, int]:
    """Minimum unit-cost log/model moves to consume every trace prefix.

    A completed prefix accepts any reachable marking. Invisible model moves and
    synchronous moves cost zero; visible model moves and log moves cost one.
    This matches prefix-consumable decoding semantics rather than final-marking
    conformance.
    """
    labels = tuple(str(label) for label in trace)
    prefix_cost = np.full(len(labels) + 1, np.inf, dtype=float)
    start = (0, runtime.initial)
    distances: Dict[Tuple[int, Tuple[int, ...]], int] = {start: 0}
    queue: List[Tuple[int, int, int, Tuple[int, ...]]] = [(0, 0, 0, runtime.initial)]
    serial = 1
    popped = 0
    while queue:
        cost, _, position, marking = heapq.heappop(queue)
        state = (position, marking)
        if cost != distances.get(state):
            continue
        popped += 1
        if not np.isfinite(prefix_cost[position]):
            prefix_cost[position] = float(cost)
            if np.isfinite(prefix_cost).all():
                break

        def relax(next_position: int, next_marking: Tuple[int, ...], added: int) -> None:
            nonlocal serial
            next_cost = cost + added
            # Consuming everything as log moves is always an upper bound.
            if next_cost > len(labels):
                return
            next_state = (next_position, next_marking)
            if next_cost < distances.get(next_state, len(labels) + 1):
                distances[next_state] = next_cost
                heapq.heappush(
                    queue, (next_cost, serial, next_position, next_marking)
                )
                serial += 1

        if position < len(labels):
            relax(position + 1, marking, 1)
        for transition in runtime.transitions:
            pre = runtime.pre[transition]
            if any(marking[index] < needed for index, needed in enumerate(pre)):
                continue
            post = runtime.post[transition]
            fired = tuple(
                marking[index] - pre[index] + post[index]
                for index in range(len(marking))
            )
            label = transition.label
            if label is None:
                relax(position, fired, 0)
            else:
                relax(position, fired, 1)
                if position < len(labels) and str(label) == labels[position]:
                    relax(position + 1, fired, 0)
    if not np.isfinite(prefix_cost).all():
        raise RuntimeError(
            f"Exact prefix search failed for task={runtime.task}, trace_length={len(labels)}"
        )
    if np.any(np.diff(prefix_cost) < -1e-9):
        raise AssertionError(f"Prefix costs must be monotone: {prefix_cost.tolist()}")
    return prefix_cost, popped


def categorical_feature_columns(
    labels: Sequence[str], tasks: Sequence[str], cameras: Sequence[str]
) -> List[str]:
    del labels
    return [
        *(f"task__{task}" for task in tasks),
        *(f"camera__{camera}" for camera in cameras),
    ]


def variant_features(categorical: Sequence[str]) -> Dict[str, List[str]]:
    metadata = [
        column
        for column in categorical
        if column.startswith(FORBIDDEN_PRIMARY_FEATURE_PREFIXES)
    ]
    return {
        "base": list(BASE_NUMERIC_FEATURES),
        "base_plus_metadata": [*BASE_NUMERIC_FEATURES, *metadata],
        "base_plus_shape": [*BASE_NUMERIC_FEATURES, *SHAPE_FEATURES],
    }


def read_export_map(export_dir: Path) -> Dict[str, int]:
    rows: Dict[str, int] = {}
    for line in read_lines(export_dir / "video_index_map.txt"):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed video_index_map row: {line!r}")
        index, case = int(fields[0]), normalize_case_id(fields[1])
        if case in rows:
            raise ValueError(f"Duplicate case in export map: {case}")
        rows[case] = index
    return rows


def build_feature_contexts(
    study_dir: Path,
    metadata: Mapping[str, Any],
    data_root: Path,
    name_to_id: Mapping[str, int],
    out_dir: Path,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    contexts: Dict[str, Dict[str, Any]] = {}
    audit_rows: List[Dict[str, Any]] = []
    context_manifests = {
        f"inner_{fold}": Path(metadata["inner_folds"][str(fold)]["train_manifest"])
        for fold in range(1, INNER_FOLDS + 1)
    }
    context_manifests["outer_train"] = Path(metadata["outer_train_manifest"])
    for context_id, manifest in context_manifests.items():
        cases = read_bundle(manifest)
        case_set = set(cases)
        participants = {
            parse_case(case, len(read_lines(ground_truth_path(data_root, case)))).participant
            for case in cases
        }
        stats_by_task = read_route_stats(data_root, cases, name_to_id)
        for task, stats in stats_by_task.items():
            audit_rows.append(
                {
                    "context_id": context_id,
                    "feature_train_manifest": str(manifest),
                    "feature_train_manifest_sha256": file_sha256(manifest),
                    "task": task,
                    "train_cases": stats.total_cases,
                    "train_frames": stats.total_frames,
                    "train_collapsed_segments": stats.total_segments,
                    "uses_process_features": False,
                }
            )
        contexts[context_id] = {
            "manifest": manifest,
            "manifest_sha256": file_sha256(manifest),
            "cases": case_set,
            "participants": participants,
            "stats": stats_by_task,
        }
    return contexts, audit_rows


def build_segment_rows(
    study_dir: Path,
    metadata: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    modes: Sequence[str],
    data_root: Path,
    id_to_name: Mapping[int, str],
    name_to_id: Mapping[str, int],
    contexts: Mapping[str, Dict[str, Any]],
    *,
    outer_fold: int,
    segment_id_start: int,
) -> Tuple[List[CaseData], pd.DataFrame, pd.DataFrame, List[str]]:
    labels = [str(index) for index in sorted(id_to_name)]
    outer_train = read_bundle(Path(metadata["outer_train_manifest"]))
    outer_test = read_bundle(Path(metadata["outer_test_manifest"]))
    all_known_cases = [*outer_train, *outer_test]
    parsed_known = [
        parse_case(case, len(read_lines(ground_truth_path(data_root, case))))
        for case in all_known_cases
    ]
    tasks_seen = sorted({info.task for info in parsed_known})
    cameras_seen = sorted({info.camera for info in parsed_known})
    categorical = categorical_feature_columns(labels, tasks_seen, cameras_seen)

    sources: List[Dict[str, Any]] = []
    for task in tasks:
        complete_path = (
            Path(task["import_manifest_path"])
            if task.get("execution_mode") == "imported"
            else Path(task["run_dir"]) / "task_complete.json"
        )
        if not complete_path.is_file():
            raise FileNotFoundError(
                f"OOF task is not complete: {task['task_id']} ({complete_path})"
            )
        complete = load_json(complete_path)
        if task.get("execution_mode") != "imported" and complete.get("status") != "complete":
            raise ValueError(f"OOF task did not complete successfully: {task['task_id']}")
        sources.append(
            {
                "scope": "oof_validation",
                "inner_fold": int(task["inner_fold"]),
                "cases": read_bundle(Path(task["heldout_manifest"])),
                "export_dir": Path(task["softmax_output_dir"]),
                "context_id": f"inner_{int(task['inner_fold'])}",
            }
        )
    sources.append(
        {
            "scope": "outer_test",
            "inner_fold": None,
            "cases": outer_test,
            "export_dir": Path(metadata["outer_release_export"]["path"]),
            "context_id": "outer_train",
        }
    )

    cases_out: List[CaseData] = []
    segment_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    next_segment_id = int(segment_id_start)
    for source in sources:
        export_dir = source["export_dir"]
        export_map = read_export_map(export_dir)
        expected_cases = source["cases"]
        missing = sorted(set(expected_cases) - set(export_map))
        if missing:
            raise ValueError(
                f"Export {export_dir} misses {len(missing)} required cases: {missing[:5]}"
            )
        context = contexts[source["context_id"]]
        feature_train_cases = context["cases"]
        feature_train_people = context["participants"]
        for case_id in expected_cases:
            source_index = export_map[case_id]
            raw_prob = np.load(export_dir / f"{source_index}_raw.npy")
            official_pred = np.load(export_dir / f"{source_index}_pred.npy").astype(np.int32)
            raw_pred = raw_prob.argmax(axis=0).astype(np.int32)
            gt = gt_indices(data_root, case_id, name_to_id)
            if raw_prob.shape != (len(id_to_name), len(gt)):
                raise ValueError(
                    f"{case_id}: raw probability shape {raw_prob.shape}, "
                    f"expected {(len(id_to_name), len(gt))}"
                )
            if official_pred.shape != gt.shape:
                raise ValueError(f"{case_id}: official pred shape {official_pred.shape} vs GT {gt.shape}")
            info = parse_case(case_id, len(gt))
            if case_id in feature_train_cases:
                raise AssertionError(
                    f"Feature leakage: eval case {case_id} occurs in {source['context_id']} training"
                )
            if info.participant in feature_train_people:
                raise AssertionError(
                    f"Subject leakage: eval participant {info.participant} occurs in "
                    f"{source['context_id']} feature training"
                )
            if info.task not in context["stats"]:
                raise ValueError(f"No train-only route statistics for {case_id} task={info.task}")
            stats: RouteStats = context["stats"][info.task]

            for mode in modes:
                prediction_int = official_pred if mode == "official" else raw_pred
                pred = prediction_int.astype(str)
                gt_text = gt.astype(str)
                probability = probability_features(raw_prob, prediction_int)
                segments = collapse_segments(pred.tolist())
                trace = [label for _, _, label in segments]
                error = pred != gt_text
                long_sub = long_substitution_mask(gt_text, pred)
                segment_ids = np.empty(len(pred), dtype=np.int64)
                n_labels = len(labels)

                for segment_index, (start, end, label) in enumerate(segments):
                    length = end - start
                    left_label = trace[segment_index - 1] if segment_index else "START"
                    right_label = (
                        trace[segment_index + 1]
                        if segment_index + 1 < len(trace)
                        else "END"
                    )
                    raw_z = duration_abs_z(
                        stats.raw_duration, stats.global_raw_duration, label, float(length)
                    )
                    norm_z = duration_abs_z(
                        stats.norm_duration,
                        stats.global_norm_duration,
                        label,
                        float(length / len(pred)),
                    )
                    neighbor_rarities = [
                        class_rarity(
                            stats.class_segment_counts[neighbor], stats.total_segments, n_labels
                        )
                        for neighbor in (left_label, right_label)
                        if neighbor not in {"START", "END"}
                    ]
                    uncertainty = 1.0 - probability["pred_probability"][start:end]
                    gt_segment = gt_text[start:end]
                    gt_counts = Counter(str(value) for value in gt_segment)
                    correct_label, correct_count = gt_counts.most_common(1)[0]
                    candidate_columns, candidate_order = segment_candidate_pool(
                        raw_prob,
                        start,
                        end,
                        id_to_name,
                        top_k=REPAIR_CANDIDATE_TOP_K,
                    )
                    correct_label_rank = int(
                        np.flatnonzero(candidate_order == int(correct_label))[0] + 1
                    )
                    shape_features = segment_probability_shape_features(
                        raw_prob, raw_pred, start, end
                    )
                    row: Dict[str, Any] = {
                        "segment_id": next_segment_id,
                        "outer_fold": outer_fold,
                        "scope": source["scope"],
                        "mode": mode,
                        "inner_fold": source["inner_fold"],
                        "case_id": case_id,
                        "source_case_id": source_index,
                        "participant": info.participant,
                        "task": info.task,
                        "camera": info.camera,
                        "segment_index": segment_index,
                        "start": start,
                        "end": end,
                        "length": length,
                        "predicted_label": label,
                        "predicted_label_name": id_to_name[int(label)],
                        "left_label": left_label,
                        "right_label": right_label,
                        "error_frames": int(np.sum(error[start:end])),
                        "error_fraction": float(np.mean(error[start:end])),
                        "correct_label": correct_label,
                        "correct_label_name": id_to_name[int(correct_label)],
                        "correct_label_fraction": float(correct_count / length),
                        "correct_label_probability_rank": correct_label_rank,
                        "correct_label_in_top5": (
                            correct_label_rank <= REPAIR_CANDIDATE_TOP_K
                        ),
                        "is_fully_correct": bool(np.all(~error[start:end])),
                        "long_substitution_frames": int(np.sum(long_sub[start:end])),
                        "long_substitution_fraction": float(np.mean(long_sub[start:end])),
                        "feature_context_id": source["context_id"],
                        "feature_train_manifest": str(context["manifest"]),
                        "feature_train_manifest_sha256": context["manifest_sha256"],
                        "feature_train_case_count": len(feature_train_cases),
                        "eval_case_in_feature_train": False,
                        "eval_participant_in_feature_train": False,
                        "uncertainty_mean": float(np.mean(uncertainty)),
                        "uncertainty_q90": float(np.quantile(uncertainty, 0.90)),
                        "entropy_mean": float(np.mean(probability["entropy"][start:end])),
                        "top1_uncertainty_mean": float(
                            np.mean(1.0 - probability["pmax"][start:end])
                        ),
                        "margin_mean": float(np.mean(probability["margin"][start:end])),
                        "pred_probability_mean": float(
                            np.mean(probability["pred_probability"][start:end])
                        ),
                        "official_override_gap_mean": float(
                            np.mean(probability["override_gap"][start:end])
                        ),
                        "duration_raw_abs_z": raw_z,
                        "duration_norm_abs_z": norm_z,
                        "segment_log_length": float(math.log1p(length)),
                        "segment_fraction": float(length / len(pred)),
                        "video_progress_mid": float((start + end) / (2 * len(pred))),
                        "class_frame_rarity": class_rarity(
                            stats.class_frame_counts[label], stats.total_frames, n_labels
                        ),
                        "class_segment_rarity": class_rarity(
                            stats.class_segment_counts[label], stats.total_segments, n_labels
                        ),
                        "neighbor_class_rarity": float(
                            np.mean(neighbor_rarities) if neighbor_rarities else 0.0
                        ),
                        **shape_features,
                        **candidate_columns,
                    }
                    for column in categorical:
                        row[column] = 0.0
                    row[f"task__{info.task}"] = 1.0
                    row[f"camera__{info.camera}"] = 1.0
                    segment_rows.append(row)
                    segment_ids[start:end] = next_segment_id
                    next_segment_id += 1

                cases_out.append(
                    CaseData(
                        outer_fold=outer_fold,
                        scope=source["scope"],
                        mode=mode,
                        inner_fold=source["inner_fold"],
                        case_id=case_id,
                        source_case_id=str(source_index),
                        participant=info.participant,
                        task=info.task,
                        gt=gt_text.copy(),
                        pred=pred.copy(),
                        long_substitution=long_sub,
                        segment_ids=segment_ids,
                    )
                )
                case_rows.append(
                    {
                        "scope": source["scope"],
                        "outer_fold": outer_fold,
                        "mode": mode,
                        "inner_fold": source["inner_fold"],
                        "case_id": case_id,
                        "source_case_id": source_index,
                        "participant": info.participant,
                        "task": info.task,
                        "camera": info.camera,
                        "n_frames": len(gt),
                        "n_segments": len(segments),
                        "error_frames": int(np.sum(error)),
                        "long_substitution_frames": int(np.sum(long_sub)),
                        "feature_context_id": source["context_id"],
                        "feature_train_manifest_sha256": context["manifest_sha256"],
                        "official_raw_changed_frames": int(np.sum(official_pred != raw_pred)),
                    }
                )
    segments_df = pd.DataFrame(segment_rows)
    cases_df = pd.DataFrame(case_rows)
    if segments_df.empty or cases_df.empty:
        raise ValueError("No selector rows were constructed")
    if segments_df["eval_case_in_feature_train"].any() or segments_df[
        "eval_participant_in_feature_train"
    ].any():
        raise AssertionError("Feature leakage flags must be uniformly false")
    return cases_out, segments_df, cases_df, categorical


def selector_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.nan_to_num(
        frame.loc[:, columns].to_numpy(dtype=float),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )


def fit_selector_scores(
    segments: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit independently inside each outer fold, then score its untouched test."""
    scored = segments.copy()
    audits: List[Dict[str, Any]] = []
    model_dir = out_dir / "selector_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for outer_fold in OUTER_FOLDS:
        fold_mask = scored["outer_fold"] == outer_fold
        for mode in sorted(scored["mode"].unique()):
            oof_mask = (
                fold_mask
                & (scored["scope"] == "oof_validation")
                & (scored["mode"] == mode)
            )
            outer_mask = (
                fold_mask
                & (scored["scope"] == "outer_test")
                & (scored["mode"] == mode)
            )
            if not oof_mask.any() or not outer_mask.any():
                raise ValueError(f"Missing rows for outer={outer_fold}, mode={mode}")
            for variant, features in feature_sets.items():
                score_column = f"{variant}_score"
                scored.loc[oof_mask | outer_mask, score_column] = np.nan
                for held_fold in range(1, INNER_FOLDS + 1):
                    train_mask = oof_mask & (scored["inner_fold"] != held_fold)
                    eval_mask = oof_mask & (scored["inner_fold"] == held_fold)
                    train = scored.loc[train_mask]
                    evaluation = scored.loc[eval_mask]
                    train_people = set(train["participant"])
                    eval_people = set(evaluation["participant"])
                    if train_people.intersection(eval_people):
                        raise AssertionError(
                            f"Selector leakage outer={outer_fold}, held={held_fold}"
                        )
                    model = selector_model()
                    targets = train["error_fraction"].to_numpy(dtype=float)
                    weights = train["length"].to_numpy(dtype=float)
                    model.fit(matrix(train, features), targets, sample_weight=weights)
                    predictions = np.clip(
                        model.predict(matrix(evaluation, features)), 0.0, 1.0
                    )
                    scored.loc[eval_mask, score_column] = predictions
                    model_path = (
                        model_dir
                        / f"outer{outer_fold}_{mode}_{variant}_internal_held{held_fold}.joblib"
                    )
                    joblib.dump(model, model_path)
                    audits.append(
                        {
                            "outer_fold": outer_fold,
                            "mode": mode,
                            "variant": variant,
                            "evaluation_scope": "internal_oof",
                            "held_inner_fold": held_fold,
                            "train_inner_folds": ",".join(
                                str(fold)
                                for fold in range(1, INNER_FOLDS + 1)
                                if fold != held_fold
                            ),
                            "train_segments": len(train),
                            "train_cases": train["case_id"].nunique(),
                            "train_participants": len(train_people),
                            "evaluation_segments": len(evaluation),
                            "evaluation_cases": evaluation["case_id"].nunique(),
                            "evaluation_participants": len(eval_people),
                            "participant_overlap": 0,
                            "train_frame_weight": int(np.sum(weights)),
                            "train_weighted_error_rate": float(
                                np.average(targets, weights=weights)
                            ),
                            "score_min": float(np.min(predictions)),
                            "score_mean": float(np.mean(predictions)),
                            "score_max": float(np.max(predictions)),
                            "feature_count": len(features),
                            "features": ",".join(features),
                            "model_path": str(model_path),
                            "model_sha256": file_sha256(model_path),
                            "uses_outer_test_targets": False,
                        }
                    )
                if scored.loc[oof_mask, score_column].isna().any():
                    raise AssertionError(
                        f"Missing cross-fit scores: outer={outer_fold}/{mode}/{variant}"
                    )

                train = scored.loc[oof_mask]
                evaluation = scored.loc[outer_mask]
                train_people = set(train["participant"])
                eval_people = set(evaluation["participant"])
                if train_people.intersection(eval_people):
                    raise AssertionError(
                        f"Official participant overlap: outer={outer_fold}/{mode}"
                    )
                model = selector_model()
                targets = train["error_fraction"].to_numpy(dtype=float)
                weights = train["length"].to_numpy(dtype=float)
                model.fit(matrix(train, features), targets, sample_weight=weights)
                predictions = np.clip(
                    model.predict(matrix(evaluation, features)), 0.0, 1.0
                )
                scored.loc[outer_mask, score_column] = predictions
                model_path = (
                    model_dir / f"outer{outer_fold}_{mode}_{variant}_official.joblib"
                )
                joblib.dump(model, model_path)
                audits.append(
                    {
                        "outer_fold": outer_fold,
                        "mode": mode,
                        "variant": variant,
                        "evaluation_scope": "official_outer_test",
                        "held_inner_fold": f"outer_{outer_fold}",
                        "train_inner_folds": "1,2,3",
                        "train_segments": len(train),
                        "train_cases": train["case_id"].nunique(),
                        "train_participants": len(train_people),
                        "evaluation_segments": len(evaluation),
                        "evaluation_cases": evaluation["case_id"].nunique(),
                        "evaluation_participants": len(eval_people),
                        "participant_overlap": 0,
                        "train_frame_weight": int(np.sum(weights)),
                        "train_weighted_error_rate": float(
                            np.average(targets, weights=weights)
                        ),
                        "score_min": float(np.min(predictions)),
                        "score_mean": float(np.mean(predictions)),
                        "score_max": float(np.max(predictions)),
                        "feature_count": len(features),
                        "features": ",".join(features),
                        "model_path": str(model_path),
                        "model_sha256": file_sha256(model_path),
                        "uses_outer_test_targets": False,
                    }
                )
                if scored.loc[outer_mask, score_column].isna().any():
                    raise AssertionError(
                        f"Missing outer scores: outer={outer_fold}/{mode}/{variant}"
                    )
    return scored, pd.DataFrame(audits)


def aggregate_tas(
    cases: Sequence[CaseData], predictions: Mapping[Any, np.ndarray]
) -> Dict[str, float]:
    total_frames = 0
    correct_frames = 0
    edits: List[float] = []
    counts = {0.10: [0, 0, 0], 0.25: [0, 0, 0], 0.50: [0, 0, 0]}
    for case in cases:
        pred = np.asarray(predictions[case.key], dtype=str)
        gt = np.asarray(case.gt, dtype=str)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction length mismatch for {case.key}")
        total_frames += len(gt)
        correct_frames += int(np.sum(pred == gt))
        # The locked Breakfast reports treated SIL=0 as an ordinary label.
        edits.append(float(_edit_score_asformer(pred.tolist(), gt.tolist(), ["background"])))
        for threshold in counts:
            tp, fp, fn = _segmental_f1_counts_asformer(
                gt.tolist(), pred.tolist(), threshold, None
            )
            counts[threshold][0] += tp
            counts[threshold][1] += fp
            counts[threshold][2] += fn
    metrics: Dict[str, float] = {
        "acc": 100.0 * safe_div(correct_frames, total_frames),
        "edit": float(np.mean(edits)),
    }
    for threshold, (tp, fp, fn) in counts.items():
        metrics[f"f1@{int(100 * threshold)}"] = 100.0 * safe_div(
            2 * tp, 2 * tp + fp + fn
        )
    return metrics


def select_spans_at_budget(
    cases: Sequence[CaseData],
    segments: pd.DataFrame,
    score_column: str,
    budget: float,
) -> Dict[Any, np.ndarray]:
    """Select complete predicted spans plus one deterministic centered cutoff."""
    masks = {case.key: np.zeros(len(case.gt), dtype=bool) for case in cases}
    requested = max(1, int(round(budget * sum(len(case.gt) for case in cases))))
    segment_ids = np.unique(np.concatenate([case.segment_ids for case in cases]))
    scoped = segments.set_index("segment_id").loc[segment_ids].copy()
    scoped["tie_key"] = [
        int(
            stable_uint64(
                row.scope,
                row.mode,
                row.inner_fold,
                row.case_id,
                row.segment_index,
            )
        )
        for row in scoped.itertuples()
    ]
    scoped = scoped.sort_values(
        [score_column, "tie_key"], ascending=[False, True], kind="mergesort"
    )
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("Cases passed to a budget selection must have unique IDs")
    remaining = requested
    cutoff_count = 0
    for row in scoped.itertuples():
        if remaining <= 0:
            break
        case = case_by_id[str(row.case_id)]
        start, end = int(row.start), int(row.end)
        length = end - start
        if length <= remaining:
            masks[case.key][start:end] = True
            remaining -= length
        else:
            cutoff_start = start + (length - remaining) // 2
            masks[case.key][cutoff_start : cutoff_start + remaining] = True
            remaining = 0
            cutoff_count += 1
    if remaining != 0:
        raise AssertionError(f"Failed to fill requested frame budget; remaining={remaining}")
    if cutoff_count > 1:
        raise AssertionError("At most one cutoff block is permitted")
    return masks


def metric_row(
    cases: Sequence[CaseData],
    masks: Mapping[Any, np.ndarray],
    baseline: Mapping[str, float],
    *,
    mode: str,
    scope: str,
    fold: Any,
    variant: str,
    budget: float,
) -> Dict[str, Any]:
    total = sum(len(case.gt) for case in cases)
    errors = sum(int(np.sum(case.pred != case.gt)) for case in cases)
    long_total = sum(int(np.sum(case.long_substitution)) for case in cases)
    flagged = 0
    flagged_errors = 0
    flagged_long = 0
    corrected: Dict[Any, np.ndarray] = {}
    for case in cases:
        mask = masks[case.key]
        error = case.pred != case.gt
        flagged += int(np.sum(mask))
        flagged_errors += int(np.sum(mask & error))
        flagged_long += int(np.sum(mask & case.long_substitution))
        oracle = case.pred.copy()
        oracle[mask] = case.gt[mask]
        corrected[case.key] = oracle
    oracle_metrics = aggregate_tas(cases, corrected)
    precision = safe_div(flagged_errors, flagged)
    return {
        "scope": scope,
        "mode": mode,
        "fold": fold,
        "variant": variant,
        "requested_budget": budget,
        "n_cases": len(cases),
        "n_frames": total,
        "baseline_error_frames": errors,
        "flagged_frames": flagged,
        "actual_coverage_pct": 100.0 * safe_div(flagged, total),
        "flagged_error_frames": flagged_errors,
        "error_precision_pct": 100.0 * precision,
        "error_recall_pct": 100.0 * safe_div(flagged_errors, errors),
        "precision_lift_vs_random": safe_div(precision, safe_div(errors, total)),
        "long_substitution_frames": long_total,
        "flagged_long_substitution_frames": flagged_long,
        "long_substitution_recall_pct": 100.0 * safe_div(flagged_long, long_total),
        "oracle_acc_gain_pp": oracle_metrics["acc"] - baseline["acc"],
        "oracle_edit_gain_pp": oracle_metrics["edit"] - baseline["edit"],
        "oracle_f1_at_10_gain_pp": oracle_metrics["f1@10"] - baseline["f1@10"],
        "oracle_f1_at_25_gain_pp": oracle_metrics["f1@25"] - baseline["f1@25"],
        "oracle_f1_at_50_gain_pp": oracle_metrics["f1@50"] - baseline["f1@50"],
        "oracle_is_non_deployable": True,
    }


def evaluate_selectors(
    cases: Sequence[CaseData],
    segments: pd.DataFrame,
    variants: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    discrimination_rows: List[Dict[str, Any]] = []
    for mode in sorted({case.mode for case in cases}):
        mode_cases = [case for case in cases if case.mode == mode]
        scopes: List[Tuple[str, Any, List[CaseData]]] = []
        for outer_fold in OUTER_FOLDS:
            for inner_fold in range(1, INNER_FOLDS + 1):
                scopes.append(
                    (
                        "internal_oof",
                        f"outer_{outer_fold}_inner_{inner_fold}",
                        [
                            case
                            for case in mode_cases
                            if case.outer_fold == outer_fold
                            and case.scope == "oof_validation"
                            and case.inner_fold == inner_fold
                        ],
                    )
                )
            scopes.append(
                (
                    "official_outer_test",
                    f"outer_{outer_fold}",
                    [
                        case
                        for case in mode_cases
                        if case.outer_fold == outer_fold and case.scope == "outer_test"
                    ],
                )
            )
        scopes.append(
            (
                "pooled_outer_test",
                "pooled",
                [case for case in mode_cases if case.scope == "outer_test"],
            )
        )
        for scope, fold, scoped_cases in scopes:
            if not scoped_cases:
                raise ValueError(f"No cases for {mode}/{scope}/{fold}")
            baseline = aggregate_tas(
                scoped_cases, {case.key: case.pred for case in scoped_cases}
            )
            baseline_rows.append(
                {
                    "scope": scope,
                    "mode": mode,
                    "fold": fold,
                    "n_cases": len(scoped_cases),
                    "n_frames": sum(len(case.gt) for case in scoped_cases),
                    "error_frames": sum(
                        int(np.sum(case.pred != case.gt)) for case in scoped_cases
                    ),
                    "long_substitution_frames": sum(
                        int(np.sum(case.long_substitution)) for case in scoped_cases
                    ),
                    **baseline,
                }
            )
            segment_ids = np.unique(
                np.concatenate([case.segment_ids for case in scoped_cases])
            )
            scoped_segments = segments.set_index("segment_id").loc[segment_ids]
            for variant in variants:
                score_column = f"{variant}_score"
                scores: List[np.ndarray] = []
                errors: List[np.ndarray] = []
                for case in scoped_cases:
                    case_scores = scoped_segments.loc[case.segment_ids, score_column]
                    scores.append(case_scores.to_numpy(dtype=float))
                    errors.append(case.pred != case.gt)
                frame_scores = np.concatenate(scores)
                frame_errors = np.concatenate(errors)
                if len(np.unique(frame_errors)) == 2:
                    auroc = float(roc_auc_score(frame_errors, frame_scores))
                    auprc = float(average_precision_score(frame_errors, frame_scores))
                else:
                    auroc = auprc = float("nan")
                discrimination_rows.append(
                    {
                        "scope": scope,
                        "mode": mode,
                        "fold": fold,
                        "variant": variant,
                        "n_frames": len(frame_scores),
                        "error_rate_pct": 100.0 * float(np.mean(frame_errors)),
                        "auroc": auroc,
                        "auprc": auprc,
                    }
                )
                for budget in FRAME_BUDGETS:
                    masks = select_spans_at_budget(
                        scoped_cases, segments, score_column, budget
                    )
                    row = metric_row(
                        scoped_cases,
                        masks,
                        baseline,
                        mode=mode,
                        scope=scope,
                        fold=fold,
                        variant=variant,
                        budget=budget,
                    )
                    expected_flagged = max(
                        1, int(round(budget * sum(len(case.gt) for case in scoped_cases)))
                    )
                    if row["flagged_frames"] != expected_flagged:
                        raise AssertionError(
                            f"Matched-budget failure for {mode}/{scope}/{fold}/{variant}/{budget}: "
                            f"{row['flagged_frames']} vs {expected_flagged}"
                        )
                    metric_rows.append(row)
    return (
        pd.DataFrame(baseline_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(discrimination_rows),
    )


def make_scale_decision(
    metrics: pd.DataFrame, metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    official = metrics[
        (metrics["mode"] == "official")
        & np.isclose(metrics["requested_budget"].astype(float), 0.05)
        & (metrics["variant"] == "base")
    ].copy()
    per_fold = official[official["scope"] == "official_outer_test"].copy()
    if set(per_fold["fold"]) != {f"outer_{fold}" for fold in OUTER_FOLDS}:
        raise ValueError("Scale decision is missing an official outer fold")
    pooled_rows = official[official["scope"] == "pooled_outer_test"]
    if len(pooled_rows) != 1:
        raise ValueError("Scale decision expected exactly one pooled base row")
    pooled = pooled_rows.iloc[0]
    rule = metadata["success_rule"]
    fold_rule = rule["all_outer_folds"]
    pooled_rule = rule["pooled"]
    recall_range = float(
        per_fold["error_recall_pct"].max() - per_fold["error_recall_pct"].min()
    )
    checks = {
        "all_folds_precision_at_least_85pct": bool(
            (
                per_fold["error_precision_pct"]
                >= float(fold_rule["minimum_error_precision_pct"])
            ).all()
        ),
        "all_folds_recall_at_least_15pct": bool(
            (
                per_fold["error_recall_pct"]
                >= float(fold_rule["minimum_error_recall_pct"])
            ).all()
        ),
        "recall_range_at_most_8pp": (
            recall_range
            <= float(fold_rule["maximum_recall_range_across_folds_pp"])
        ),
        "pooled_precision_at_least_85pct": (
            float(pooled["error_precision_pct"])
            >= float(pooled_rule["minimum_error_precision_pct"])
        ),
        "pooled_recall_at_least_18pct": (
            float(pooled["error_recall_pct"])
            >= float(pooled_rule["minimum_error_recall_pct"])
        ),
    }
    return {
        "decision_budget": 0.05,
        "primary_mode": "official",
        "primary_variant": "base",
        "criteria_were_pre_specified": True,
        "checks": checks,
        "green_light_repair_head": all(checks.values()),
        "per_fold": {
            str(row.fold): {
                "error_recall_pct": float(row.error_recall_pct),
                "error_precision_pct": float(row.error_precision_pct),
                "long_substitution_recall_pct": float(
                    row.long_substitution_recall_pct
                ),
                "oracle_acc_gain_pp": float(row.oracle_acc_gain_pp),
                "oracle_f1_at_25_gain_pp": float(row.oracle_f1_at_25_gain_pp),
            }
            for row in per_fold.itertuples()
        },
        "recall_range_across_folds_pp": recall_range,
        "pooled": {
            "error_recall_pct": float(pooled["error_recall_pct"]),
            "error_precision_pct": float(pooled["error_precision_pct"]),
            "long_substitution_recall_pct": float(
                pooled["long_substitution_recall_pct"]
            ),
            "oracle_acc_gain_pp": float(pooled["oracle_acc_gain_pp"]),
            "oracle_f1_at_25_gain_pp": float(
                pooled["oracle_f1_at_25_gain_pp"]
            ),
        },
        "success_rule": rule,
    }


def make_shape_decision(
    metrics: pd.DataFrame, metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply the locked exploratory shape-feature comparison against base."""
    selected = metrics[
        (metrics["mode"] == "official")
        & np.isclose(metrics["requested_budget"].astype(float), 0.05)
        & (metrics["variant"].isin(["base", "base_plus_shape"]))
        & (
            metrics["scope"].isin(
                ["official_outer_test", "pooled_outer_test"]
            )
        )
    ].copy()
    expected_folds = {*(f"outer_{fold}" for fold in OUTER_FOLDS), "pooled"}
    if set(selected["fold"]) != expected_folds:
        raise ValueError("Shape decision is missing an official fold or pooled row")
    if selected.groupby(["fold", "variant"]).size().ne(1).any():
        raise ValueError("Shape decision expected one row per fold and variant")

    deltas: Dict[str, Dict[str, float]] = {}
    for fold in sorted(expected_folds):
        rows = selected[selected["fold"] == fold].set_index("variant")
        base = rows.loc["base"]
        shape = rows.loc["base_plus_shape"]
        deltas[fold] = {
            "error_recall_gain_pp": float(
                shape.error_recall_pct - base.error_recall_pct
            ),
            "error_precision_gain_pp": float(
                shape.error_precision_pct - base.error_precision_pct
            ),
            "long_substitution_recall_gain_pp": float(
                shape.long_substitution_recall_pct
                - base.long_substitution_recall_pct
            ),
            "oracle_acc_gain_difference_pp": float(
                shape.oracle_acc_gain_pp - base.oracle_acc_gain_pp
            ),
            "oracle_f1_at_25_gain_difference_pp": float(
                shape.oracle_f1_at_25_gain_pp
                - base.oracle_f1_at_25_gain_pp
            ),
        }
    rule = metadata["shape_variant_comparison_rule"]
    positive_folds = sum(
        deltas[f"outer_{fold}"]["error_recall_gain_pp"] > 0.0
        for fold in OUTER_FOLDS
    )
    pooled = deltas["pooled"]
    checks = {
        "pooled_error_recall_gain_at_least_0p5pp": (
            pooled["error_recall_gain_pp"]
            >= float(rule["minimum_pooled_error_recall_gain_pp"])
        ),
        "positive_error_recall_gain_in_at_least_3_of_4_folds": (
            positive_folds
            >= int(rule["minimum_outer_folds_with_positive_error_recall_gain"])
        ),
        "pooled_oracle_acc_gain_not_worse": (
            pooled["oracle_acc_gain_difference_pp"]
            >= float(rule["minimum_pooled_oracle_acc_gain_difference_pp"])
        ),
        "pooled_oracle_f1_at_25_gain_not_worse": (
            pooled["oracle_f1_at_25_gain_difference_pp"]
            >= float(rule["minimum_pooled_oracle_f1_at_25_gain_difference_pp"])
        ),
    }
    return {
        "decision_budget": 0.05,
        "primary_mode": "official",
        "variant": "base_plus_shape",
        "reference": "base",
        "analysis_time_exploratory_only": True,
        "criteria_were_pre_specified": True,
        "checks": checks,
        "retain_shape_features": all(checks.values()),
        "positive_outer_fold_count": positive_folds,
        "deltas": deltas,
        "comparison_rule": rule,
    }


def write_findings(
    out_dir: Path,
    baselines: pd.DataFrame,
    metrics: pd.DataFrame,
    decisions: Mapping[str, Any],
    shape_decision: Mapping[str, Any],
) -> None:
    pooled_baseline = baselines[
        (baselines["scope"] == "pooled_outer_test") & (baselines["mode"] == "official")
    ].iloc[0]
    outer = metrics[
        (metrics["scope"].isin(["official_outer_test", "pooled_outer_test"]))
        & (metrics["mode"] == "official")
        & np.isclose(metrics["requested_budget"].astype(float), 0.05)
    ].copy()
    lines = [
        "# Breakfast four-fold video-aware selector scale-up",
        "",
        "## Status",
        "",
        (
            "This is a leakage-safe four-fold study. Every outer-fold selector is trained "
            "only from that fold's subject-disjoint OOF DiffAct predictions."
        ),
        "",
        "## Outer-fold baseline",
        "",
        (
            f"Pooled official DiffAct: Acc {pooled_baseline.acc:.3f}, "
            f"Edit {pooled_baseline.edit:.3f}, F1@10 {pooled_baseline['f1@10']:.3f}, "
            f"F1@25 {pooled_baseline['f1@25']:.3f}, "
            f"F1@50 {pooled_baseline['f1@50']:.3f}."
        ),
        "",
        "## Matched 5% frame budget",
        "",
        "| Scope | Selector | Error recall | Error precision | Long-sub recall | Oracle ΔAcc | Oracle ΔF1@25 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    order = [*(f"outer_{fold}" for fold in OUTER_FOLDS), "pooled"]
    for scope_name in order:
        scope_rows = outer[outer["fold"] == scope_name].set_index("variant")
        for variant in VARIANTS:
            row = scope_rows.loc[variant]
            lines.append(
                f"| {scope_name} | {variant} | {row.error_recall_pct:.3f}% | "
                f"{row.error_precision_pct:.3f}% | "
                f"{row.long_substitution_recall_pct:.3f}% | "
                f"{row.oracle_acc_gain_pp:+.3f} pp | "
                f"{row.oracle_f1_at_25_gain_pp:+.3f} pp |"
            )
    lines.extend(["", "## Pre-registered base-selector gate", ""])
    for name, passed in decisions["checks"].items():
        lines.append(
            f"- {'PASS' if passed else 'FAIL'}: {name}"
        )
    scale = decisions["green_light_repair_head"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "The deployment-safe base selector passed; green-light repair-head development."
                if scale
                else "The base selector did not meet the locked consistency rule; do not "
                "advance the repair head automatically."
            ),
            "",
            "Oracle corrections replace selected frames with ground truth and are diagnostic ceilings only. "
            "They are not deployable predictions. `base_plus_metadata` is an analysis-only "
            "ablation; the primary base uses exactly 15 deployment-safe numeric features. "
            "No DFG or Petri selector is evaluated.",
            "",
        ]
    )
    lines.extend(["## Exploratory `base_plus_shape` gate", ""])
    for name, passed in shape_decision["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend(
        [
            "",
            (
                "Retain the raw-probability shape features for repair-head localization."
                if shape_decision["retain_shape_features"]
                else "Drop the shape features and retain the frozen 15-feature base."
            ),
            "",
            "`runner_up_gap` is the explicitly requested mean p1−p2 feature. "
            "It is mathematically identical to the existing `margin_mean` base feature; "
            "the other four shape features are new signals.",
            "",
        ]
    )
    (out_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(
    metadata: Mapping[str, Any],
    cases: Sequence[CaseData],
    segments: pd.DataFrame,
    model_audit: pd.DataFrame,
    baselines: pd.DataFrame,
    metrics: pd.DataFrame,
    context_audit: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for outer_fold in OUTER_FOLDS:
        fold_meta = metadata["folds"][str(outer_fold)]
        for mode in MODES:
            oof = [
                case
                for case in cases
                if case.outer_fold == outer_fold
                and case.mode == mode
                and case.scope == "oof_validation"
            ]
            outer = [
                case
                for case in cases
                if case.outer_fold == outer_fold
                and case.mode == mode
                and case.scope == "outer_test"
            ]
            expected_oof = int(fold_meta["outer_train_case_count"])
            expected_outer = int(fold_meta["outer_test_case_count"])
            add(
                f"outer {outer_fold}/{mode}: OOF coverage",
                len(oof) == expected_oof
                and len({case.case_id for case in oof}) == expected_oof,
                f"rows={len(oof)}, expected={expected_oof}",
            )
            add(
                f"outer {outer_fold}/{mode}: test coverage",
                len(outer) == expected_outer
                and len({case.case_id for case in outer}) == expected_outer,
                f"rows={len(outer)}, expected={expected_outer}",
            )
    add(
        "feature case leakage absent",
        not segments["eval_case_in_feature_train"].astype(bool).any(),
        "all segment flags false",
    )
    add(
        "feature participant leakage absent",
        not segments["eval_participant_in_feature_train"].astype(bool).any(),
        "all segment flags false",
    )
    add(
        "selector participant overlap absent",
        bool((model_audit["participant_overlap"] == 0).all()),
        f"max overlap={model_audit['participant_overlap'].max()}",
    )
    add(
        "outer targets unused by selector fit",
        not model_audit["uses_outer_test_targets"].astype(bool).any(),
        "model audit uniformly false",
    )
    score_columns = [f"{variant}_score" for variant in VARIANTS]
    add(
        "all learned scores finite",
        bool(np.isfinite(segments[score_columns].to_numpy(dtype=float)).all()),
        f"columns={score_columns}",
    )
    add(
        "matched budgets exact",
        bool(
            np.allclose(
                metrics["actual_coverage_pct"].to_numpy(dtype=float),
                100.0 * metrics["flagged_frames"].to_numpy(dtype=float)
                / metrics["n_frames"].to_numpy(dtype=float),
            )
        ),
        "reported coverage reconciles to flagged/n_frames",
    )
    add(
        "no process features computed",
        bool((context_audit["uses_process_features"] == False).all()),  # noqa: E712
        f"context rows={len(context_audit)}",
    )
    primary = list(feature_sets["base"])
    add(
        "primary feature list exact",
        primary == list(BASE_NUMERIC_FEATURES),
        f"count={len(primary)}",
    )
    forbidden = [
        column
        for column in primary
        if column.startswith(FORBIDDEN_PRIMARY_FEATURE_PREFIXES)
    ]
    add(
        "primary excludes task/camera metadata",
        not forbidden,
        f"forbidden={forbidden}",
    )
    add(
        "shape feature list exact",
        list(feature_sets["base_plus_shape"])
        == [*BASE_NUMERIC_FEATURES, *SHAPE_FEATURES],
        f"shape_count={len(feature_sets['base_plus_shape'])}",
    )
    shape_values = segments[list(SHAPE_FEATURES)].to_numpy(dtype=float)
    add(
        "shape features finite",
        bool(np.isfinite(shape_values).all()),
        f"rows={len(segments)}",
    )
    add(
        "runner-up gap reconciles to base margin",
        bool(
            np.allclose(
                segments["runner_up_gap"].to_numpy(dtype=float),
                segments["margin_mean"].to_numpy(dtype=float),
                atol=1e-12,
            )
        ),
        "requested runner_up_gap is the existing mean p1-p2 definition",
    )
    probability_columns = [
        f"candidate_rank_{rank}_mean_probability"
        for rank in range(1, REPAIR_CANDIDATE_TOP_K + 1)
    ]
    candidate_probability = segments[probability_columns].to_numpy(dtype=float)
    add(
        "top-5 candidate probabilities valid and sorted",
        bool(
            np.isfinite(candidate_probability).all()
            and (candidate_probability >= 0.0).all()
            and (candidate_probability <= 1.0 + 1e-7).all()
            and (np.diff(candidate_probability, axis=1) <= 1e-12).all()
        ),
        f"columns={probability_columns}",
    )
    candidate_ids = [
        f"candidate_rank_{rank}_class_id"
        for rank in range(1, REPAIR_CANDIDATE_TOP_K + 1)
    ]
    add(
        "top-5 candidate IDs unique per segment",
        bool(
            np.asarray(
                [
                    len(set(row)) == REPAIR_CANDIDATE_TOP_K
                    for row in segments[candidate_ids]
                    .to_numpy(dtype=int)
                    .tolist()
                ],
                dtype=bool,
            ).all()
        ),
        f"rows={len(segments)}",
    )
    rank = segments["correct_label_probability_rank"].to_numpy(dtype=int)
    add(
        "correct-label top-5 flags reconcile",
        bool(
            (rank >= 1).all()
            and (
                segments["correct_label_in_top5"].astype(bool).to_numpy()
                == (rank <= REPAIR_CANDIDATE_TOP_K)
            ).all()
        ),
        "correct_label_in_top5 equals rank<=5",
    )
    add(
        "outer baseline cardinality",
        bool(
            (
                baselines[
                    (baselines.scope == "pooled_outer_test")
                    & (baselines["mode"] == "official")
                ].iloc[0].n_cases
                == sum(
                    int(metadata["folds"][str(fold)]["outer_test_case_count"])
                    for fold in OUTER_FOLDS
                )
            )
        ),
        "pooled outer test must cover all official cases",
    )
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise AssertionError(f"Analysis validation failed: {failed}")
    return result


def validated_feature_sets(
    study_dir: Path,
    metadata: Mapping[str, Any],
    categorical: Sequence[str],
) -> Dict[str, List[str]]:
    config_path = Path(metadata["selector_config"])
    if config_path != study_dir / "selector_config.json":
        raise ValueError("Selector config path escaped the immutable study")
    if file_sha256(config_path) != metadata["selector_config_sha256"]:
        raise ValueError("Selector config hash changed after study generation")
    config = load_json(config_path)
    declared_base = config["variants"]["base"]["feature_columns"]
    if declared_base != list(BASE_NUMERIC_FEATURES):
        raise ValueError(
            "Primary selector feature list must equal BASE_NUMERIC_FEATURES exactly"
        )
    forbidden = [
        column
        for column in declared_base
        if column.startswith(FORBIDDEN_PRIMARY_FEATURE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(
            f"Deployment-forbidden metadata entered the primary features: {forbidden}"
        )
    if any(name in config["variants"] for name in config["forbidden_variants"]):
        raise ValueError("A rejected process-feature variant re-entered selector config")
    declared_shape = config["variants"]["base_plus_shape"]
    if declared_shape["base_feature_columns"] != list(BASE_NUMERIC_FEATURES):
        raise ValueError("Shape variant must inherit the frozen base exactly")
    if declared_shape["additional_feature_columns"] != list(SHAPE_FEATURES):
        raise ValueError("Shape variant feature list differs from SHAPE_FEATURES")
    feature_sets = variant_features(categorical)
    if feature_sets["base"] != declared_base:
        raise ValueError("Runtime primary features differ from immutable selector config")
    if feature_sets["base_plus_shape"] != [
        *BASE_NUMERIC_FEATURES,
        *SHAPE_FEATURES,
    ]:
        raise ValueError("Runtime shape features differ from immutable selector config")
    if set(feature_sets) != set(VARIANTS):
        raise ValueError(f"Unexpected runtime variants: {sorted(feature_sets)}")
    return feature_sets


def validated_repair_corpus_schema(
    study_dir: Path, metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    schema_path = Path(metadata["repair_corpus_schema"])
    if schema_path != study_dir / "repair_corpus_schema.json":
        raise ValueError("Repair-corpus schema path escaped the immutable study")
    if file_sha256(schema_path) != metadata["repair_corpus_schema_sha256"]:
        raise ValueError("Repair-corpus schema hash changed after study generation")
    schema = load_json(schema_path)
    candidate = schema["candidate_pool"]
    if int(candidate["top_k"]) != REPAIR_CANDIDATE_TOP_K:
        raise ValueError("Repair candidate pool size differs from the locked top-k")
    if candidate["fields_per_rank"] != list(REPAIR_CANDIDATE_FIELDS):
        raise ValueError("Repair candidate field schema changed")
    if candidate["columns"] != list(repair_candidate_columns()):
        raise ValueError("Repair candidate columns differ from the locked schema")
    if schema["primary_feature_columns"] != list(BASE_NUMERIC_FEATURES):
        raise ValueError("Repair schema primary features differ from frozen base")
    if schema["exploratory_shape_feature_columns"] != list(SHAPE_FEATURES):
        raise ValueError("Repair schema shape features changed")
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    out_dir = (
        args.out_dir or (study_dir / "analysis" / "video_selector_allfolds_v2")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_json(study_dir / "study_metadata.json")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported study protocol: {metadata.get('protocol_version')}")
    verify_source_digest(metadata, Path(metadata["diffact_root"]))
    data_root = Path(metadata["data_root"])
    tasks = load_json(study_dir / "tasks.json")["tasks"]
    id_to_name, name_to_id = parse_mapping(data_root / DATASET / "mapping.txt")

    all_cases: List[CaseData] = []
    segment_frames: List[pd.DataFrame] = []
    case_frames: List[pd.DataFrame] = []
    context_rows: List[Dict[str, Any]] = []
    categorical_columns: set[str] = set()
    next_segment_id = 0
    print("[1/5] Building train-only contexts and four-fold segment features", flush=True)
    for outer_fold in OUTER_FOLDS:
        fold_metadata = metadata["folds"][str(outer_fold)]
        fold_tasks = [
            task for task in tasks if int(task["outer_fold"]) == outer_fold
        ]
        contexts, fold_context_rows = build_feature_contexts(
            study_dir,
            fold_metadata,
            data_root,
            name_to_id,
            out_dir / f"outer_fold_{outer_fold}",
        )
        for row in fold_context_rows:
            row["outer_fold"] = outer_fold
        context_rows.extend(fold_context_rows)
        fold_cases, fold_segments, fold_inventory, categorical = build_segment_rows(
            study_dir,
            fold_metadata,
            fold_tasks,
            args.modes,
            data_root,
            id_to_name,
            name_to_id,
            contexts,
            outer_fold=outer_fold,
            segment_id_start=next_segment_id,
        )
        next_segment_id = int(fold_segments["segment_id"].max()) + 1
        all_cases.extend(fold_cases)
        segment_frames.append(fold_segments)
        case_frames.append(fold_inventory)
        categorical_columns.update(categorical)

    segments = pd.concat(segment_frames, ignore_index=True, sort=False).fillna(0.0)
    case_inventory = pd.concat(case_frames, ignore_index=True, sort=False)
    context_audit = pd.DataFrame(context_rows)
    context_audit.to_csv(out_dir / "feature_context_audit.csv", index=False)
    feature_sets = validated_feature_sets(
        study_dir, metadata, sorted(categorical_columns)
    )
    repair_schema = validated_repair_corpus_schema(study_dir, metadata)
    segments.to_csv(out_dir / "segment_features_unscored.csv", index=False)
    case_inventory.to_csv(out_dir / "case_inventory.csv", index=False)

    print("[2/5] Cross-fitting deployment-safe selectors independently per fold", flush=True)
    scored, model_audit = fit_selector_scores(segments, feature_sets, out_dir)
    scored.to_csv(out_dir / "segment_scores.csv", index=False)
    model_audit.to_csv(out_dir / "selector_model_audit.csv", index=False)

    print("[3/5] Evaluating per-fold and pooled matched frame budgets", flush=True)
    baselines, budget_metrics, discrimination = evaluate_selectors(
        all_cases, scored, list(VARIANTS)
    )
    baselines.to_csv(out_dir / "baseline_metrics.csv", index=False)
    budget_metrics.to_csv(out_dir / "selector_budget_metrics.csv", index=False)
    discrimination.to_csv(out_dir / "selector_discrimination.csv", index=False)
    decision = make_scale_decision(budget_metrics, metadata)
    atomic_write_json(out_dir / "scale_decision.json", decision)
    shape_decision = make_shape_decision(budget_metrics, metadata)
    atomic_write_json(out_dir / "shape_variant_decision.json", shape_decision)

    print("[4/5] Writing the repair-head OOF training corpus", flush=True)
    repair_corpus = scored[
        (scored["scope"] == "oof_validation")
        & (scored["mode"] == "official")
    ].copy()
    expected_case_records = int(
        metadata["repair_corpus_contract"]["expected_case_records"]
    )
    observed_case_records = repair_corpus[
        ["outer_fold", "case_id"]
    ].drop_duplicates().shape[0]
    if observed_case_records != expected_case_records:
        raise ValueError(
            f"Repair corpus case coverage {observed_case_records}, "
            f"expected {expected_case_records}"
        )
    required_repair_columns = {
        "correct_label",
        "correct_label_name",
        "correct_label_probability_rank",
        "correct_label_in_top5",
        "correct_label_fraction",
        "error_fraction",
        "base_score",
        *repair_candidate_columns(),
    }
    if not required_repair_columns.issubset(repair_corpus.columns):
        raise ValueError("Repair corpus lacks correct-label or selector-score columns")
    repair_corpus.to_csv(out_dir / "repair_training_corpus_segments.csv", index=False)

    print("[5/5] Validating and writing the decision readout", flush=True)
    checks = validate_outputs(
        metadata,
        all_cases,
        scored,
        model_audit,
        baselines,
        budget_metrics,
        context_audit,
        feature_sets,
    )
    checks.to_csv(out_dir / "validation_checks.csv", index=False)
    write_findings(
        out_dir, baselines, budget_metrics, decision, shape_decision
    )
    completed_utc = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "completed_utc": completed_utc,
        "study_dir": str(study_dir),
        "out_dir": str(out_dir),
        "source_digest": metadata["source_provenance"]["source_digest"],
        "modes": args.modes,
        "variants": list(VARIANTS),
        "frame_budgets": FRAME_BUDGETS,
        "long_substitution_definition": {
            "minimum_contiguous_error_length": LONG_MIN_LEN,
            "minimum_predicted_label_homogeneity": LONG_HOMOGENEITY,
        },
        "selector_hyperparameters": selector_model().get_params(),
        "feature_sets": feature_sets,
        "primary_mode": "official",
        "primary_variant": "base",
        "exploratory_shape_variant": "base_plus_shape",
        "shape_variant_retain": shape_decision["retain_shape_features"],
        "oracle_outputs_are_diagnostic_only": True,
        "repair_corpus_case_records": observed_case_records,
        "repair_corpus_schema_sha256": metadata[
            "repair_corpus_schema_sha256"
        ],
        "repair_candidate_columns": repair_schema["candidate_pool"]["columns"],
        "validation_passed": True,
    }
    atomic_write_json(out_dir / "run_metadata.json", run_metadata)
    atomic_write_json(
        out_dir / "analysis_complete.json",
        {
            "completed": True,
            "completed_utc": completed_utc,
            "green_light_repair_head": decision["green_light_repair_head"],
            "retain_shape_features": shape_decision["retain_shape_features"],
            "repair_corpus_case_records": observed_case_records,
            "source_digest": metadata["source_provenance"]["source_digest"],
        },
    )
    print(f"Analysis complete: {out_dir}")
    print((out_dir / "findings.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
