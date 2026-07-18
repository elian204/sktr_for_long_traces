#!/usr/bin/env python3
"""Leakage-safe video-aware DiffAct span-selector bake-off for Breakfast."""

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
    DATASET,
    DEFAULT_STUDY_DIR,
    FRAME_BUDGETS,
    INNER_FOLDS,
    PROTOCOL_VERSION,
    WORKSPACE_ROOT,
    atomic_write_json,
    file_sha256,
    ground_truth_path,
    load_json,
    normalize_case_id,
    parse_case,
    read_bundle,
    read_lines,
    verify_source_digest,
)


if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation import _edit_score_asformer, _segmental_f1_counts_asformer  # noqa: E402


RANDOM_STATE = 20260717
LONG_MIN_LEN = 100
LONG_HOMOGENEITY = 0.90
MODES = ("official", "raw_argmax")
VARIANT_PROCESS_KIND = {
    "base": "none",
    "base_plus_dfg": "dfg",
    "base_plus_prefix_petri": "prefix_petri",
}

BASE_NUMERIC_FEATURES = (
    "uncertainty_mean",
    "uncertainty_q90",
    "entropy_mean",
    "top1_uncertainty_mean",
    "margin_mean",
    "pred_probability_mean",
    "official_override_gap_mean",
    "duration_raw_abs_z",
    "duration_norm_abs_z",
    "segment_log_length",
    "segment_fraction",
    "video_progress_mid",
    "class_frame_rarity",
    "class_segment_rarity",
    "neighbor_class_rarity",
)
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
    def key(self) -> Tuple[str, str, int | None, str]:
        return (self.scope, self.mode, self.inner_fold, self.case_id)


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
    columns: List[str] = []
    columns.extend(f"predicted_label__{label}" for label in labels)
    columns.extend(f"left_label__{label}" for label in ["START", *labels])
    columns.extend(f"right_label__{label}" for label in [*labels, "END"])
    columns.extend(f"task__{task}" for task in tasks)
    columns.extend(f"camera__{camera}" for camera in cameras)
    return columns


def variant_features(categorical: Sequence[str]) -> Dict[str, List[str]]:
    base = [*BASE_NUMERIC_FEATURES, *categorical]
    return {
        "base": base,
        "base_plus_dfg": [*base, *DFG_FEATURES],
        "base_plus_prefix_petri": [*base, *PETRI_FEATURES],
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
        petri_by_task: Dict[str, PetriRuntime] = {}
        for task, stats in stats_by_task.items():
            model_path = out_dir / "process_models" / context_id / f"{task}.pnml"
            runtime = build_route_petri(task, stats.traces, model_path)
            petri_by_task[task] = runtime
            audit_rows.append(
                {
                    "context_id": context_id,
                    "feature_train_manifest": str(manifest),
                    "feature_train_manifest_sha256": file_sha256(manifest),
                    "task": task,
                    "train_cases": stats.total_cases,
                    "train_frames": stats.total_frames,
                    "train_collapsed_segments": stats.total_segments,
                    "petri_places": len(runtime.places),
                    "petri_transitions": len(runtime.transitions),
                    "petri_model_path": str(model_path),
                    "petri_model_sha256": file_sha256(model_path),
                }
            )
        contexts[context_id] = {
            "manifest": manifest,
            "manifest_sha256": file_sha256(manifest),
            "cases": case_set,
            "participants": participants,
            "stats": stats_by_task,
            "petri": petri_by_task,
            "prefix_cache": {},
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
        complete_path = Path(task["run_dir"]) / "task_complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(
                f"OOF task is not complete: {task['task_id']} ({complete_path})"
            )
        complete = load_json(complete_path)
        if complete.get("status") != "complete":
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
    next_segment_id = 0
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
            if info.task not in context["stats"] or info.task not in context["petri"]:
                raise ValueError(f"No route model for {case_id} task={info.task}")
            stats: RouteStats = context["stats"][info.task]
            runtime: PetriRuntime = context["petri"][info.task]

            for mode in modes:
                prediction_int = official_pred if mode == "official" else raw_pred
                pred = prediction_int.astype(str)
                gt_text = gt.astype(str)
                probability = probability_features(raw_prob, prediction_int)
                segments = collapse_segments(pred.tolist())
                trace = [label for _, _, label in segments]
                dfg = dfg_features(trace, stats)
                cache_key = (info.task, tuple(trace))
                prefix_cache: Dict[Any, Tuple[np.ndarray, int]] = context["prefix_cache"]
                if cache_key not in prefix_cache:
                    prefix_cache[cache_key] = exact_prefix_costs(trace, runtime)
                prefix_cost, search_states = prefix_cache[cache_key]
                prefix_increment = np.diff(prefix_cost)
                if len(prefix_increment) != len(segments):
                    raise AssertionError("Prefix feature length does not match predicted segments")
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
                    row: Dict[str, Any] = {
                        "segment_id": next_segment_id,
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
                        "left_label": left_label,
                        "right_label": right_label,
                        "error_frames": int(np.sum(error[start:end])),
                        "error_fraction": float(np.mean(error[start:end])),
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
                        "petri_prefix_cost_before": float(prefix_cost[segment_index]),
                        "petri_prefix_cost_after": float(prefix_cost[segment_index + 1]),
                        "petri_prefix_cost_increment": float(prefix_increment[segment_index]),
                        "petri_prefix_cost_rate": float(
                            prefix_cost[segment_index + 1] / (segment_index + 1)
                        ),
                        "petri_prefix_violation": float(prefix_increment[segment_index] > 0),
                        "petri_case_final_prefix_cost": float(prefix_cost[-1]),
                        "petri_case_final_prefix_cost_rate": float(
                            prefix_cost[-1] / max(len(trace), 1)
                        ),
                        "petri_search_states": int(search_states),
                    }
                    for feature_name, values in dfg.items():
                        row[feature_name] = float(values[segment_index])
                    for column in categorical:
                        row[column] = 0.0
                    row[f"predicted_label__{label}"] = 1.0
                    row[f"left_label__{left_label}"] = 1.0
                    row[f"right_label__{right_label}"] = 1.0
                    row[f"task__{info.task}"] = 1.0
                    row[f"camera__{info.camera}"] = 1.0
                    segment_rows.append(row)
                    segment_ids[start:end] = next_segment_id
                    next_segment_id += 1

                cases_out.append(
                    CaseData(
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
                        "dfg_violation_segments": int(
                            np.sum(dfg["dfg_segment_severity"] > 0)
                        ),
                        "petri_prefix_cost": float(prefix_cost[-1]),
                        "petri_violation_segments": int(np.sum(prefix_increment > 0)),
                        "petri_search_states": int(search_states),
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
    """Cross-fit OOF scores, then fit all OOF rows for untouched outer test."""
    scored = segments.copy()
    audits: List[Dict[str, Any]] = []
    model_dir = out_dir / "selector_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for mode in sorted(scored["mode"].unique()):
        oof_mask = (scored["scope"] == "oof_validation") & (scored["mode"] == mode)
        outer_mask = (scored["scope"] == "outer_test") & (scored["mode"] == mode)
        if not oof_mask.any() or not outer_mask.any():
            raise ValueError(f"Missing OOF or outer rows for mode={mode}")
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
                        f"Selector CV participant leakage for mode={mode}, held_fold={held_fold}"
                    )
                model = selector_model()
                targets = train["error_fraction"].to_numpy(dtype=float)
                weights = train["length"].to_numpy(dtype=float)
                model.fit(matrix(train, features), targets, sample_weight=weights)
                predictions = np.clip(model.predict(matrix(evaluation, features)), 0.0, 1.0)
                scored.loc[eval_mask, score_column] = predictions
                model_path = model_dir / f"{mode}_{variant}_internal_held{held_fold}.joblib"
                joblib.dump(model, model_path)
                audits.append(
                    {
                        "mode": mode,
                        "variant": variant,
                        "evaluation_scope": "internal_oof",
                        "held_inner_fold": held_fold,
                        "train_inner_folds": ",".join(
                            str(fold) for fold in range(1, INNER_FOLDS + 1) if fold != held_fold
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
                raise AssertionError(f"Missing internal cross-fit scores: {mode}/{variant}")

            train = scored.loc[oof_mask]
            evaluation = scored.loc[outer_mask]
            train_people = set(train["participant"])
            eval_people = set(evaluation["participant"])
            if train_people.intersection(eval_people):
                raise AssertionError(f"Official outer train/test participants overlap for {mode}")
            model = selector_model()
            targets = train["error_fraction"].to_numpy(dtype=float)
            weights = train["length"].to_numpy(dtype=float)
            model.fit(matrix(train, features), targets, sample_weight=weights)
            predictions = np.clip(model.predict(matrix(evaluation, features)), 0.0, 1.0)
            scored.loc[outer_mask, score_column] = predictions
            model_path = model_dir / f"{mode}_{variant}_outer_fold1.joblib"
            joblib.dump(model, model_path)
            audits.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "evaluation_scope": "official_outer_test",
                    "held_inner_fold": "outer_1",
                    "train_inner_folds": "1,2,3",
                    "train_segments": len(train),
                    "train_cases": train["case_id"].nunique(),
                    "train_participants": len(train_people),
                    "evaluation_segments": len(evaluation),
                    "evaluation_cases": evaluation["case_id"].nunique(),
                    "evaluation_participants": len(eval_people),
                    "participant_overlap": 0,
                    "train_frame_weight": int(np.sum(weights)),
                    "train_weighted_error_rate": float(np.average(targets, weights=weights)),
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
                raise AssertionError(f"Missing official outer scores: {mode}/{variant}")
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
        for fold in range(1, INNER_FOLDS + 1):
            scopes.append(
                (
                    "internal_oof",
                    fold,
                    [
                        case
                        for case in mode_cases
                        if case.scope == "oof_validation" and case.inner_fold == fold
                    ],
                )
            )
        scopes.append(
            (
                "official_outer_test",
                "outer_1",
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


def make_pilot_decision(metrics: pd.DataFrame) -> Dict[str, Any]:
    official = metrics[
        (metrics["mode"] == "official")
        & np.isclose(metrics["requested_budget"].astype(float), 0.05)
    ].copy()
    decisions: Dict[str, Any] = {}
    for variant in ("base_plus_dfg", "base_plus_prefix_petri"):
        internal_deltas: List[float] = []
        for fold in range(1, INNER_FOLDS + 1):
            rows = official[
                (official["scope"] == "internal_oof") & (official["fold"].astype(str) == str(fold))
            ].set_index("variant")
            internal_deltas.append(
                float(rows.loc[variant, "error_recall_pct"] - rows.loc["base", "error_recall_pct"])
            )
        outer = official[official["scope"] == "official_outer_test"].set_index("variant")
        outer_recall_delta = float(
            outer.loc[variant, "error_recall_pct"] - outer.loc["base", "error_recall_pct"]
        )
        outer_acc_delta = float(
            outer.loc[variant, "oracle_acc_gain_pp"] - outer.loc["base", "oracle_acc_gain_pp"]
        )
        outer_f1_delta = float(
            outer.loc[variant, "oracle_f1_at_25_gain_pp"]
            - outer.loc["base", "oracle_f1_at_25_gain_pp"]
        )
        pass_flags = {
            "outer_error_recall_delta_at_least_0_5pp": outer_recall_delta >= 0.5,
            "positive_internal_folds_at_least_2_of_3": sum(
                delta > 0 for delta in internal_deltas
            )
            >= 2,
            "outer_oracle_acc_nonnegative_vs_base": outer_acc_delta >= -1e-12,
            "outer_oracle_f1_at_25_nonnegative_vs_base": outer_f1_delta >= -1e-12,
        }
        decisions[variant] = {
            "internal_error_recall_delta_pp": internal_deltas,
            "positive_internal_fold_count": sum(delta > 0 for delta in internal_deltas),
            "outer_error_recall_delta_pp": outer_recall_delta,
            "outer_oracle_acc_gain_delta_pp": outer_acc_delta,
            "outer_oracle_f1_at_25_gain_delta_pp": outer_f1_delta,
            "criteria": pass_flags,
            "passes_pilot_gate": all(pass_flags.values()),
        }
    return {
        "decision_budget": 0.05,
        "primary_mode": "official",
        "criteria_were_pre_specified": True,
        "variants": decisions,
        "scale_to_all_four_outer_folds": any(
            value["passes_pilot_gate"] for value in decisions.values()
        ),
        "one_outer_fold_is_preliminary": True,
    }


def write_findings(
    out_dir: Path,
    baselines: pd.DataFrame,
    metrics: pd.DataFrame,
    decisions: Mapping[str, Any],
) -> None:
    outer_baseline = baselines[
        (baselines["scope"] == "official_outer_test") & (baselines["mode"] == "official")
    ].iloc[0]
    outer = metrics[
        (metrics["scope"] == "official_outer_test")
        & (metrics["mode"] == "official")
        & np.isclose(metrics["requested_budget"].astype(float), 0.05)
    ].set_index("variant")
    lines = [
        "# Breakfast video-aware span selector pilot",
        "",
        "## Status",
        "",
        (
            "This is a leakage-safe official outer-fold-1 pilot over "
            f"{int(outer_baseline.n_cases)} videos and {int(outer_baseline.n_frames):,} frames. "
            "The selector was trained only from subject-disjoint OOF DiffAct predictions."
        ),
        "",
        "## Outer-fold baseline",
        "",
        (
            f"Official DiffAct: Acc {outer_baseline.acc:.3f}, Edit {outer_baseline.edit:.3f}, "
            f"F1@10 {outer_baseline['f1@10']:.3f}, F1@25 {outer_baseline['f1@25']:.3f}, "
            f"F1@50 {outer_baseline['f1@50']:.3f}."
        ),
        "",
        "## Matched 5% frame budget",
        "",
        "| Selector | Error recall | Error precision | Long-sub recall | Oracle ΔAcc | Oracle ΔF1@25 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("base", "base_plus_dfg", "base_plus_prefix_petri"):
        row = outer.loc[variant]
        lines.append(
            f"| {variant} | {row.error_recall_pct:.3f}% | {row.error_precision_pct:.3f}% | "
            f"{row.long_substitution_recall_pct:.3f}% | {row.oracle_acc_gain_pp:+.3f} pp | "
            f"{row.oracle_f1_at_25_gain_pp:+.3f} pp |"
        )
    lines.extend(["", "## Pre-registered process-feature gate", ""])
    for variant, decision in decisions["variants"].items():
        verdict = "PASS" if decision["passes_pilot_gate"] else "NO-GO"
        lines.append(
            f"- **{variant}: {verdict}.** Outer error-recall delta vs base "
            f"{decision['outer_error_recall_delta_pp']:+.3f} pp; positive internal folds "
            f"{decision['positive_internal_fold_count']}/3; oracle ΔAcc vs base "
            f"{decision['outer_oracle_acc_gain_delta_pp']:+.3f} pp; oracle ΔF1@25 vs base "
            f"{decision['outer_oracle_f1_at_25_gain_delta_pp']:+.3f} pp."
        )
    scale = decisions["scale_to_all_four_outer_folds"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "At least one process augmentation passed; proceed to all four official outer folds."
                if scale
                else "Neither process augmentation passed; do not spend four-fold compute on process features."
            ),
            "",
            "Oracle corrections replace selected frames with ground truth and are diagnostic ceilings only. "
            "They are not deployable predictions. Raw argmax results are a sensitivity analysis; official "
            "DiffAct is primary.",
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
    process_audit: pd.DataFrame,
) -> pd.DataFrame:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for mode in MODES:
        oof = [case for case in cases if case.mode == mode and case.scope == "oof_validation"]
        outer = [case for case in cases if case.mode == mode and case.scope == "outer_test"]
        add(
            f"{mode}: OOF covers outer train exactly once",
            len(oof) == 1460 and len({case.case_id for case in oof}) == 1460,
            f"rows={len(oof)}, unique={len({case.case_id for case in oof})}",
        )
        add(
            f"{mode}: outer test coverage",
            len(outer) == 252 and len({case.case_id for case in outer}) == 252,
            f"rows={len(outer)}, unique={len({case.case_id for case in outer})}",
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
    score_columns = [f"{variant}_score" for variant in VARIANT_PROCESS_KIND]
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
        "route Petri models persisted",
        len(process_audit) == 40
        and all(Path(path).is_file() for path in process_audit["petri_model_path"]),
        f"models={len(process_audit)} (expected 4 contexts x 10 tasks)",
    )
    add(
        "outer baseline cardinality",
        bool(
            (
                baselines[
                    (baselines.scope == "official_outer_test")
                    & (baselines["mode"] == "official")
                ].iloc[0].n_cases
                == metadata["outer_test_case_count"]
            )
        ),
        f"expected={metadata['outer_test_case_count']}",
    )
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise AssertionError(f"Analysis validation failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    out_dir = (args.out_dir or (study_dir / "analysis" / "video_selector_v1")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_json(study_dir / "study_metadata.json")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported study protocol: {metadata.get('protocol_version')}")
    diffact_root = Path(metadata["diffact_root"])
    verify_source_digest(metadata, diffact_root)
    data_root = Path(metadata["data_root"])
    task_payload = load_json(study_dir / "tasks.json")
    tasks = task_payload["tasks"]
    id_to_name, name_to_id = parse_mapping(data_root / DATASET / "mapping.txt")

    print("[1/5] Discovering route-conditioned train-only process models", flush=True)
    contexts, process_rows = build_feature_contexts(
        study_dir, metadata, data_root, name_to_id, out_dir
    )
    process_audit = pd.DataFrame(process_rows)
    process_audit.to_csv(out_dir / "process_model_audit.csv", index=False)

    print("[2/5] Building OOF and official outer-test segment features", flush=True)
    cases, segments, case_inventory, categorical = build_segment_rows(
        study_dir,
        metadata,
        tasks,
        args.modes,
        data_root,
        id_to_name,
        name_to_id,
        contexts,
    )
    feature_sets = variant_features(categorical)
    segments.to_csv(out_dir / "segment_features_unscored.csv", index=False)
    case_inventory.to_csv(out_dir / "case_inventory.csv", index=False)

    print("[3/5] Cross-fitting fixed selectors and scoring untouched outer fold 1", flush=True)
    scored, model_audit = fit_selector_scores(segments, feature_sets, out_dir)
    scored.to_csv(out_dir / "segment_scores.csv", index=False)
    model_audit.to_csv(out_dir / "selector_model_audit.csv", index=False)

    print("[4/5] Evaluating complete-span selections at matched frame budgets", flush=True)
    baselines, budget_metrics, discrimination = evaluate_selectors(
        cases, scored, list(VARIANT_PROCESS_KIND)
    )
    baselines.to_csv(out_dir / "baseline_metrics.csv", index=False)
    budget_metrics.to_csv(out_dir / "selector_budget_metrics.csv", index=False)
    discrimination.to_csv(out_dir / "selector_discrimination.csv", index=False)
    decisions = make_pilot_decision(budget_metrics)
    atomic_write_json(out_dir / "pilot_decision.json", decisions)

    print("[5/5] Validating and writing the decision readout", flush=True)
    checks = validate_outputs(
        metadata,
        cases,
        scored,
        model_audit,
        baselines,
        budget_metrics,
        process_audit,
    )
    checks.to_csv(out_dir / "validation_checks.csv", index=False)
    write_findings(out_dir, baselines, budget_metrics, decisions)
    run_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "out_dir": str(out_dir),
        "source_digest": metadata["source_provenance"]["source_digest"],
        "modes": args.modes,
        "variants": VARIANT_PROCESS_KIND,
        "frame_budgets": FRAME_BUDGETS,
        "long_substitution_definition": {
            "minimum_contiguous_error_length": LONG_MIN_LEN,
            "minimum_predicted_label_homogeneity": LONG_HOMOGENEITY,
        },
        "selector_hyperparameters": selector_model().get_params(),
        "feature_sets": feature_sets,
        "primary_mode": "official",
        "oracle_outputs_are_diagnostic_only": True,
        "validation_passed": True,
    }
    atomic_write_json(out_dir / "run_metadata.json", run_metadata)
    print(f"Analysis complete: {out_dir}")
    print((out_dir / "findings.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
