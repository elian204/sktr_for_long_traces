#!/usr/bin/env python3
"""Run honest low-data Petri-net postprocessing on exported DiffAct softmax."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from low_data_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPERIMENT_DIR,
    FRACTIONS,
    SEEDS,
    WORKSPACE_ROOT,
    collapse_runs,
    load_ground_truth_indices,
    load_mapping,
    one_hot_softmax,
    read_case_manifest,
)

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.calibration import calibrate_probabilities, softmax_numpy  # noqa: E402
from src.evaluation import compute_tas_metrics_asformer  # noqa: E402
from src.incremental_softmax_recovery import incremental_softmax_recovery, setup_logging  # noqa: E402
from src.utils import inverse_softmax, linear_prob_combiner  # noqa: E402


HP_STRATEGIES = {
    "trigram_heavy": [0.1, 0.15, 0.75],
    "unigram_super_heavy": [0.75, 0.15, 0.1],
}

DATASET_HP_DEFAULTS = {
    ("50salads", "diffact"): {"alpha": 0.3, "strategy": "unigram_super_heavy"},
    ("gtea", "diffact"): {"alpha": 0.95, "strategy": "trigram_heavy"},
    ("breakfast", "diffact"): {"alpha": 0.7, "strategy": "trigram_heavy"},
}

PAPER_DIFFACT_SKTR_METHOD = "petri_conformance"
PAPER_DIFFACT_SKTR_CHUNK_SIZE = 11
PAPER_DIFFACT_SKTR_WORKERS = 7
PAPER_DIFFACT_SKTR_PROGRESS_CHUNKS = 20
PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION = "frame_events"


def load_softmax_case_map(softmax_dir: Path) -> Dict[str, str]:
    case_map_path = softmax_dir / "case_id_map.csv"
    if case_map_path.is_file():
        case_map = pd.read_csv(case_map_path)
        return {
            str(row["original_case_id"]): str(row["sequential_case_id"])
            for _, row in case_map.iterrows()
        }

    video_index_path = softmax_dir / "video_index_map.txt"
    if video_index_path.is_file():
        mapping: Dict[str, str] = {}
        with video_index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"Malformed video index row in {video_index_path}: {line!r}")
                sequential_case_id, original_case_id = parts
                mapping[str(original_case_id)] = str(sequential_case_id)
        return mapping

    raise FileNotFoundError(
        f"Missing softmax case map: expected {case_map_path} or {video_index_path}"
    )


def apply_temperature_to_softmax_list(
    softmax_list: Sequence[np.ndarray],
    temperature: float,
) -> List[np.ndarray]:
    calibrated: List[np.ndarray] = []
    for mat in softmax_list:
        logits = inverse_softmax(np.asarray(mat, dtype=np.float64))
        calibrated.append(softmax_numpy(logits / float(temperature), axis=0).astype(np.float32))
    return calibrated


def build_eval_df_and_softmax(
    *,
    data_root: Path,
    dataset: str,
    eval_cases: Sequence[str],
    eval_softmax_dir: Path,
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    label_to_idx, labels = load_mapping(data_root / dataset / "mapping.txt")
    n_classes = len(labels)
    eval_case_to_source = load_softmax_case_map(eval_softmax_dir)

    rows: List[Dict[str, str]] = []
    softmax_list: List[np.ndarray] = []
    for seq_id, case_id in enumerate(eval_cases):
        source_case = eval_case_to_source.get(str(case_id))
        if source_case is None:
            raise ValueError(f"{case_id}: missing from softmax case map in {eval_softmax_dir}")
        labels_idx = load_ground_truth_indices(data_root, dataset, case_id, label_to_idx)
        mat_path = eval_softmax_dir / f"{source_case}.npy"
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing eval softmax matrix: {mat_path}")
        mat = np.load(mat_path)
        if mat.shape[0] != n_classes:
            raise ValueError(f"{mat_path}: classes={mat.shape[0]} expected {n_classes}")
        if mat.shape[1] != len(labels_idx):
            raise ValueError(f"{case_id}: softmax T={mat.shape[1]} ground truth T={len(labels_idx)}")
        seq_s = str(seq_id)
        for lab in labels_idx:
            rows.append({"case:concept:name": seq_s, "concept:name": str(lab)})
        softmax_list.append(np.asarray(mat, dtype=np.float32))
    return pd.DataFrame(rows), softmax_list


def learn_temperature_from_validation(
    *,
    data_root: Path,
    dataset: str,
    experiment_dir: Path,
    seed: int,
    fraction: int,
    temp_bounds: Tuple[float, float],
) -> float:
    val_cases = read_case_manifest(experiment_dir / "manifests" / "val_cases.txt")
    val_softmax_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}" / "softmax_val"
    if not val_softmax_dir.is_dir():
        raise FileNotFoundError(
            f"Validation softmax is required for calibration but was not found: {val_softmax_dir}"
        )
    val_df, val_softmax = build_eval_df_and_softmax(
        data_root=data_root,
        dataset=dataset,
        eval_cases=val_cases,
        eval_softmax_dir=val_softmax_dir,
    )
    temperature = calibrate_probabilities(
        softmax_list=val_softmax,
        df=val_df,
        temp_bounds=temp_bounds,
        only_return_temperature=True,
    )
    return float(temperature)


def iter_requested(seeds: Iterable[int], fractions: Iterable[int], max_runs: Optional[int]):
    count = 0
    for seed in seeds:
        for fraction in fractions:
            if max_runs is not None and count >= max_runs:
                return
            count += 1
            yield seed, fraction


def levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ai in enumerate(a, start=1):
        cur[0] = i
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[-1]


def build_combined_inputs(
    *,
    data_root: Path,
    dataset: str,
    train_cases: Sequence[str],
    eval_cases: Sequence[str],
    eval_softmax_dir: Path,
    out_dir: Path,
    temperature: Optional[float] = None,
    petri_discovery_representation: str = "run_collapsed_segments",
) -> Tuple[pd.DataFrame, List[np.ndarray], List[str], List[str], Path]:
    if petri_discovery_representation not in {"run_collapsed_segments", "frame_events"}:
        raise ValueError(
            "petri_discovery_representation must be one of "
            "{'run_collapsed_segments', 'frame_events'}"
        )
    label_to_idx, labels = load_mapping(data_root / dataset / "mapping.txt")
    n_classes = len(labels)
    rows: List[Dict[str, str]] = []
    softmax_list: List[np.ndarray] = []
    train_seq_ids: List[str] = []
    eval_seq_ids: List[str] = []
    mapping_rows: List[Dict[str, Any]] = []
    train_input_events = 0
    train_original_frames = 0

    for seq_id, case_id in enumerate(train_cases):
        full_labels_idx = load_ground_truth_indices(data_root, dataset, case_id, label_to_idx)
        train_original_frames += len(full_labels_idx)
        if petri_discovery_representation == "run_collapsed_segments":
            labels_idx = list(collapse_runs(full_labels_idx))
        else:
            labels_idx = full_labels_idx
        train_input_events += len(labels_idx)
        seq_s = str(seq_id)
        train_seq_ids.append(seq_s)
        for lab in labels_idx:
            rows.append({"case:concept:name": seq_s, "concept:name": str(lab)})
        softmax_list.append(one_hot_softmax(labels_idx, n_classes))
        mapping_rows.append(
            {
                "role": "train",
                "original_case_id": case_id,
                "sequential_case_id": seq_s,
                "softmax_list_index": seq_id,
                "source_softmax_case_id": "",
                "original_frame_events": len(full_labels_idx),
                "petri_discovery_events": len(labels_idx),
            }
        )

    offset = len(train_cases)
    eval_case_to_source = load_softmax_case_map(eval_softmax_dir)
    eval_frame_events = 0

    for j, case_id in enumerate(eval_cases):
        seq_id = offset + j
        seq_s = str(seq_id)
        source_case = eval_case_to_source.get(str(case_id))
        if source_case is None:
            raise ValueError(f"{case_id}: missing from softmax case map in {eval_softmax_dir}")
        labels_idx = load_ground_truth_indices(data_root, dataset, case_id, label_to_idx)
        mat_path = eval_softmax_dir / f"{source_case}.npy"
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing eval softmax matrix: {mat_path}")
        mat = np.load(mat_path)
        if mat.shape[0] != n_classes:
            raise ValueError(f"{mat_path}: classes={mat.shape[0]} expected {n_classes}")
        if mat.shape[1] != len(labels_idx):
            raise ValueError(f"{case_id}: softmax T={mat.shape[1]} ground truth T={len(labels_idx)}")
        if temperature is not None:
            mat = apply_temperature_to_softmax_list([mat], temperature)[0]
        eval_seq_ids.append(seq_s)
        eval_frame_events += len(labels_idx)
        for lab in labels_idx:
            rows.append({"case:concept:name": seq_s, "concept:name": str(lab)})
        softmax_list.append(np.asarray(mat, dtype=np.float32))
        mapping_rows.append(
            {
                "role": "eval",
                "original_case_id": case_id,
                "sequential_case_id": seq_s,
                "softmax_list_index": seq_id,
                "source_softmax_case_id": source_case,
                "original_frame_events": len(labels_idx),
                "petri_discovery_events": "",
            }
        )

    df = pd.DataFrame(rows)
    map_path = out_dir / "combined_case_id_map.csv"
    pd.DataFrame(mapping_rows).to_csv(map_path, index=False)

    checks = {
        "n_train_cases": len(train_cases),
        "n_eval_cases": len(eval_cases),
        "n_combined_cases": len(train_cases) + len(eval_cases),
        "n_softmax_matrices": len(softmax_list),
        "all_softmax_lengths_match_ground_truth": True,
        "class_count": n_classes,
        "class_mapping_path": str(data_root / dataset / "mapping.txt"),
        "softmax_source_dir": str(eval_softmax_dir),
        "temperature_applied": temperature,
        "petri_discovery_representation": petri_discovery_representation,
        "train_original_frame_events": train_original_frames,
        "train_petri_discovery_events": train_input_events,
        "eval_frame_events": eval_frame_events,
        "eval_postprocessing_representation": "frame_events",
    }
    (out_dir / "alignment_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return df, softmax_list, train_seq_ids, eval_seq_ids, map_path


def allowed_transition_set(df: pd.DataFrame, train_cases: Sequence[str]) -> set[tuple[str, str]]:
    train_set = set(str(c) for c in train_cases)
    allowed: set[tuple[str, str]] = set()
    for case, g in df[df["case:concept:name"].isin(train_set)].groupby("case:concept:name", sort=False):
        seq = [str(x) for x in g["concept:name"].tolist()]
        collapsed = collapse_runs(seq)
        allowed.update((str(a), str(b)) for a, b in zip(collapsed, collapsed[1:]))
    return allowed


def illegal_transition_pct(
    results_df: pd.DataFrame,
    pred_col: str,
    allowed: set[tuple[str, str]],
) -> float:
    illegal = 0
    total = 0
    for _, g in results_df.groupby("case:concept:name", sort=False):
        collapsed = collapse_runs([str(x) for x in g[pred_col].tolist()])
        for a, b in zip(collapsed, collapsed[1:]):
            total += 1
            if (str(a), str(b)) not in allowed:
                illegal += 1
    return 100.0 * illegal / total if total else 0.0


def transition_viterbi_decode(
    softmax_matrix: np.ndarray,
    allowed_transitions: set[tuple[str, str]],
    *,
    illegal_penalty: float,
    eps: float,
) -> Tuple[List[str], List[float]]:
    """Decode all frames with a soft penalty for run-to-run transitions absent from training."""
    probs = np.asarray(softmax_matrix, dtype=np.float64)
    n_classes, n_steps = probs.shape
    log_probs = np.log(np.maximum(probs, eps))

    transition_score = np.full((n_classes, n_classes), -float(illegal_penalty), dtype=np.float64)
    for i in range(n_classes):
        transition_score[i, i] = 0.0
    for src, dst in allowed_transitions:
        try:
            transition_score[int(src), int(dst)] = 0.0
        except (TypeError, ValueError):
            continue

    scores = np.empty((n_classes, n_steps), dtype=np.float64)
    backptr = np.zeros((n_classes, n_steps), dtype=np.int16)
    scores[:, 0] = log_probs[:, 0]
    for t in range(1, n_steps):
        candidates = scores[:, t - 1][:, None] + transition_score
        backptr[:, t] = np.argmax(candidates, axis=0).astype(np.int16)
        scores[:, t] = candidates[backptr[:, t], np.arange(n_classes)] + log_probs[:, t]

    path = np.empty(n_steps, dtype=np.int16)
    path[-1] = int(np.argmax(scores[:, -1]))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backptr[path[t], t]

    preds = [str(int(x)) for x in path]
    move_costs = [0.0]
    for prev, cur in zip(preds, preds[1:]):
        if prev == cur or (prev, cur) in allowed_transitions:
            move_costs.append(0.0)
        else:
            move_costs.append(float(illegal_penalty))
    return preds, move_costs


def build_recovery_records(
    *,
    case_id: str,
    sktr_preds: Sequence[str],
    argmax_preds: Sequence[str],
    ground_truth_sequence: Sequence[str],
    softmax_matrix: np.ndarray,
    sktr_move_costs: Sequence[float],
    prob_threshold: float,
    postprocess_seconds: float,
) -> Tuple[pd.DataFrame, float, float]:
    seq_len = len(ground_truth_sequence)
    if len(sktr_preds) != seq_len or len(argmax_preds) != seq_len:
        raise ValueError(
            f"{case_id}: prediction length mismatch "
            f"sktr={len(sktr_preds)} argmax={len(argmax_preds)} gt={seq_len}"
        )
    sktr_array = np.asarray(sktr_preds, dtype=str)
    argmax_array = np.asarray(argmax_preds, dtype=str)
    gt_array = np.asarray(ground_truth_sequence, dtype=str)
    sktr_matches = sktr_array == gt_array
    argmax_matches = argmax_array == gt_array

    all_probs: List[List[float]] = []
    all_activities: List[List[str]] = []
    for t in range(seq_len):
        probs = softmax_matrix[:, t]
        keep = np.where(probs >= prob_threshold)[0]
        all_probs.append([round(float(probs[i]), 2) for i in keep])
        all_activities.append([str(int(i)) for i in keep])

    records = pd.DataFrame(
        {
            "case:concept:name": [case_id] * seq_len,
            "step": list(range(seq_len)),
            "sktr_activity": list(sktr_preds),
            "argmax_activity": list(argmax_preds),
            "ground_truth": list(ground_truth_sequence),
            "all_probs": all_probs,
            "all_activities": all_activities,
            "is_correct": sktr_matches,
            "cumulative_accuracy": np.cumsum(sktr_matches) / np.arange(1, seq_len + 1),
            "sktr_move_cost": [round(float(x), 4) for x in sktr_move_costs],
            "postprocess_seconds": [float(postprocess_seconds)] * seq_len,
        }
    )
    return records, float(np.mean(sktr_matches)), float(np.mean(argmax_matches))


def write_case_output_atomic(records_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    records_df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def run_transition_viterbi_recovery(
    *,
    df: pd.DataFrame,
    softmax_list: Sequence[np.ndarray],
    eval_seq_ids: Sequence[str],
    allowed_transitions: set[tuple[str, str]],
    prob_threshold: float,
    illegal_penalty: float,
    case_output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    all_records: List[pd.DataFrame] = []
    sktr_accs: List[float] = []
    argmax_accs: List[float] = []
    case_output_dir.mkdir(parents=True, exist_ok=True)
    for seq_case in eval_seq_ids:
        seq_s = str(seq_case)
        case_start = time.perf_counter()
        case_df = df[df["case:concept:name"].astype(str) == seq_s]
        if case_df.empty:
            raise ValueError(f"{seq_s}: no evaluation rows found")
        mat = np.asarray(softmax_list[int(seq_s)], dtype=np.float32)
        gt = [str(x) for x in case_df["concept:name"].tolist()]
        argmax_preds = [str(int(x)) for x in np.argmax(mat, axis=0)]
        sktr_preds, move_costs = transition_viterbi_decode(
            mat,
            allowed_transitions,
            illegal_penalty=illegal_penalty,
            eps=prob_threshold,
        )
        records, sktr_acc, argmax_acc = build_recovery_records(
            case_id=seq_s,
            sktr_preds=sktr_preds,
            argmax_preds=argmax_preds,
            ground_truth_sequence=gt,
            softmax_matrix=mat,
            sktr_move_costs=move_costs,
            prob_threshold=prob_threshold,
            postprocess_seconds=time.perf_counter() - case_start,
        )
        write_case_output_atomic(records, case_output_dir / f"{seq_s}.csv")
        all_records.append(records)
        sktr_accs.append(sktr_acc)
        argmax_accs.append(argmax_acc)

    results_df = pd.concat(all_records, ignore_index=True)
    accuracy_dict = {"sktr_accuracy": sktr_accs, "argmax_accuracy": argmax_accs}
    return results_df, accuracy_dict


def mean_run_collapsed_distance(results_df: pd.DataFrame, pred_col: str) -> float:
    distances: List[int] = []
    for _, g in results_df.groupby("case:concept:name", sort=False):
        gt = collapse_runs([str(x) for x in g["ground_truth"].tolist()])
        pred = collapse_runs([str(x) for x in g[pred_col].tolist()])
        distances.append(levenshtein_distance(pred, gt))
    return float(np.mean(distances)) if distances else 0.0


def run_collapsed_distance_for_case(case_df: pd.DataFrame, pred_col: str) -> int:
    gt = collapse_runs([str(x) for x in case_df["ground_truth"].tolist()])
    pred = collapse_runs([str(x) for x in case_df[pred_col].tolist()])
    return levenshtein_distance(pred, gt)


def build_per_case_metric_rows(
    *,
    dataset: str,
    seed: int,
    fraction: int,
    condition: str,
    split: str,
    post_method: str,
    results_df: pd.DataFrame,
    combined_map_path: Path,
    mapping_path: Path,
    allowed_transitions: set[tuple[str, str]],
    calibrated: bool,
) -> pd.DataFrame:
    map_df = pd.read_csv(combined_map_path)
    seq_to_original = {
        str(row["sequential_case_id"]): str(row["original_case_id"])
        for _, row in map_df[map_df["role"] == "eval"].iterrows()
    }
    rows: List[Dict[str, Any]] = []
    method_map = {
        "raw_argmax": "argmax_activity",
        post_method: "sktr_activity",
    }
    for seq_case, case_df in results_df.groupby("case:concept:name", sort=False):
        seq_case_s = str(seq_case)
        original_case = seq_to_original.get(seq_case_s, seq_case_s)
        for method, pred_col in method_map.items():
            metrics = compute_tas_metrics_asformer(
                case_df,
                pred_col=pred_col,
                gt_col="ground_truth",
                case_col="case:concept:name",
                dataset_name=dataset,
                mapping_path=mapping_path,
            )
            metric_values = {
                "frame_accuracy": metrics["acc"],
                "edit": metrics["edit"],
                "f1@10": metrics["f1@10"],
                "f1@25": metrics["f1@25"],
                "f1@50": metrics["f1@50"],
                "run_collapsed_levenshtein": float(run_collapsed_distance_for_case(case_df, pred_col)),
                "illegal_transition_pct": illegal_transition_pct(case_df, pred_col, allowed_transitions),
            }
            if method == "petri_conformance":
                metric_values["alignment_cost"] = float(case_df["sktr_move_cost"].sum())
            elif method != "raw_argmax" and "sktr_move_cost" in case_df.columns:
                metric_values["transition_penalty_cost"] = float(case_df["sktr_move_cost"].sum())
            for metric_name, metric_value in metric_values.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "fraction": fraction,
                        "condition": condition,
                        "split": split,
                        "method": method,
                        "calibrated": bool(calibrated),
                        "original_case_id": original_case,
                        "sequential_case_id": seq_case_s,
                        "n_frames": int(len(case_df)),
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                    }
                )
    return pd.DataFrame(rows)


def build_metric_rows(
    *,
    dataset: str,
    seed: int,
    fraction: int,
    condition: str,
    split: str,
    post_method: str,
    results_df: pd.DataFrame,
    mapping_path: Path,
    allowed_transitions: set[tuple[str, str]],
    total_seconds: float,
    calibrated: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    method_map = {
        "raw_argmax": "argmax_activity",
        post_method: "sktr_activity",
    }
    for method, pred_col in method_map.items():
        metrics = compute_tas_metrics_asformer(
            results_df,
            pred_col=pred_col,
            gt_col="ground_truth",
            case_col="case:concept:name",
            dataset_name=dataset,
            mapping_path=mapping_path,
        )
        metric_values = {
            "frame_accuracy": metrics["acc"],
            "edit": metrics["edit"],
            "f1@10": metrics["f1@10"],
            "f1@25": metrics["f1@25"],
            "f1@50": metrics["f1@50"],
            "run_collapsed_levenshtein": mean_run_collapsed_distance(results_df, pred_col),
            "illegal_transition_pct": illegal_transition_pct(results_df, pred_col, allowed_transitions),
        }
        if method == "petri_conformance":
            metric_values["mean_alignment_cost"] = float(results_df["sktr_move_cost"].mean())
            metric_values["total_alignment_cost"] = float(results_df["sktr_move_cost"].sum())
        elif method != "raw_argmax" and "sktr_move_cost" in results_df.columns:
            metric_values["mean_transition_penalty"] = float(results_df["sktr_move_cost"].mean())
            metric_values["total_transition_penalty"] = float(results_df["sktr_move_cost"].sum())
        for metric_name, metric_value in metric_values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fraction": fraction,
                    "condition": condition,
                    "split": split,
                    "method": method,
                    "calibrated": bool(calibrated),
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "n_cases": int(results_df["case:concept:name"].nunique()),
                    "total_seconds": float(total_seconds),
                }
            )
    return pd.DataFrame(rows)


def build_runtime_tables(
    *,
    seed: int,
    fraction: int,
    condition: str,
    split: str,
    method: str,
    results_df: pd.DataFrame,
    combined_map_path: Path,
    total_seconds: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_cases = max(1, results_df["case:concept:name"].nunique())
    per_case_rows: List[Dict[str, Any]] = []
    if "postprocess_seconds" in results_df.columns:
        map_df = pd.read_csv(combined_map_path)
        seq_to_original = {
            str(row["sequential_case_id"]): str(row["original_case_id"])
            for _, row in map_df[map_df["role"] == "eval"].iterrows()
        }
        for seq_case, group in results_df.groupby("case:concept:name", sort=False):
            seconds = pd.to_numeric(group["postprocess_seconds"], errors="coerce").dropna()
            if seconds.empty:
                continue
            seq_case_s = str(seq_case)
            per_case_rows.append(
                {
                    "seed": seed,
                    "fraction": fraction,
                    "condition": condition,
                    "method": method,
                    "split": split,
                    "original_case_id": seq_to_original.get(seq_case_s, seq_case_s),
                    "sequential_case_id": seq_case_s,
                    "n_frames": int(len(group)),
                    "seconds": float(seconds.iloc[0]),
                }
            )

    per_case_runtime = pd.DataFrame(
        per_case_rows,
        columns=[
            "seed",
            "fraction",
            "condition",
            "method",
            "split",
            "original_case_id",
            "sequential_case_id",
            "n_frames",
            "seconds",
        ],
    )
    if per_case_runtime.empty:
        mean_seconds = float(total_seconds / n_cases)
        median_seconds = float("nan")
    else:
        mean_seconds = float(per_case_runtime["seconds"].mean())
        median_seconds = float(per_case_runtime["seconds"].median())

    runtime_df = pd.DataFrame(
        [
            {
                "seed": seed,
                "fraction": fraction,
                "condition": condition,
                "method": method,
                "split": split,
                "total_seconds": float(total_seconds),
                "mean_seconds_per_case": mean_seconds,
                "median_seconds_per_case": median_seconds,
            }
        ]
    )
    return runtime_df, per_case_runtime


def validation_selection_score(metrics_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    petri = metrics_df[
        (metrics_df["method"] != "raw_argmax")
        & (metrics_df["metric_name"].isin(["edit", "f1@10", "f1@25", "f1@50"]))
    ]
    values = {
        str(row["metric_name"]): float(row["metric_value"])
        for _, row in petri.iterrows()
    }
    score_terms = [values[name] for name in ("edit", "f1@10", "f1@25", "f1@50") if name in values]
    score = float(np.mean(score_terms)) if score_terms else float("-inf")
    return score, values


def select_validation_hyperparameters(
    *,
    experiment_dir: Path,
    condition: str,
    seed: int,
    fraction: int,
    method: str,
) -> Optional[Dict[str, Any]]:
    """Select Petri postprocessing parameters from validation metrics only."""
    run_root = experiment_dir / "petri" / condition / f"seed_{seed}" / f"frac_{fraction}"
    if not run_root.exists():
        return None
    candidates: List[Dict[str, Any]] = []
    metric_paths = sorted(set(run_root.glob("val*/metrics.csv")) | set(run_root.glob("*/val*/metrics.csv")))
    for metrics_path in metric_paths:
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            continue
        config_path = metrics_path.parent / "postprocess_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        if str(config.get("method")) != str(method):
            continue
        score, score_metrics = validation_selection_score(metrics_df)
        candidates.append(
            {
                "candidate_id": metrics_path.parent.name,
                "metrics_path": str(metrics_path),
                "config_path": str(config_path),
                "score": score,
                "score_metrics": score_metrics,
                "method": config.get("method"),
                "chunk_size": config.get("chunk_size"),
                "prob_threshold": config.get("prob_threshold"),
                "transition_illegal_penalty": config.get("transition_illegal_penalty"),
                "cost_function": config.get("cost_function"),
                "model_move_cost": config.get("model_move_cost"),
                "log_move_cost": config.get("log_move_cost"),
                "tau_move_cost": config.get("tau_move_cost"),
                "conformance_switch_penalty_weight": config.get(
                    "conformance_switch_penalty_weight"
                ),
                "use_calibration": config.get("use_calibration"),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    selected = dict(candidates[0])
    payload = {
        "dataset_selection_scope": "validation_only",
        "selection_uses_test_data": False,
        "selection_metric": "mean(edit,f1@10,f1@25,f1@50)",
        "condition": condition,
        "seed": seed,
        "fraction": fraction,
        "n_candidates": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }
    selection_path = run_root / method / "selected_hyperparameters.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["selection_path"] = str(selection_path)
    return payload


def legacy_output_matches_method(out_dir: Path, method: str, calibrated: bool) -> bool:
    config_path = out_dir / "postprocess_config.json"
    metrics_path = out_dir / "metrics.csv"
    if not config_path.is_file() or not metrics_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return False
    return str(config.get("method")) == str(method) and bool(config.get("calibrated", False)) == bool(calibrated)


def resolve_postprocess_output_dir(
    *,
    experiment_dir: Path,
    condition: str,
    seed: int,
    fraction: int,
    method: str,
    split_dir: str,
    calibrated: bool,
) -> Path:
    run_root = experiment_dir / "petri" / condition / f"seed_{seed}" / f"frac_{fraction}"
    legacy = run_root / split_dir
    if legacy_output_matches_method(legacy, method, calibrated):
        return legacy
    return run_root / method / split_dir


def run_one(
    *,
    experiment_dir: Path,
    data_root: Path,
    dataset: str,
    seed: int,
    fraction: int,
    split: str,
    condition: str,
    method: str,
    chunk_size: int,
    prob_threshold: float,
    model_move_cost: float,
    conformance_switch_penalty_weight: float,
    progress_log_interval_chunks: int,
    petri_discovery_representation: str,
    transition_illegal_penalty: float,
    calibrated: bool,
    temp_bounds: Tuple[float, float],
    workers: int,
    inner_parallel: bool,
    force: bool,
) -> Optional[Path]:
    if calibrated and split != "test":
        raise ValueError("Validation-learned calibration is only evaluated on --split test")
    split_dir = f"{split}_calibrated" if calibrated else split
    out_dir = resolve_postprocess_output_dir(
        experiment_dir=experiment_dir,
        condition=condition,
        seed=seed,
        fraction=fraction,
        method=method,
        split_dir=split_dir,
        calibrated=calibrated,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    metrics_csv = out_dir / "metrics.csv"
    per_case_metrics_csv = out_dir / "per_case_metrics.csv"
    per_case_runtime_csv = out_dir / "per_case_runtime.csv"
    if (
        metrics_csv.is_file()
        and results_csv.is_file()
        and per_case_metrics_csv.is_file()
        and per_case_runtime_csv.is_file()
        and not force
    ):
        print(f"SKIP seed={seed} frac={fraction} split={split}: existing {metrics_csv}")
        return metrics_csv

    subset_cases = read_case_manifest(
        experiment_dir / "manifests" / f"seed_{seed}" / f"train_cases_frac_{fraction}.txt"
    )
    if condition == "structural_prior_full_train_petri":
        train_cases = read_case_manifest(experiment_dir / "manifests" / "train_pool_cases.txt")
    else:
        train_cases = subset_cases
    eval_cases = read_case_manifest(experiment_dir / "manifests" / f"{split}_cases.txt")
    softmax_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}" / f"softmax_{split}"
    if not softmax_dir.is_dir():
        print(f"MISSING softmax seed={seed} frac={fraction} split={split}: {softmax_dir}")
        return None

    learned_temperature: Optional[float] = None
    if calibrated:
        learned_temperature = learn_temperature_from_validation(
            data_root=data_root,
            dataset=dataset,
            experiment_dir=experiment_dir,
            seed=seed,
            fraction=fraction,
            temp_bounds=temp_bounds,
        )
        (out_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fraction": fraction,
                    "condition": condition,
                    "temperature": learned_temperature,
                    "temp_bounds": list(temp_bounds),
                    "fit_split": "val",
                    "applied_split": split,
                    "uses_test_for_fitting": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    validation_selection: Optional[Dict[str, Any]] = None
    if split == "test":
        validation_selection = select_validation_hyperparameters(
            experiment_dir=experiment_dir,
            condition=condition,
            seed=seed,
            fraction=fraction,
            method=method,
        )
        if validation_selection is None:
            raise FileNotFoundError(
                "Test postprocessing requires validation metrics for hyperparameter "
                f"selection with method={method}: "
                f"{experiment_dir / 'petri' / condition / f'seed_{seed}' / f'frac_{fraction}'}"
            )
        selected_hp = validation_selection["selected"]
        if selected_hp.get("method") is not None:
            method = str(selected_hp["method"])
        if selected_hp.get("chunk_size") is not None:
            chunk_size = int(selected_hp["chunk_size"])
        if selected_hp.get("prob_threshold") is not None:
            prob_threshold = float(selected_hp["prob_threshold"])
        if selected_hp.get("transition_illegal_penalty") is not None:
            transition_illegal_penalty = float(selected_hp["transition_illegal_penalty"])
        if selected_hp.get("model_move_cost") is not None:
            model_move_cost = float(selected_hp["model_move_cost"])
        if selected_hp.get("conformance_switch_penalty_weight") is not None:
            conformance_switch_penalty_weight = float(
                selected_hp["conformance_switch_penalty_weight"]
            )

    df, softmax_list, train_seq_ids, eval_seq_ids, map_path = build_combined_inputs(
        data_root=data_root,
        dataset=dataset,
        train_cases=train_cases,
        eval_cases=eval_cases,
        eval_softmax_dir=softmax_dir,
        out_dir=out_dir,
        temperature=learned_temperature,
        petri_discovery_representation=petri_discovery_representation,
    )
    allowed = allowed_transition_set(df, train_seq_ids)
    hp = DATASET_HP_DEFAULTS[(dataset, "diffact")]
    weights = HP_STRATEGIES[hp["strategy"]]
    cfg = {
        "method": method,
        "n_train_traces": None,
        "n_test_traces": None,
        "train_cases": train_seq_ids,
        "test_cases": eval_seq_ids,
        "ensure_train_variant_diversity": False,
        "ensure_test_variant_diversity": False,
        "allow_train_cases_in_test": False,
        "use_same_traces_for_train_test": False,
        "compute_marking_transition_map": False,
        "sequential_sampling": False,
        "n_indices": 10**9,
        "n_per_run": None,
        "independent_sampling": True,
        "prob_threshold": prob_threshold,
        "chunk_size": chunk_size,
        "conformance_switch_penalty_weight": conformance_switch_penalty_weight,
        "conditioning_alpha": hp["alpha"],
        "conditioning_interpolation_weights": weights,
        "conditioning_combine_fn": linear_prob_combiner,
        "max_hist_len": 3,
        "conditioning_n_prev_labels": 3,
        "conditioning_state_mode": "topm",
        "conditioning_top_m": 1,
        "candidate_top_k": 3,
        "candidate_top_p": 1.0,
        "candidate_min_k": 1,
        "candidate_source": "conditioned",
        "candidate_apply_to_sync": True,
        "restrict_log_moves": True,
        "restrict_model_moves_to_tau": True,
        "max_consecutive_tau_moves": 8,
        "progress_log_interval_chunks": progress_log_interval_chunks,
        "transition_illegal_penalty": transition_illegal_penalty,
        "enabled_cache_size": 100000,
        "cost_function": "linear",
        "model_move_cost": model_move_cost,
        "log_move_cost": 1.0,
        "tau_move_cost": 1e-6,
        "non_sync_penalty": 1.0,
        "use_calibration": False,
        "use_collapsed_runs": True,
        "round_precision": 2,
        "random_seed": seed,
        "save_model": False,
        "save_model_path": None,
        "parallel_processing": False,
        "dataset_parallelization": inner_parallel,
        "dataset_parallelization_context": None,
        "max_workers": workers,
        "merge_mismatched_boundaries": False,
        "case_output_dir": str(out_dir / "case_outputs"),
        "resume_case_outputs": True,
        "verbose": True,
        "log_level": logging.INFO,
    }
    serializable_cfg = {
        k: (str(v) if callable(v) else v)
        for k, v in cfg.items()
        if k != "conditioning_combine_fn"
    }
    serializable_cfg["conditioning_combine_fn"] = "linear_prob_combiner"
    serializable_cfg.update(
        {
            "dataset": dataset,
            "seed": seed,
            "fraction": fraction,
            "split": split,
            "condition": condition,
            "train_original_cases": train_cases,
            "eval_original_cases": eval_cases,
            "combined_case_map": str(map_path),
            "softmax_dir": str(softmax_dir),
            "calibrated": calibrated,
            "learned_temperature": learned_temperature,
            "temp_bounds": list(temp_bounds),
            "calibration_fit_split": "val" if calibrated else None,
            "calibration_uses_test_data": False,
            "validation_selection": validation_selection,
            "main_condition_petri_discovery": (
                "same_fraction_training_subset"
                if condition == "honest_low_data_petri"
                else "full_training_pool_structural_prior"
            ),
            "no_event_level_subsampling": cfg["n_indices"] >= 10**9 and not cfg["sequential_sampling"],
            "petri_discovery_trace_representation": petri_discovery_representation,
            "eval_postprocessing_trace_representation": "frame_events",
            "paper_diffact_sktr_config": (
                method == PAPER_DIFFACT_SKTR_METHOD
                and chunk_size == PAPER_DIFFACT_SKTR_CHUNK_SIZE
                and workers == PAPER_DIFFACT_SKTR_WORKERS
                and petri_discovery_representation == PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION
                and cfg["restrict_log_moves"]
                and cfg["restrict_model_moves_to_tau"]
                and cfg["max_consecutive_tau_moves"] == 8
                and cfg["conditioning_state_mode"] == "topm"
                and cfg["conditioning_top_m"] == 1
                and cfg["candidate_top_k"] == 3
            ),
        }
    )
    (out_dir / "postprocess_config.json").write_text(
        json.dumps(serializable_cfg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if validation_selection is not None:
        (out_dir / "selected_hyperparameters.json").write_text(
            json.dumps(validation_selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    start = time.perf_counter()
    if method == "petri_conformance":
        recovery_cfg = {
            k: v
            for k, v in cfg.items()
            if k not in {"method", "transition_illegal_penalty"}
        }
        results_df, accuracy_dict, _ = incremental_softmax_recovery(
            df=df,
            softmax_lst=softmax_list,
            **recovery_cfg,
        )
    elif method == "petri_transition_viterbi":
        results_df, accuracy_dict = run_transition_viterbi_recovery(
            df=df,
            softmax_list=softmax_list,
            eval_seq_ids=eval_seq_ids,
            allowed_transitions=allowed,
            prob_threshold=prob_threshold,
            illegal_penalty=transition_illegal_penalty,
            case_output_dir=out_dir / "case_outputs",
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    total_seconds = time.perf_counter() - start
    results_df.to_csv(results_csv, index=False)
    (out_dir / "accuracy_dict.json").write_text(
        json.dumps(accuracy_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_df = build_metric_rows(
        dataset=dataset,
        seed=seed,
        fraction=fraction,
        condition=condition,
        split=split,
        post_method=method,
        results_df=results_df,
        mapping_path=data_root / dataset / "mapping.txt",
        allowed_transitions=allowed,
        total_seconds=total_seconds,
        calibrated=calibrated,
    )
    metric_df.to_csv(metrics_csv, index=False)
    per_case_df = build_per_case_metric_rows(
        dataset=dataset,
        seed=seed,
        fraction=fraction,
        condition=condition,
        split=split,
        post_method=method,
        results_df=results_df,
        combined_map_path=map_path,
        mapping_path=data_root / dataset / "mapping.txt",
        allowed_transitions=allowed,
        calibrated=calibrated,
    )
    per_case_df.to_csv(per_case_metrics_csv, index=False)
    method_name = f"{method}_calibrated" if calibrated else method
    runtime_df, per_case_runtime_df = build_runtime_tables(
        seed=seed,
        fraction=fraction,
        condition=condition,
        split=split,
        method=method_name,
        results_df=results_df,
        combined_map_path=map_path,
        total_seconds=total_seconds,
    )
    runtime_df.to_csv(out_dir / "runtime.csv", index=False)
    per_case_runtime_df.to_csv(per_case_runtime_csv, index=False)
    print(f"Wrote {metrics_csv}")
    return metrics_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--fractions", nargs="+", type=int, default=list(FRACTIONS))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--condition",
        choices=["honest_low_data_petri", "structural_prior_full_train_petri"],
        default="honest_low_data_petri",
    )
    parser.add_argument(
        "--method",
        choices=["petri_transition_viterbi", "petri_conformance"],
        default=PAPER_DIFFACT_SKTR_METHOD,
        help=(
            "Postprocessing method. petri_conformance is the exact alignment-based "
            "SKTR path used by the prior full-data DiffAct recovery."
        ),
    )
    parser.add_argument(
        "--transition-illegal-penalty",
        type=float,
        default=2.0,
        help="Soft Viterbi penalty for a non-self transition absent from the training Petri/variant relation.",
    )
    parser.add_argument("--chunk-size", type=int, default=PAPER_DIFFACT_SKTR_CHUNK_SIZE)
    parser.add_argument("--prob-threshold", type=float, default=1e-6)
    parser.add_argument("--model-move-cost", type=float, default=1.0)
    parser.add_argument("--conformance-switch-penalty-weight", type=float, default=1.0)
    parser.add_argument(
        "--progress-log-interval-chunks",
        type=int,
        default=PAPER_DIFFACT_SKTR_PROGRESS_CHUNKS,
        help=(
            "Log conformance progress every N chunks inside each case. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--petri-discovery-representation",
        choices=["run_collapsed_segments", "frame_events"],
        default=PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION,
        help=(
            "Representation used for Petri-net discovery from the training subset. "
            "The prior full-data DiffAct+SKTR run used frame_events. Evaluation "
            "traces are always postprocessed and scored at frame level."
        ),
    )
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--temp-bounds", nargs=2, type=float, default=[1.0, 10.0])
    parser.add_argument("--workers", type=int, default=PAPER_DIFFACT_SKTR_WORKERS)
    parser.add_argument(
        "--inner-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use SKTR dataset-level parallelization across evaluation cases. "
            "Enabled by default for the low-data experiment; use "
            "--no-inner-parallel for sequential case processing."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    data_root = args.data_root.resolve()
    metadata = json.loads((experiment_dir / "experiment_metadata.json").read_text())
    dataset = metadata["dataset"]
    setup_logging()

    commands: List[str] = []
    completed = 0
    for seed, fraction in iter_requested(args.seeds, args.fractions, args.max_runs):
        parallel_flag = "--inner-parallel" if args.inner_parallel else "--no-inner-parallel"
        command = (
            f"python {Path(__file__).resolve()} --experiment-dir {experiment_dir} "
            f"--data-root {data_root} --seeds {seed} --fractions {fraction} "
            f"--split {args.split} --condition {args.condition} "
            f"--method {args.method} --transition-illegal-penalty {args.transition_illegal_penalty} "
            f"--chunk-size {args.chunk_size} --prob-threshold {args.prob_threshold} "
            f"--model-move-cost {args.model_move_cost} "
            f"--conformance-switch-penalty-weight {args.conformance_switch_penalty_weight} "
            f"--progress-log-interval-chunks {args.progress_log_interval_chunks} "
            f"--petri-discovery-representation {args.petri_discovery_representation} "
            f"--workers {args.workers} {parallel_flag} --execute"
        )
        if args.calibrated:
            command += f" --calibrated --temp-bounds {args.temp_bounds[0]} {args.temp_bounds[1]}"
        commands.append(command)
        if not args.execute:
            print(f"PREPARED seed={seed} frac={fraction}: {command}")
            continue
        path = run_one(
            experiment_dir=experiment_dir,
            data_root=data_root,
            dataset=dataset,
            seed=seed,
            fraction=fraction,
            split=args.split,
            condition=args.condition,
            method=args.method,
            chunk_size=args.chunk_size,
            prob_threshold=args.prob_threshold,
            model_move_cost=args.model_move_cost,
            conformance_switch_penalty_weight=args.conformance_switch_penalty_weight,
            progress_log_interval_chunks=args.progress_log_interval_chunks,
            petri_discovery_representation=args.petri_discovery_representation,
            transition_illegal_penalty=args.transition_illegal_penalty,
            calibrated=args.calibrated,
            temp_bounds=(float(args.temp_bounds[0]), float(args.temp_bounds[1])),
            workers=args.workers,
            inner_parallel=args.inner_parallel,
            force=args.force,
        )
        if path is not None:
            completed += 1

    cal_suffix = "_calibrated" if args.calibrated else ""
    cmd_path = experiment_dir / "petri" / f"run_{args.condition}_{args.split}{cal_suffix}_commands.sh"
    cmd_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n")
    cmd_path.chmod(0o755)
    print(f"Completed Petri runs: {completed}")
    print(f"Petri commands: {cmd_path}")
    if not args.execute:
        print("No Petri postprocessing was launched; pass --execute after softmax exports exist.")


if __name__ == "__main__":
    main()
