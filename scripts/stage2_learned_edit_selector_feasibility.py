#!/usr/bin/env python3
"""
Stage-2.0 learned edit-selector feasibility.

Read-only diagnostic over existing DiffAct/SKTR artifacts. Generates a small,
auditable candidate-edit set, computes its GT oracle ceiling, then tests whether
a fold-pure learned selector can recover useful edits on held-out folds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage0_duration_confidence_diagnostics import (  # noqa: E402
    DATASETS,
    FoldContext,
    DurationPrior,
    build_duration_priors,
    get_folds,
    load_case_output,
    load_complete_case_set,
    load_fold_context,
    segments,
)
from stage1_boundary_duration_precheck import duration_score  # noqa: E402
from stage1a0_smoothed_crossing_precheck import (  # noqa: E402
    smoothed_crossing_boundary,
    smooth_softmax,
)
from src.cv_utils import DEFAULT_DATA_ROOT  # noqa: E402
from src.evaluation import compute_tas_metrics_from_sequences, tas_metrics  # noqa: E402


DEFAULT_OUT_DIR = "/data1/eli-bogdanov/sktr_runs/stage2_learned_edit_selector_v1"
BOUNDARY_DELTAS = [-25, -10, -5, 5, 10, 25]
ISLAND_MAX_LEN = 25
ISLAND_MARGIN_THRESHOLD = 0.10
UTILITY_TAU = 0.0
ACC_EPSILON = 0.10
SMOOTH_SIGMAS = [5.0, 11.0, 25.0]
EDIT_BUDGETS = [0.01, 0.03]
LOCK_THRESHOLDS = [0.95, 0.98]
PROBA_THRESHOLDS = [0.50, 0.70]
OUTLIER_CASES = {("50salads", 1, "1"), ("50salads", 5, "49")}
METRIC_KEYS = ["acc", "edit", "f1@10", "f1@25", "f1@50"]

CEILING_PATHS = {
    "gtea": [
        Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "sktr_ceiling_analysis_gtea_skip_all_v4/all_ceiling_cases.csv"
        )
    ],
    "breakfast": [
        Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "sktr_ceiling_breakfast_skip_all_v1/all_ceiling_cases.csv"
        )
    ],
    "50salads": [
        Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "sktr_ceiling_50salads_fold1_run_rerun_64gb_v1/"
            "completed_ceiling_cases.csv"
        ),
        Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "sktr_ceiling_50salads_isolated_v1/completed_ceiling_cases.csv"
        ),
    ],
}


@dataclass
class Candidate:
    candidate_id: str
    dataset: str
    fold: int
    case_id: str
    edit_type: str
    anchor_id: str
    start: int
    end: int
    old_label: str
    new_label: str
    delta: int
    features: Dict[str, Any]
    delta_metrics: Dict[str, float]
    utility: float
    helpful: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--rare-class-min-segments", type=int, default=5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Convenience mode: candidate/oracle smoke for GTEA fold 1.",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def int_label(label: str) -> int:
    try:
        return int(label)
    except Exception:
        return -1


def softmax_frame_features(mat: np.ndarray) -> Dict[str, np.ndarray]:
    probs = np.asarray(mat, dtype=float)
    top1 = probs.max(axis=0)
    if probs.shape[0] > 1:
        top2 = np.partition(probs, -2, axis=0)[-2]
    else:
        top2 = np.zeros(probs.shape[1])
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=0)
    return {
        "pmax": top1,
        "margin": top1 - top2,
        "entropy": entropy,
    }


def interval_stats(values: np.ndarray, start: int, end: int) -> Tuple[float, float, float]:
    if end <= start:
        return 0.0, 0.0, 0.0
    sub = values[start:end]
    return float(np.mean(sub)), float(np.min(sub)), float(np.max(sub))


def label_prob_sum(mat: np.ndarray, label: str, start: int, end: int) -> float:
    idx = int_label(label)
    if idx < 0 or idx >= mat.shape[0] or end <= start:
        return 0.0
    return float(np.log(np.clip(mat[idx, start:end], 1e-12, 1.0)).sum())


def apply_candidate_to_sequence(seq: Sequence[str], cand: Candidate) -> List[str]:
    out = [str(x) for x in seq]
    out[cand.start : cand.end] = [cand.new_label] * (cand.end - cand.start)
    return out


def apply_candidate_raw(
    seq: Sequence[str],
    start: int,
    end: int,
    new_label: str,
) -> List[str]:
    out = [str(x) for x in seq]
    out[start:end] = [str(new_label)] * (end - start)
    return out


def metric_deltas(gt: Sequence[str], base: Sequence[str], pred: Sequence[str]) -> Dict[str, float]:
    base_m = tas_metrics(gt, base)
    pred_m = tas_metrics(gt, pred)
    return {k: float(pred_m[k] - base_m[k]) for k in METRIC_KEYS}


def utility_from_deltas(delta: Dict[str, float]) -> float:
    acc = delta["acc"]
    return float(
        acc
        + 0.5 * delta["f1@50"]
        + 0.5 * delta["edit"]
        - 5.0 * max(0.0, -acc)
    )


def duration_pair_score(
    priors: Dict[str, DurationPrior],
    left_class: str,
    right_class: str,
    left_len: int,
    right_len: int,
    total_len: int,
    variant: str,
) -> float:
    if left_class not in priors or right_class not in priors:
        return 0.0
    return float(
        duration_score(priors[left_class], left_len, total_len, variant, 9.0)
        + duration_score(priors[right_class], right_len, total_len, variant, 9.0)
    )


def load_process_features(dataset: str) -> Dict[Tuple[int, str], Dict[str, float]]:
    paths = CEILING_PATHS.get(dataset, [])
    frames = []
    for path in paths:
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    out: Dict[Tuple[int, str], Dict[str, float]] = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        key = (int(d["fold"]), str(d["case_id"]))
        feats: Dict[str, float] = {}
        for col in [
            "argmax_log_moves",
            "argmax_model_moves",
            "sktr_log_moves",
            "sktr_model_moves",
        ]:
            feats[f"process_{col}"] = finite_float(d.get(col, 0.0))
        for col in [
            "argmax_accepted_exact",
            "argmax_accepted_exact_tau_completed",
            "sktr_accepted_exact",
            "sktr_accepted_exact_tau_completed",
        ]:
            feats[f"process_{col}"] = 1.0 if bool(d.get(col, False)) else 0.0
        feats["process_argmax_total_moves"] = (
            feats.get("process_argmax_log_moves", 0.0)
            + feats.get("process_argmax_model_moves", 0.0)
        )
        feats["process_sktr_total_moves"] = (
            feats.get("process_sktr_log_moves", 0.0)
            + feats.get("process_sktr_model_moves", 0.0)
        )
        out[key] = feats
    return out


def candidate_feature_base(
    *,
    ctx: FoldContext,
    case_id: str,
    n_frames: int,
    process_features: Dict[str, float],
) -> Dict[str, Any]:
    base = {
        "dataset_id": {"50salads": 0, "breakfast": 1, "gtea": 2}[ctx.dataset],
        "fold": int(ctx.fold),
        "case_id_numeric": int_label(case_id),
        "n_frames": float(n_frames),
    }
    base.update(process_features)
    return base


def make_boundary_candidate(
    *,
    ctx: FoldContext,
    case_id: str,
    gt: Sequence[str],
    argmax: Sequence[str],
    mat: np.ndarray,
    sf: Dict[str, np.ndarray],
    priors: Dict[str, DurationPrior],
    smooth_crossings: Dict[Tuple[int, float], int],
    process_features: Dict[str, float],
    seg_index: int,
    prev_b: int,
    b: int,
    next_b: int,
    left_class: str,
    right_class: str,
    delta: int,
) -> Optional[Candidate]:
    new_b = b + delta
    if new_b <= prev_b or new_b >= next_b:
        return None
    if delta > 0:
        start, end = b, new_b
        old_label, new_label = right_class, left_class
    else:
        start, end = new_b, b
        old_label, new_label = left_class, right_class
    if end <= start:
        return None

    pred = apply_candidate_raw(argmax, start, end, new_label)
    deltas = metric_deltas(gt, argmax, pred)
    utility = utility_from_deltas(deltas)
    n = len(argmax)
    old_lp = label_prob_sum(mat, old_label, start, end)
    new_lp = label_prob_sum(mat, new_label, start, end)
    mean_pmax, min_pmax, max_pmax = interval_stats(sf["pmax"], start, end)
    mean_margin, min_margin, max_margin = interval_stats(sf["margin"], start, end)
    mean_entropy, min_entropy, max_entropy = interval_stats(sf["entropy"], start, end)
    before_raw = duration_pair_score(
        priors, left_class, right_class, b - prev_b, next_b - b, n, "raw"
    )
    after_raw = duration_pair_score(
        priors, left_class, right_class, new_b - prev_b, next_b - new_b, n, "raw"
    )
    before_norm = duration_pair_score(
        priors, left_class, right_class, b - prev_b, next_b - b, n, "normalized"
    )
    after_norm = duration_pair_score(
        priors, left_class, right_class, new_b - prev_b, next_b - new_b, n, "normalized"
    )
    features = candidate_feature_base(
        ctx=ctx,
        case_id=case_id,
        n_frames=n,
        process_features=process_features,
    )
    features.update(
        {
            "edit_type_boundary": 1.0,
            "edit_type_island": 0.0,
            "delta": float(delta),
            "abs_delta": float(abs(delta)),
            "direction": float(1 if delta > 0 else -1),
            "changed_len": float(end - start),
            "changed_len_norm": float((end - start) / n),
            "anchor_index": float(seg_index),
            "anchor_pos_norm": float(b / n),
            "left_class": float(int_label(left_class)),
            "right_class": float(int_label(right_class)),
            "old_label": float(int_label(old_label)),
            "new_label": float(int_label(new_label)),
            "same_neighbor_aba": 0.0,
            "left_len_before": float(b - prev_b),
            "right_len_before": float(next_b - b),
            "left_len_after": float(new_b - prev_b),
            "right_len_after": float(next_b - new_b),
            "min_len_after": float(min(new_b - prev_b, next_b - new_b)),
            "logprob_delta_sum": float(new_lp - old_lp),
            "logprob_delta_per_frame": float((new_lp - old_lp) / (end - start)),
            "mean_pmax": mean_pmax,
            "min_pmax": min_pmax,
            "max_pmax": max_pmax,
            "mean_margin": mean_margin,
            "min_margin": min_margin,
            "max_margin": max_margin,
            "mean_entropy": mean_entropy,
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "high_conf_frac_095": float(np.mean(sf["pmax"][start:end] >= 0.95)),
            "high_conf_frac_098": float(np.mean(sf["pmax"][start:end] >= 0.98)),
            "duration_raw_delta": float(after_raw - before_raw),
            "duration_norm_delta": float(after_norm - before_norm),
        }
    )
    for sigma in SMOOTH_SIGMAS:
        b_smooth = smooth_crossings.get((seg_index, sigma), b)
        features[f"smooth_sigma_{int(sigma)}_signed_offset"] = float(new_b - b_smooth)
        features[f"smooth_sigma_{int(sigma)}_abs_offset"] = float(abs(new_b - b_smooth))
        features[f"smooth_sigma_{int(sigma)}_move_agrees"] = float(
            np.sign(new_b - b) == np.sign(b_smooth - b) and b_smooth != b
        )
    cand_id = f"{ctx.dataset}_f{ctx.fold}_c{case_id}_b{seg_index}_d{delta}"
    return Candidate(
        candidate_id=cand_id,
        dataset=ctx.dataset,
        fold=int(ctx.fold),
        case_id=str(case_id),
        edit_type="boundary_shift",
        anchor_id=f"boundary_{seg_index}",
        start=int(start),
        end=int(end),
        old_label=str(old_label),
        new_label=str(new_label),
        delta=int(delta),
        features=features,
        delta_metrics=deltas,
        utility=utility,
        helpful=bool(utility > UTILITY_TAU and deltas["acc"] >= -ACC_EPSILON),
    )


def make_island_candidate(
    *,
    ctx: FoldContext,
    case_id: str,
    gt: Sequence[str],
    argmax: Sequence[str],
    mat: np.ndarray,
    sf: Dict[str, np.ndarray],
    priors: Dict[str, DurationPrior],
    process_features: Dict[str, float],
    seg_index: int,
    start: int,
    end: int,
    label: str,
    left_label: str,
    right_label: str,
) -> Optional[Candidate]:
    if left_label != right_label:
        return None
    length = end - start
    if length <= 0 or length > ISLAND_MAX_LEN:
        return None
    mean_margin, min_margin, max_margin = interval_stats(sf["margin"], start, end)
    if mean_margin > ISLAND_MARGIN_THRESHOLD:
        return None
    new_label = left_label
    pred = apply_candidate_raw(argmax, start, end, new_label)
    deltas = metric_deltas(gt, argmax, pred)
    utility = utility_from_deltas(deltas)
    n = len(argmax)
    old_lp = label_prob_sum(mat, label, start, end)
    new_lp = label_prob_sum(mat, new_label, start, end)
    mean_pmax, min_pmax, max_pmax = interval_stats(sf["pmax"], start, end)
    mean_entropy, min_entropy, max_entropy = interval_stats(sf["entropy"], start, end)
    before_raw = duration_score(priors[label], length, n, "raw", 9.0) if label in priors else 0.0
    after_raw = (
        duration_score(priors[new_label], length, n, "raw", 9.0)
        if new_label in priors
        else 0.0
    )
    before_norm = (
        duration_score(priors[label], length, n, "normalized", 9.0)
        if label in priors
        else 0.0
    )
    after_norm = (
        duration_score(priors[new_label], length, n, "normalized", 9.0)
        if new_label in priors
        else 0.0
    )
    features = candidate_feature_base(
        ctx=ctx,
        case_id=case_id,
        n_frames=n,
        process_features=process_features,
    )
    features.update(
        {
            "edit_type_boundary": 0.0,
            "edit_type_island": 1.0,
            "delta": 0.0,
            "abs_delta": 0.0,
            "direction": 0.0,
            "changed_len": float(length),
            "changed_len_norm": float(length / n),
            "anchor_index": float(seg_index),
            "anchor_pos_norm": float(start / n),
            "left_class": float(int_label(left_label)),
            "right_class": float(int_label(right_label)),
            "old_label": float(int_label(label)),
            "new_label": float(int_label(new_label)),
            "same_neighbor_aba": 1.0,
            "left_len_before": 0.0,
            "right_len_before": 0.0,
            "left_len_after": 0.0,
            "right_len_after": 0.0,
            "min_len_after": float(length),
            "logprob_delta_sum": float(new_lp - old_lp),
            "logprob_delta_per_frame": float((new_lp - old_lp) / length),
            "mean_pmax": mean_pmax,
            "min_pmax": min_pmax,
            "max_pmax": max_pmax,
            "mean_margin": mean_margin,
            "min_margin": min_margin,
            "max_margin": max_margin,
            "mean_entropy": mean_entropy,
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "high_conf_frac_095": float(np.mean(sf["pmax"][start:end] >= 0.95)),
            "high_conf_frac_098": float(np.mean(sf["pmax"][start:end] >= 0.98)),
            "duration_raw_delta": float(after_raw - before_raw),
            "duration_norm_delta": float(after_norm - before_norm),
        }
    )
    for sigma in SMOOTH_SIGMAS:
        features[f"smooth_sigma_{int(sigma)}_signed_offset"] = 0.0
        features[f"smooth_sigma_{int(sigma)}_abs_offset"] = 0.0
        features[f"smooth_sigma_{int(sigma)}_move_agrees"] = 0.0
    cand_id = f"{ctx.dataset}_f{ctx.fold}_c{case_id}_i{seg_index}"
    return Candidate(
        candidate_id=cand_id,
        dataset=ctx.dataset,
        fold=int(ctx.fold),
        case_id=str(case_id),
        edit_type="aba_island_delete",
        anchor_id=f"island_{seg_index}",
        start=int(start),
        end=int(end),
        old_label=str(label),
        new_label=str(new_label),
        delta=0,
        features=features,
        delta_metrics=deltas,
        utility=utility,
        helpful=bool(utility > UTILITY_TAU and deltas["acc"] >= -ACC_EPSILON),
    )


def generate_case_candidates(
    *,
    ctx: FoldContext,
    case_id: str,
    priors: Dict[str, DurationPrior],
    process_features: Dict[Tuple[int, str], Dict[str, float]],
) -> Tuple[Dict[str, Any], List[Candidate]]:
    case_df = load_case_output(ctx.dataset, ctx.fold, case_id)
    gt = case_df["ground_truth"].astype(str).tolist()
    argmax = case_df["argmax_activity"].astype(str).tolist()
    sktr = case_df["sktr_activity"].astype(str).tolist()
    mat = ctx.case_to_mat[str(case_id)]
    if not (len(gt) == len(argmax) == len(sktr) == mat.shape[1]):
        raise ValueError(f"Length mismatch {ctx.dataset} fold {ctx.fold} case {case_id}")
    raw_argmax = mat.argmax(axis=0).astype(str).tolist()
    if raw_argmax != argmax:
        bad = [i for i, (a, b) in enumerate(zip(raw_argmax, argmax)) if a != b][:10]
        raise ValueError(
            f"Raw softmax argmax mismatch {ctx.dataset} fold {ctx.fold} case {case_id}: {bad}"
        )

    sf = softmax_frame_features(mat)
    segs = segments(argmax)
    smoothed = {sigma: smooth_softmax(mat, sigma) for sigma in SMOOTH_SIGMAS}
    smooth_crossings: Dict[Tuple[int, float], int] = {}
    for i, (left_seg, right_seg) in enumerate(zip(segs[:-1], segs[1:])):
        _, b, left_class = left_seg
        _, _, right_class = right_seg
        for sigma, smooth_mat in smoothed.items():
            b_smooth, _, _, _, _ = smoothed_crossing_boundary(
                smooth_mat=smooth_mat,
                left_class=str(left_class),
                right_class=str(right_class),
                boundary=int(b),
                window=25,
            )
            smooth_crossings[(i, sigma)] = int(b_smooth)

    proc = process_features.get((int(ctx.fold), str(case_id)), {})
    candidates: List[Candidate] = []
    for i, (left_seg, right_seg) in enumerate(zip(segs[:-1], segs[1:])):
        prev_b = 0 if i == 0 else int(segs[i - 1][1])
        _, b, left_class = left_seg
        _, next_b, right_class = right_seg
        for delta in BOUNDARY_DELTAS:
            cand = make_boundary_candidate(
                ctx=ctx,
                case_id=str(case_id),
                gt=gt,
                argmax=argmax,
                mat=mat,
                sf=sf,
                priors=priors,
                smooth_crossings=smooth_crossings,
                process_features=proc,
                seg_index=i,
                prev_b=int(prev_b),
                b=int(b),
                next_b=int(next_b),
                left_class=str(left_class),
                right_class=str(right_class),
                delta=int(delta),
            )
            if cand is not None:
                candidates.append(cand)

    for i in range(1, len(segs) - 1):
        start, end, label = segs[i]
        left_label = str(segs[i - 1][2])
        right_label = str(segs[i + 1][2])
        cand = make_island_candidate(
            ctx=ctx,
            case_id=str(case_id),
            gt=gt,
            argmax=argmax,
            mat=mat,
            sf=sf,
            priors=priors,
            process_features=proc,
            seg_index=i,
            start=int(start),
            end=int(end),
            label=str(label),
            left_label=left_label,
            right_label=right_label,
        )
        if cand is not None:
            candidates.append(cand)

    case = {
        "dataset": ctx.dataset,
        "fold": int(ctx.fold),
        "case_id": str(case_id),
        "gt": gt,
        "argmax": argmax,
        "sktr": sktr,
        "n_frames": len(gt),
    }
    return case, candidates


def candidates_to_frame(candidates: Sequence[Candidate]) -> pd.DataFrame:
    rows = []
    for cand in candidates:
        row = {
            "candidate_id": cand.candidate_id,
            "dataset": cand.dataset,
            "fold": cand.fold,
            "case_id": cand.case_id,
            "edit_type": cand.edit_type,
            "anchor_id": cand.anchor_id,
            "start": cand.start,
            "end": cand.end,
            "old_label": cand.old_label,
            "new_label": cand.new_label,
            "delta": cand.delta,
            "utility": cand.utility,
            "helpful": cand.helpful,
        }
        for key in METRIC_KEYS:
            row[f"delta_{key.replace('@', '_')}"] = cand.delta_metrics[key]
        row.update(cand.features)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(candidates: pd.DataFrame) -> List[str]:
    excluded = {
        "candidate_id",
        "dataset",
        "fold",
        "case_id",
        "edit_type",
        "anchor_id",
        "start",
        "end",
        "old_label",
        "new_label",
        "utility",
        "helpful",
        "delta_acc",
        "delta_edit",
        "delta_f1_10",
        "delta_f1_25",
        "delta_f1_50",
    }
    cols = [
        c
        for c in candidates.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(candidates[c])
    ]
    return sorted(cols)


def non_overlaps(used: List[Tuple[int, int]], start: int, end: int) -> bool:
    return all(end <= s or start >= e for s, e in used)


def apply_selected_candidates(
    case: Dict[str, Any],
    selected: Sequence[Candidate],
) -> List[str]:
    pred = [str(x) for x in case["argmax"]]
    for cand in selected:
        pred[cand.start : cand.end] = [cand.new_label] * (cand.end - cand.start)
    return pred


def oracle_select_case(cands: Sequence[Candidate], n_frames: int) -> List[Candidate]:
    best_by_anchor: Dict[str, Candidate] = {}
    for cand in cands:
        cur = best_by_anchor.get(cand.anchor_id)
        if cur is None or (cand.utility, cand.delta_metrics["acc"]) > (
            cur.utility,
            cur.delta_metrics["acc"],
        ):
            best_by_anchor[cand.anchor_id] = cand
    selected: List[Candidate] = []
    used: List[Tuple[int, int]] = []
    for cand in sorted(best_by_anchor.values(), key=lambda c: c.utility, reverse=True):
        if cand.utility <= 0.0:
            continue
        if non_overlaps(used, cand.start, cand.end):
            selected.append(cand)
            used.append((cand.start, cand.end))
    return selected


def case_key(case: Dict[str, Any]) -> Tuple[str, int, str]:
    return (str(case["dataset"]), int(case["fold"]), str(case["case_id"]))


def aggregate_eval(
    cases: Sequence[Dict[str, Any]],
    pred_by_case: Dict[Tuple[str, int, str], List[str]],
) -> Dict[str, float]:
    gt = [case["gt"] for case in cases]
    pred = [pred_by_case[case_key(case)] for case in cases]
    return compute_tas_metrics_from_sequences(gt, pred)


def base_eval(cases: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    return compute_tas_metrics_from_sequences(
        [case["gt"] for case in cases], [case[key] for case in cases]
    )


def make_candidate_lookup(candidates: Sequence[Candidate]) -> Dict[str, Candidate]:
    return {cand.candidate_id: cand for cand in candidates}


def build_oracle_ceiling(
    cases: Sequence[Dict[str, Any]],
    candidates: Sequence[Candidate],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_case: Dict[Tuple[str, int, str], List[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_case[(cand.dataset, cand.fold, cand.case_id)].append(cand)

    pred_by_case: Dict[Tuple[str, int, str], List[str]] = {}
    selected_rows: List[Dict[str, Any]] = []
    for case in cases:
        key = case_key(case)
        selected = oracle_select_case(by_case.get(key, []), int(case["n_frames"]))
        pred_by_case[key] = apply_selected_candidates(case, selected)
        for cand in selected:
            selected_rows.append(
                {
                    "dataset": cand.dataset,
                    "fold": cand.fold,
                    "case_id": cand.case_id,
                    "candidate_id": cand.candidate_id,
                    "edit_type": cand.edit_type,
                    "utility": cand.utility,
                    "changed_frames": cand.end - cand.start,
                }
            )

    rows: List[Dict[str, Any]] = []
    for dataset in sorted({case["dataset"] for case in cases}):
        ds_cases = [case for case in cases if case["dataset"] == dataset]
        ds_cands = [cand for cand in candidates if cand.dataset == dataset]
        for fold_value, group in [("all", ds_cases)] + [
            (fold, [case for case in ds_cases if case["fold"] == fold])
            for fold in sorted({case["fold"] for case in ds_cases})
        ]:
            group_case_ids = {case["case_id"] for case in group}
            group_cands = [cand for cand in ds_cands if cand.case_id in group_case_ids]
            base = base_eval(group, "argmax")
            sktr = base_eval(group, "sktr")
            pred = aggregate_eval(group, pred_by_case)
            selected = [
                row
                for row in selected_rows
                if row["dataset"] == dataset and row["case_id"] in group_case_ids
            ]
            type_counts = Counter(cand.edit_type for cand in group_cands)
            selected_type_counts = Counter(row["edit_type"] for row in selected)
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold_value,
                    "n_cases": len(group),
                    "n_candidates": len(group_cands),
                    "boundary_shift_candidates": type_counts.get("boundary_shift", 0),
                    "aba_island_candidates": type_counts.get("aba_island_delete", 0),
                    "oracle_applied_edits": len(selected),
                    "oracle_applied_boundary_shift": selected_type_counts.get(
                        "boundary_shift", 0
                    ),
                    "oracle_applied_aba_island": selected_type_counts.get(
                        "aba_island_delete", 0
                    ),
                    "oracle_edited_frames": int(sum(row["changed_frames"] for row in selected)),
                    "argmax_acc": base["acc"],
                    "argmax_edit": base["edit"],
                    "argmax_f1_10": base["f1@10"],
                    "argmax_f1_25": base["f1@25"],
                    "argmax_f1_50": base["f1@50"],
                    "sktr_acc": sktr["acc"],
                    "sktr_edit": sktr["edit"],
                    "sktr_f1_10": sktr["f1@10"],
                    "sktr_f1_25": sktr["f1@25"],
                    "sktr_f1_50": sktr["f1@50"],
                    "oracle_acc": pred["acc"],
                    "oracle_edit": pred["edit"],
                    "oracle_f1_10": pred["f1@10"],
                    "oracle_f1_25": pred["f1@25"],
                    "oracle_f1_50": pred["f1@50"],
                    "delta_acc": pred["acc"] - base["acc"],
                    "delta_edit": pred["edit"] - base["edit"],
                    "delta_f1_10": pred["f1@10"] - base["f1@10"],
                    "delta_f1_25": pred["f1@25"] - base["f1@25"],
                    "delta_f1_50": pred["f1@50"] - base["f1@50"],
                    "oracle_uses_test_gt": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def oracle_gate(oracle_df: pd.DataFrame) -> Dict[str, bool]:
    gates = {}
    focus = oracle_df[oracle_df["fold"].astype(str) == "all"]
    for row in focus.itertuples(index=False):
        gates[str(row.dataset)] = bool(row.delta_edit >= 1.0 or row.delta_f1_50 >= 1.0)
    return gates


def model_from_params(params: Dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        max_iter=int(params["max_iter"]),
        l2_regularization=0.0,
        random_state=42,
    )


def predict_scores(model: Any, x: pd.DataFrame) -> np.ndarray:
    if model is None or len(x) == 0:
        return np.zeros(len(x), dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return np.full(len(x), float(model.classes_[0] == 1))
        idx = list(model.classes_).index(True) if True in model.classes_ else 1
        return proba[:, idx]
    return np.zeros(len(x), dtype=float)


def train_model(train_df: pd.DataFrame, features: Sequence[str], params: Dict[str, Any]) -> Any:
    if train_df.empty or train_df["helpful"].nunique() < 2:
        return None
    model = model_from_params(params)
    x = train_df[list(features)].fillna(0.0)
    y = train_df["helpful"].astype(bool)
    model.fit(x, y)
    return model


def select_candidates_by_scores(
    *,
    case: Dict[str, Any],
    case_candidate_df: pd.DataFrame,
    candidate_lookup: Dict[str, Candidate],
    threshold: float,
    budget: float,
    lock_threshold: float,
) -> List[Candidate]:
    if case_candidate_df.empty:
        return []
    frame_budget = int(math.floor(float(case["n_frames"]) * budget))
    if frame_budget <= 0:
        return []
    selected: List[Candidate] = []
    used: List[Tuple[int, int]] = []
    used_frames = 0
    seen_anchors: set[str] = set()
    sort_cols = ["score", "utility_feature_tiebreak"]
    df = case_candidate_df.copy()
    df["utility_feature_tiebreak"] = df["logprob_delta_sum"].astype(float)
    for row in df.sort_values(sort_cols, ascending=False).itertuples(index=False):
        if float(row.score) < threshold:
            continue
        cand = candidate_lookup[str(row.candidate_id)]
        if cand.anchor_id in seen_anchors:
            continue
        changed = cand.end - cand.start
        if used_frames + changed > frame_budget:
            continue
        if float(row.max_pmax) >= lock_threshold:
            continue
        if not non_overlaps(used, cand.start, cand.end):
            continue
        selected.append(cand)
        used.append((cand.start, cand.end))
        used_frames += changed
        seen_anchors.add(cand.anchor_id)
    return selected


def evaluate_config_on_cases(
    *,
    cases: Sequence[Dict[str, Any]],
    scored_df: pd.DataFrame,
    candidate_lookup: Dict[str, Candidate],
    threshold: float,
    budget: float,
    lock_threshold: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    pred_by_case: Dict[Tuple[str, int, str], List[str]] = {}
    case_rows: List[Dict[str, Any]] = []
    for case in cases:
        sub = scored_df[
            (scored_df["dataset"] == case["dataset"])
            & (scored_df["fold"] == case["fold"])
            & (scored_df["case_id"].astype(str) == str(case["case_id"]))
        ]
        selected = select_candidates_by_scores(
            case=case,
            case_candidate_df=sub,
            candidate_lookup=candidate_lookup,
            threshold=threshold,
            budget=budget,
            lock_threshold=lock_threshold,
        )
        pred = apply_selected_candidates(case, selected)
        pred_by_case[case_key(case)] = pred
        base_m = tas_metrics(case["gt"], case["argmax"])
        pred_m = tas_metrics(case["gt"], pred)
        helpful_applied = sum(c.helpful for c in selected)
        case_rows.append(
            {
                "dataset": case["dataset"],
                "fold": case["fold"],
                "case_id": case["case_id"],
                "n_edits_applied": len(selected),
                "frames_edited": int(sum(c.end - c.start for c in selected)),
                "precision_at_applied_edits": safe_div(helpful_applied, len(selected)),
                "delta_acc": pred_m["acc"] - base_m["acc"],
                "delta_edit": pred_m["edit"] - base_m["edit"],
                "delta_f1_10": pred_m["f1@10"] - base_m["f1@10"],
                "delta_f1_25": pred_m["f1@25"] - base_m["f1@25"],
                "delta_f1_50": pred_m["f1@50"] - base_m["f1@50"],
            }
        )
    base = base_eval(cases, "argmax")
    pred_metrics = aggregate_eval(cases, pred_by_case)
    summary = {
        "delta_acc": pred_metrics["acc"] - base["acc"],
        "delta_edit": pred_metrics["edit"] - base["edit"],
        "delta_f1_10": pred_metrics["f1@10"] - base["f1@10"],
        "delta_f1_25": pred_metrics["f1@25"] - base["f1@25"],
        "delta_f1_50": pred_metrics["f1@50"] - base["f1@50"],
        "acc": pred_metrics["acc"],
        "edit": pred_metrics["edit"],
        "f1_10": pred_metrics["f1@10"],
        "f1_25": pred_metrics["f1@25"],
        "f1_50": pred_metrics["f1@50"],
    }
    return summary, case_rows


def selection_score(summary: Dict[str, float]) -> float:
    acc_penalty = max(0.0, -0.10 - summary["delta_acc"])
    return float(
        summary["delta_edit"]
        + summary["delta_f1_50"]
        + 0.5 * summary["delta_f1_25"]
        - 10.0 * acc_penalty
    )


def inner_select_config(
    *,
    train_cases: Sequence[Dict[str, Any]],
    train_df: pd.DataFrame,
    features: Sequence[str],
    candidate_lookup: Dict[str, Candidate],
) -> Dict[str, Any]:
    folds = sorted({int(case["fold"]) for case in train_cases})
    if len(folds) < 2 or train_df["helpful"].nunique() < 2:
        return {
            "learning_rate": 0.05,
            "max_leaf_nodes": 7,
            "max_iter": 60,
            "threshold": 1.01,
            "budget": 0.01,
            "lock_threshold": 0.95,
            "inner_score": 0.0,
        }
    # Keep the feasibility pass intentionally small: the expensive part is
    # repeatedly evaluating edited sequences with TAS metrics inside inner CV.
    param_grid = [{"learning_rate": 0.05, "max_leaf_nodes": 7, "max_iter": 40}]
    best: Optional[Dict[str, Any]] = None
    for params in param_grid:
        scored_parts = []
        for val_fold in folds:
            fit_df = train_df[train_df["fold"].astype(int) != val_fold]
            val_df = train_df[train_df["fold"].astype(int) == val_fold].copy()
            model = train_model(fit_df, features, params)
            val_df["score"] = predict_scores(model, val_df[list(features)].fillna(0.0))
            scored_parts.append(val_df)
        scored = pd.concat(scored_parts, ignore_index=True)
        for threshold in PROBA_THRESHOLDS:
            for budget in EDIT_BUDGETS:
                for lock in LOCK_THRESHOLDS:
                    fold_scores = []
                    for val_fold in folds:
                        val_cases = [case for case in train_cases if case["fold"] == val_fold]
                        val_scored = scored[scored["fold"].astype(int) == val_fold]
                        summary, _ = evaluate_config_on_cases(
                            cases=val_cases,
                            scored_df=val_scored,
                            candidate_lookup=candidate_lookup,
                            threshold=threshold,
                            budget=budget,
                            lock_threshold=lock,
                        )
                        fold_scores.append(selection_score(summary))
                    mean_score = float(np.mean(fold_scores))
                    config = {
                        **params,
                        "threshold": threshold,
                        "budget": budget,
                        "lock_threshold": lock,
                        "inner_score": mean_score,
                    }
                    if best is None or mean_score > best["inner_score"]:
                        best = config
    assert best is not None
    return best


def run_outer_fold_learning(
    *,
    dataset: str,
    cases: Sequence[Dict[str, Any]],
    candidates_df: pd.DataFrame,
    candidate_lookup: Dict[str, Candidate],
    features: Sequence[str],
    gates: Dict[str, bool],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    heldout_rows: List[Dict[str, Any]] = []
    case_rows_all: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    ds_cases = [case for case in cases if case["dataset"] == dataset]
    ds_df = candidates_df[candidates_df["dataset"] == dataset].copy()
    if not gates.get(dataset, False):
        for fold in sorted({case["fold"] for case in ds_cases}):
            test_cases = [case for case in ds_cases if case["fold"] == fold]
            heldout_rows.append(
                {
                    "dataset": dataset,
                    "outer_unit": "fold",
                    "fold": fold,
                    "case_id": "all",
                    "gate_passed": False,
                    "selected_config": "oracle_gate_failed",
                    "delta_acc": 0.0,
                    "delta_edit": 0.0,
                    "delta_f1_10": 0.0,
                    "delta_f1_25": 0.0,
                    "delta_f1_50": 0.0,
                    "n_cases": len(test_cases),
                    "n_videos_improved_f1_50": 0,
                    "n_videos_harmed_f1_50": 0,
                    "n_videos_improved_acc": 0,
                    "n_videos_harmed_acc": 0,
                    "worst_case_harm_acc": 0.0,
                    "n_edits_applied": 0,
                    "precision_at_applied_edits": float("nan"),
                }
            )
        return heldout_rows, case_rows_all, importance_rows

    for test_fold in sorted({case["fold"] for case in ds_cases}):
        print(f"[stage2]   outer fold {dataset} {test_fold}", flush=True)
        train_cases = [case for case in ds_cases if case["fold"] != test_fold]
        test_cases = [case for case in ds_cases if case["fold"] == test_fold]
        train_ids = {(case["fold"], case["case_id"]) for case in train_cases}
        test_ids = {(case["fold"], case["case_id"]) for case in test_cases}
        train_df = ds_df[
            ds_df.apply(lambda r: (int(r["fold"]), str(r["case_id"])) in train_ids, axis=1)
        ].copy()
        test_df = ds_df[
            ds_df.apply(lambda r: (int(r["fold"]), str(r["case_id"])) in test_ids, axis=1)
        ].copy()
        config = inner_select_config(
            train_cases=train_cases,
            train_df=train_df,
            features=features,
            candidate_lookup=candidate_lookup,
        )
        model = train_model(train_df, features, config)
        test_df["score"] = predict_scores(model, test_df[list(features)].fillna(0.0))
        summary, case_rows = evaluate_config_on_cases(
            cases=test_cases,
            scored_df=test_df,
            candidate_lookup=candidate_lookup,
            threshold=float(config["threshold"]),
            budget=float(config["budget"]),
            lock_threshold=float(config["lock_threshold"]),
        )
        applied = int(sum(row["n_edits_applied"] for row in case_rows))
        precision_vals = [
            row["precision_at_applied_edits"]
            for row in case_rows
            if row["n_edits_applied"] > 0
        ]
        heldout_rows.append(
            {
                "dataset": dataset,
                "outer_unit": "fold",
                "fold": test_fold,
                "case_id": "all",
                "gate_passed": True,
                "selected_config": json.dumps(config, sort_keys=True),
                **summary,
                "n_cases": len(test_cases),
                "n_videos_improved_f1_50": int(
                    sum(row["delta_f1_50"] > 1e-12 for row in case_rows)
                ),
                "n_videos_harmed_f1_50": int(
                    sum(row["delta_f1_50"] < -1e-12 for row in case_rows)
                ),
                "n_videos_improved_acc": int(
                    sum(row["delta_acc"] > 1e-12 for row in case_rows)
                ),
                "n_videos_harmed_acc": int(
                    sum(row["delta_acc"] < -1e-12 for row in case_rows)
                ),
                "worst_case_harm_acc": float(min(row["delta_acc"] for row in case_rows))
                if case_rows
                else 0.0,
                "n_edits_applied": applied,
                "precision_at_applied_edits": float(np.mean(precision_vals))
                if precision_vals
                else float("nan"),
            }
        )
        for row in case_rows:
            row.update(
                {
                    "outer_unit": "fold",
                    "selected_config": json.dumps(config, sort_keys=True),
                }
            )
        case_rows_all.extend(case_rows)

        if model is not None and test_df["helpful"].nunique() > 1:
            try:
                result = permutation_importance(
                    model,
                    test_df[list(features)].fillna(0.0),
                    test_df["helpful"].astype(bool),
                    scoring="average_precision",
                    n_repeats=3,
                    random_state=42,
                )
                for feat, mean, std in zip(
                    features, result.importances_mean, result.importances_std
                ):
                    importance_rows.append(
                        {
                            "dataset": dataset,
                            "outer_unit": "fold",
                            "fold": test_fold,
                            "feature": feat,
                            "importance_mean": float(mean),
                            "importance_std": float(std),
                        }
                    )
            except Exception as exc:
                importance_rows.append(
                    {
                        "dataset": dataset,
                        "outer_unit": "fold",
                        "fold": test_fold,
                        "feature": "__permutation_importance_failed__",
                        "importance_mean": 0.0,
                        "importance_std": 0.0,
                        "error": str(exc),
                    }
                )
    return heldout_rows, case_rows_all, importance_rows


def run_gtea_leave_one_video(
    *,
    cases: Sequence[Dict[str, Any]],
    candidates_df: pd.DataFrame,
    candidate_lookup: Dict[str, Candidate],
    features: Sequence[str],
    gates: Dict[str, bool],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not gates.get("gtea", False):
        return [], []
    ds_cases = [case for case in cases if case["dataset"] == "gtea"]
    ds_df = candidates_df[candidates_df["dataset"] == "gtea"].copy()
    rows: List[Dict[str, Any]] = []
    case_rows_all: List[Dict[str, Any]] = []
    for test_case in ds_cases:
        print(f"[stage2]   leave-one-video GTEA case {test_case['case_id']}", flush=True)
        train_cases = [case for case in ds_cases if case["case_id"] != test_case["case_id"]]
        train_ids = {(case["fold"], case["case_id"]) for case in train_cases}
        train_df = ds_df[
            ds_df.apply(lambda r: (int(r["fold"]), str(r["case_id"])) in train_ids, axis=1)
        ].copy()
        test_df = ds_df[
            (ds_df["fold"].astype(int) == int(test_case["fold"]))
            & (ds_df["case_id"].astype(str) == str(test_case["case_id"]))
        ].copy()
        config = inner_select_config(
            train_cases=train_cases,
            train_df=train_df,
            features=features,
            candidate_lookup=candidate_lookup,
        )
        model = train_model(train_df, features, config)
        test_df["score"] = predict_scores(model, test_df[list(features)].fillna(0.0))
        summary, case_rows = evaluate_config_on_cases(
            cases=[test_case],
            scored_df=test_df,
            candidate_lookup=candidate_lookup,
            threshold=float(config["threshold"]),
            budget=float(config["budget"]),
            lock_threshold=float(config["lock_threshold"]),
        )
        row = case_rows[0]
        rows.append(
            {
                "dataset": "gtea",
                "outer_unit": "video",
                "fold": test_case["fold"],
                "case_id": test_case["case_id"],
                "gate_passed": True,
                "selected_config": json.dumps(config, sort_keys=True),
                **summary,
                "n_cases": 1,
                "n_videos_improved_f1_50": int(row["delta_f1_50"] > 1e-12),
                "n_videos_harmed_f1_50": int(row["delta_f1_50"] < -1e-12),
                "n_videos_improved_acc": int(row["delta_acc"] > 1e-12),
                "n_videos_harmed_acc": int(row["delta_acc"] < -1e-12),
                "worst_case_harm_acc": row["delta_acc"],
                "n_edits_applied": row["n_edits_applied"],
                "precision_at_applied_edits": row["precision_at_applied_edits"],
            }
        )
        row.update(
            {
                "outer_unit": "video",
                "selected_config": json.dumps(config, sort_keys=True),
            }
        )
        case_rows_all.append(row)
    return rows, case_rows_all


def run_swept_best_optimistic(
    *,
    cases: Sequence[Dict[str, Any]],
    candidates_df: pd.DataFrame,
    candidate_lookup: Dict[str, Candidate],
    features: Sequence[str],
    gates: Dict[str, bool],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in sorted({case["dataset"] for case in cases}):
        ds_cases = [case for case in cases if case["dataset"] == dataset]
        ds_df = candidates_df[candidates_df["dataset"] == dataset].copy()
        if not gates.get(dataset, False):
            rows.append(
                {
                    "dataset": dataset,
                    "optimistic": True,
                    "selected_config": "oracle_gate_failed",
                    "delta_acc": 0.0,
                    "delta_edit": 0.0,
                    "delta_f1_10": 0.0,
                    "delta_f1_25": 0.0,
                    "delta_f1_50": 0.0,
                    "n_cases": len(ds_cases),
                    "n_edits_applied": 0,
                    "worst_case_harm_acc": 0.0,
                    "precision_at_applied_edits": float("nan"),
                }
            )
            continue
        config_model = {"learning_rate": 0.05, "max_leaf_nodes": 7, "max_iter": 40}
        model = train_model(ds_df, features, config_model)
        ds_df["score"] = predict_scores(model, ds_df[list(features)].fillna(0.0))
        best_row: Optional[Dict[str, Any]] = None
        for threshold in [0.30, 0.50, 0.70]:
            for budget in EDIT_BUDGETS:
                for lock in LOCK_THRESHOLDS:
                    summary, case_rows = evaluate_config_on_cases(
                        cases=ds_cases,
                        scored_df=ds_df,
                        candidate_lookup=candidate_lookup,
                        threshold=threshold,
                        budget=budget,
                        lock_threshold=lock,
                    )
                    applied = int(sum(row["n_edits_applied"] for row in case_rows))
                    precision_vals = [
                        row["precision_at_applied_edits"]
                        for row in case_rows
                        if row["n_edits_applied"] > 0
                    ]
                    candidate_row = {
                        "dataset": dataset,
                        "optimistic": True,
                        "selected_config": json.dumps(
                            {
                                **config_model,
                                "threshold": threshold,
                                "budget": budget,
                                "lock_threshold": lock,
                            },
                            sort_keys=True,
                        ),
                        **summary,
                        "n_cases": len(ds_cases),
                        "n_edits_applied": applied,
                        "worst_case_harm_acc": float(
                            min(row["delta_acc"] for row in case_rows)
                        )
                        if case_rows
                        else 0.0,
                        "precision_at_applied_edits": float(np.mean(precision_vals))
                        if precision_vals
                        else float("nan"),
                        "selection_score": selection_score(summary),
                    }
                    if best_row is None or candidate_row["selection_score"] > best_row["selection_score"]:
                        best_row = candidate_row
        assert best_row is not None
        rows.append(best_row)
    return pd.DataFrame(rows)


def binomial_two_sided_p(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    prob = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(2000):
        sample = vals[rng.integers(0, len(vals), len(vals))]
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_significance(case_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    fold_rows = case_rows[case_rows["outer_unit"] == "fold"].copy()
    for dataset in sorted(fold_rows["dataset"].unique()):
        ds = fold_rows[fold_rows["dataset"] == dataset]
        subsets = [("all", ds)]
        if dataset == "50salads":
            mask1 = ~((ds["case_id"].astype(str) == "1") & (ds["fold"].astype(int) == 1))
            mask49 = ~((ds["case_id"].astype(str) == "49") & (ds["fold"].astype(int) == 5))
            subsets.extend(
                [
                    ("without_case1", ds[mask1]),
                    ("without_case49", ds[mask49]),
                    ("without_case1_case49", ds[mask1 & mask49]),
                ]
            )
        for subset_name, sub in subsets:
            for metric in ["delta_f1_50", "delta_edit", "delta_acc"]:
                vals = sub[metric].to_numpy(dtype=float)
                lo, hi = bootstrap_ci(vals, rng)
                pos = int(np.sum(vals > 1e-12))
                neg = int(np.sum(vals < -1e-12))
                rows.append(
                    {
                        "dataset": dataset,
                        "subset": subset_name,
                        "metric": metric,
                        "n_cases": int(len(vals)),
                        "mean_delta": float(np.mean(vals)) if len(vals) else float("nan"),
                        "bootstrap_ci_low": lo,
                        "bootstrap_ci_high": hi,
                        "sign_test_p": binomial_two_sided_p(pos, neg),
                        "fraction_videos_improved": safe_div(pos, len(vals)),
                        "positive_cases": pos,
                        "negative_cases": neg,
                    }
                )
    return pd.DataFrame(rows)


def write_smoke_report(
    out_dir: Path,
    candidates_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    parity: Dict[str, Any],
) -> None:
    gtea_f1 = candidates_df[
        (candidates_df["dataset"] == "gtea") & (candidates_df["fold"].astype(int) == 1)
    ]
    examples = gtea_f1.sort_values("utility", ascending=False).head(12)
    lines = [
        "# Stage-2.0 Learned Edit Selector Smoke",
        "",
        "Scope: GTEA fold 1 candidate/oracle smoke. Candidate utility uses GT; candidate features do not.",
        "",
        "## Parity",
        "",
    ]
    for key, value in parity.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Candidate Counts",
            "",
            f"- GTEA fold 1 candidates: {len(gtea_f1)}",
            f"- boundary shifts: {int((gtea_f1['edit_type'] == 'boundary_shift').sum())}",
            f"- ABA islands: {int((gtea_f1['edit_type'] == 'aba_island_delete').sum())}",
            "",
            "## Example Scored Candidates",
            "",
        ]
    )
    if examples.empty:
        lines.append("- none")
    else:
        for row in examples.itertuples(index=False):
            lines.append(
                f"- case {row.case_id}, {row.edit_type}, {row.anchor_id}, "
                f"frames=[{row.start}:{row.end}), {row.old_label}->{row.new_label}, "
                f"delta={row.delta}, U={row.utility:.3f}, dAcc={row.delta_acc:.3f}, "
                f"dEdit={row.delta_edit:.3f}, dF1@50={row.delta_f1_50:.3f}, "
                f"mean_margin={row.mean_margin:.3f}, logprob_delta={row.logprob_delta_sum:.3f}."
            )
    lines.extend(["", "## Oracle Rows", ""])
    smoke_oracle = oracle_df[
        (oracle_df["dataset"] == "gtea") & (oracle_df["fold"].astype(str) == "1")
    ]
    for row in smoke_oracle.itertuples(index=False):
        lines.append(
            f"- fold {row.fold}: candidates={row.n_candidates}, applied={row.oracle_applied_edits}, "
            f"dAcc={row.delta_acc:.3f}, dEdit={row.delta_edit:.3f}, dF1@50={row.delta_f1_50:.3f}."
        )
    lines.extend(
        [
            "",
            "## Hand-check Notes",
            "",
            "Candidate intervals were inspected for boundary-shift sign convention and ABA island replacement. The smoke verifies softmax-to-argmax parity before any candidate features are written.",
        ]
    )
    (out_dir / "stage2_smoke_report.md").write_text("\n".join(lines) + "\n")


def write_summary(
    out_dir: Path,
    oracle_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    swept_df: pd.DataFrame,
    gates: Dict[str, bool],
) -> None:
    lines = [
        "# Stage-2.0 Learned Edit Selector Feasibility Summary",
        "",
        "Candidate edits are boundary shifts in {-25,-10,-5,+5,+10,+25} plus low-confidence ABA island deletion. GT is used for candidate labels/oracle and held-out evaluation only; model/threshold selection is nested within training folds.",
        "",
    ]
    oracle_focus = oracle_df[oracle_df["fold"].astype(str) == "all"]
    if heldout_df.empty or "outer_unit" not in heldout_df.columns:
        heldout_focus = pd.DataFrame()
    else:
        heldout_focus = heldout_df[heldout_df["outer_unit"] == "fold"]
    for row in oracle_focus.sort_values("dataset").itertuples(index=False):
        dataset = row.dataset
        lines.extend([f"## {dataset}", ""])
        lines.append(
            f"Oracle candidate ceiling: candidates={row.n_candidates}, applied={row.oracle_applied_edits}, "
            f"dAcc={row.delta_acc:.3f}, dEdit={row.delta_edit:.3f}, dF1@50={row.delta_f1_50:.3f}."
        )
        if not gates.get(dataset, False):
            lines.append("Gate 1: NO-GO; oracle candidate set does not clear +1 Edit or +1 F1@50.")
            lines.append("")
            continue
        if not swept_df.empty and "dataset" in swept_df.columns:
            swept = swept_df[swept_df["dataset"] == dataset]
            if not swept.empty:
                s = swept.iloc[0]
                lines.append(
                    f"Optimistic swept-best: dAcc={s.delta_acc:.3f}, "
                    f"dEdit={s.delta_edit:.3f}, dF1@50={s.delta_f1_50:.3f}, "
                    f"worst Acc harm={s.worst_case_harm_acc:.3f}, "
                    f"edits={int(s.n_edits_applied)}. This is test-swept and not deployable."
                )
        if heldout_focus.empty or "dataset" not in heldout_focus.columns:
            ds_hold = pd.DataFrame()
        else:
            ds_hold = heldout_focus[heldout_focus["dataset"] == dataset]
        if ds_hold.empty:
            lines.append("Gate 2: not evaluated.")
            lines.append("")
            continue
        weighted = {
            metric: float(np.average(ds_hold[metric], weights=ds_hold["n_cases"]))
            for metric in ["delta_acc", "delta_edit", "delta_f1_50"]
        }
        worst = float(ds_hold["worst_case_harm_acc"].min())
        edits = int(ds_hold["n_edits_applied"].sum())
        improved = int(ds_hold["n_videos_improved_f1_50"].sum())
        harmed = int(ds_hold["n_videos_harmed_f1_50"].sum())
        total_cases = int(ds_hold["n_cases"].sum())
        lines.append(
            f"Nested held-out: dAcc={weighted['delta_acc']:.3f}, "
            f"dEdit={weighted['delta_edit']:.3f}, dF1@50={weighted['delta_f1_50']:.3f}, "
            f"worst Acc harm={worst:.3f}, edits={edits}, F1@50 improved/harmed={improved}/{harmed} of {total_cases}."
        )
        sig = sig_df[
            (sig_df["dataset"] == dataset)
            & (sig_df["subset"] == "all")
            & (sig_df["metric"] == "delta_f1_50")
        ]
        ci_ok = False
        if not sig.empty:
            srow = sig.iloc[0]
            ci_ok = float(srow["bootstrap_ci_low"]) > 0
            lines.append(
                f"F1@50 case-mean CI: mean={srow.mean_delta:.3f}, "
                f"95% CI=[{srow.bootstrap_ci_low:.3f},{srow.bootstrap_ci_high:.3f}], "
                f"sign-test p={srow.sign_test_p:.4f}."
            )
        go = (
            weighted["delta_f1_50"] > 0
            and weighted["delta_edit"] > 0
            and weighted["delta_acc"] >= -0.10
            and worst > -1.0
            and improved > harmed
            and ci_ok
        )
        lines.append(
            "Gate 2 decision: "
            + (
                "GO; nested held-out gain clears the publishability bar."
                if go
                else "NO-GO; gain is not robust/Acc-neutral enough under nested held-out evaluation."
            )
        )
        lines.append("")
    (out_dir / "stage2_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets = ["gtea"]
        args.folds = [1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    complete_cases = load_complete_case_set()
    all_cases: List[Dict[str, Any]] = []
    all_candidates: List[Candidate] = []
    loaded_cases = 0
    skipped_incomplete = 0
    process_by_dataset = {dataset: load_process_features(dataset) for dataset in args.datasets}

    for dataset in args.datasets:
        for fold in get_folds(dataset, args.folds, args.data_root):
            print(f"[stage2] loading candidates: {dataset} fold {fold}", flush=True)
            ctx = load_fold_context(dataset, fold, args.data_root)
            priors, _, _ = build_duration_priors(ctx, int(args.rare_class_min_segments))
            test_cases = ctx.test_cases[: args.case_limit] if args.case_limit else ctx.test_cases
            for case_id in test_cases:
                key = (dataset, fold, str(case_id))
                if key not in complete_cases:
                    skipped_incomplete += 1
                    continue
                case, candidates = generate_case_candidates(
                    ctx=ctx,
                    case_id=str(case_id),
                    priors=priors,
                    process_features=process_by_dataset.get(dataset, {}),
                )
                all_cases.append(case)
                all_candidates.extend(candidates)
                loaded_cases += 1

    if not all_cases:
        raise ValueError("No complete cases loaded")
    print(
        f"[stage2] generated {len(all_candidates)} candidates for {len(all_cases)} cases",
        flush=True,
    )
    candidates_df = candidates_to_frame(all_candidates)
    features = feature_columns(candidates_df)
    candidate_lookup = make_candidate_lookup(all_candidates)
    print("[stage2] computing candidate-set oracle ceiling", flush=True)
    oracle_df, oracle_selected_df = build_oracle_ceiling(all_cases, all_candidates)
    gates = oracle_gate(oracle_df)

    heldout_rows: List[Dict[str, Any]] = []
    heldout_case_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    if not args.smoke:
        for dataset in sorted(set(args.datasets)):
            print(f"[stage2] nested fold-CV learning: {dataset}", flush=True)
            hr, cr, ir = run_outer_fold_learning(
                dataset=dataset,
                cases=all_cases,
                candidates_df=candidates_df,
                candidate_lookup=candidate_lookup,
                features=features,
                gates=gates,
            )
            heldout_rows.extend(hr)
            heldout_case_rows.extend(cr)
            importance_rows.extend(ir)
        if "gtea" in args.datasets and gates.get("gtea", False):
            print("[stage2] GTEA leave-one-video robustness", flush=True)
            video_rows, video_case_rows = run_gtea_leave_one_video(
                cases=all_cases,
                candidates_df=candidates_df,
                candidate_lookup=candidate_lookup,
                features=features,
                gates=gates,
            )
            heldout_rows.extend(video_rows)
            heldout_case_rows.extend(video_case_rows)

    heldout_df = pd.DataFrame(heldout_rows)
    heldout_case_df = pd.DataFrame(heldout_case_rows)
    swept_df = (
        run_swept_best_optimistic(
            cases=all_cases,
            candidates_df=candidates_df,
            candidate_lookup=candidate_lookup,
            features=features,
            gates=gates,
        )
        if not args.smoke
        else pd.DataFrame()
    )
    sig_df = (
        build_significance(heldout_case_df)
        if not heldout_case_df.empty and "outer_unit" in heldout_case_df.columns
        else pd.DataFrame()
    )
    importance_df = pd.DataFrame(importance_rows)

    candidates_df.to_csv(out_dir / "stage2_candidates.csv", index=False)
    oracle_df.to_csv(out_dir / "stage2_candidate_oracle_ceiling.csv", index=False)
    oracle_selected_df.to_csv(out_dir / "stage2_oracle_selected_edits.csv", index=False)
    swept_df.to_csv(out_dir / "stage2_swept_best_optimistic.csv", index=False)
    heldout_df.to_csv(out_dir / "stage2_learnability_heldout.csv", index=False)
    heldout_case_df.to_csv(out_dir / "stage2_learnability_case_details.csv", index=False)
    sig_df.to_csv(out_dir / "stage2_significance.csv", index=False)
    importance_df.to_csv(out_dir / "stage2_feature_importance.csv", index=False)

    parity = {
        "raw_softmax_argmax_matches_case_csv": True,
        "frame_counts_match_gt_argmax_sktr_softmax": True,
        "incomplete_cases_excluded": True,
        "loaded_cases": loaded_cases,
        "skipped_incomplete_cases": skipped_incomplete,
        "n_candidates": int(len(candidates_df)),
        "n_features": int(len(features)),
        "gt_used_for_candidate_labels_and_eval": True,
        "candidate_features_use_gt": False,
        "strict_outer_fold_cv_for_main_results": not args.smoke,
        "gtea_leave_one_video_robustness": (not args.smoke and "gtea" in args.datasets),
    }
    write_smoke_report(out_dir, candidates_df, oracle_df, parity)
    write_summary(out_dir, oracle_df, heldout_df, sig_df, swept_df, gates)

    payload = {
        "out_dir": str(out_dir),
        "datasets": args.datasets,
        "folds": args.folds,
        "boundary_deltas": BOUNDARY_DELTAS,
        "utility": "dAcc + 0.5*dF1@50 + 0.5*dEdit - 5*max(0,-dAcc)",
        "helpful_label": f"utility>{UTILITY_TAU} and delta_acc>=-{ACC_EPSILON}",
        "oracle_gates": gates,
        "feature_columns": features,
        "parity_flags": parity,
        "outputs": [
            "stage2_candidates.csv",
            "stage2_candidate_oracle_ceiling.csv",
            "stage2_learnability_heldout.csv",
            "stage2_swept_best_optimistic.csv",
            "stage2_significance.csv",
            "stage2_feature_importance.csv",
            "stage2_smoke_report.md",
            "stage2_summary.md",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Stage-2.0 learned edit-selector feasibility to {out_dir}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
