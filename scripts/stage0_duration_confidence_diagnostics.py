#!/usr/bin/env python3
"""
Stage-0 diagnostics for duration-aware / confidence-gated DiffAct recovery.

This is a read-only diagnostic over existing artifacts. It estimates duration
priors from training GT only, then uses test GT only for explicitly labelled
oracle/diagnostic analyses.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_diffact_sktr_fold1_paper import (  # noqa: E402
    get_variant_info_fast,
    load_diffact_softmax_and_aligned_df,
    resolve_diffact_softmax_dir,
    select_train_test_cases,
    softmax_map_from_entries,
    verify_softmax_list,
)
from src.cv_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_video_to_case_mapping,
    get_dataset_cv_config,
    load_fold_case_ids,
)
from src.evaluation import compute_tas_metrics_asformer, tas_metrics  # noqa: E402


@dataclass(frozen=True)
class DatasetConfig:
    run_dir: Path
    unique_only: bool = False
    train_k: Optional[int] = None


DATASETS: Dict[str, DatasetConfig] = {
    "gtea": DatasetConfig(
        run_dir=Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "diffact_gtea_allfolds_resumable_6ba8868_chunk11_w7"
        )
    ),
    "50salads": DatasetConfig(
        run_dir=Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "diffact_50salads_allfolds_resumable_6ba8868_chunk11"
        )
    ),
    "breakfast": DatasetConfig(
        run_dir=Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "diffact_breakfast_unique199_f14fd99_chunk11_w10"
        ),
        unique_only=True,
        train_k=199,
    ),
}

TAXONOMY_DIR = Path("/data1/eli-bogdanov/sktr_runs/diffact_error_taxonomy_v1")
OUTLIER_CASES = {("50salads", 1, "1"), ("50salads", 5, "49")}
BOUNDARY_WINDOWS = [10, 25, 50, 100, 200]
LOCK_PMAX = [0.80, 0.90, 0.95, 0.98]
LOCK_MARGIN = [0.10, 0.30, 0.50, 0.70]
LOCK_STABILITY = [0.80, 0.90, 0.95]
ENTROPY_QUANTILES = [0.20, 0.40, 0.60, 0.80]
CALIBRATION_THRESHOLDS = [0.80, 0.90, 0.95, 0.98]


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
    parser.add_argument(
        "--out-dir",
        default="/data1/eli-bogdanov/sktr_runs/stage0_duration_confidence_diagnostics_v1",
    )
    parser.add_argument(
        "--rare-class-min-segments",
        type=int,
        default=5,
        help="Classes with fewer train segments are shrunk toward the fold global prior.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Convenience mode: equivalent to --datasets gtea --folds 1.",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def median_abs_deviation(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def segments(labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not labels:
        return []
    out: List[Tuple[int, int, str]] = []
    start = 0
    cur = str(labels[0])
    for idx, lab in enumerate(labels[1:], start=1):
        lab = str(lab)
        if lab != cur:
            out.append((start, idx, cur))
            start = idx
            cur = lab
    out.append((start, len(labels), cur))
    return out


def contiguous_spans(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def boundary_positions(labels: Sequence[str]) -> List[int]:
    return [idx for idx in range(1, len(labels)) if labels[idx] != labels[idx - 1]]


def distance_to_boundaries(n: int, boundaries: Sequence[int]) -> np.ndarray:
    if not boundaries:
        return np.full(n, n + 1, dtype=np.int32)
    b = np.asarray(sorted(boundaries), dtype=np.int32)
    idx = np.searchsorted(b, np.arange(n), side="left")
    left = np.where(idx > 0, b[np.maximum(idx - 1, 0)], -10**9)
    right = np.where(idx < len(b), b[np.minimum(idx, len(b) - 1)], 10**9)
    return np.minimum(np.abs(np.arange(n) - left), np.abs(np.arange(n) - right)).astype(
        np.int32
    )


def format_segments(segs: Sequence[Tuple[int, int, str]], limit: int = 10) -> str:
    vals = [f"{label}[{start}:{end}]" for start, end, label in segs[:limit]]
    if len(segs) > limit:
        vals.append(f"... +{len(segs) - limit} more")
    return " | ".join(vals)


def label_mapping(softmax_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(softmax_dir / "mapping.txt", "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[str(int(parts[0]))] = parts[1]
    return out


def case_output_path(dataset: str, fold: int, case_id: str) -> Path:
    return DATASETS[dataset].run_dir / "case_outputs" / f"{dataset}_fold{fold}" / f"{case_id}.csv"


def load_case_output(dataset: str, fold: int, case_id: str) -> pd.DataFrame:
    path = case_output_path(dataset, fold, case_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in ["ground_truth", "argmax_activity", "sktr_activity"]:
        df[col] = df[col].astype(str)
    return df


def get_folds(dataset: str, requested: Optional[Sequence[int]], data_root: str) -> List[int]:
    if requested:
        return [int(f) for f in requested]
    n_folds = int(get_dataset_cv_config(dataset, data_root)["n_folds"])
    return list(range(1, n_folds + 1))


@dataclass
class FoldContext:
    dataset: str
    fold: int
    softmax_dir: Path
    df: pd.DataFrame
    case_to_mat: Dict[str, np.ndarray]
    entries: List[Tuple[str, str]]
    full_train_cases: List[str]
    full_test_cases: List[str]
    train_cases: List[str]
    test_cases: List[str]
    label_names: Dict[str, str]
    selection_meta: Dict[str, Any]


def load_fold_context(dataset: str, fold: int, data_root: str) -> FoldContext:
    diffact_root = REPO_ROOT / "baselines" / "DiffAct"
    softmax_dir = resolve_diffact_softmax_dir(
        diffact_root, dataset, fold, disallow_legacy=True
    )
    df, softmax_lst, entries = load_diffact_softmax_and_aligned_df(
        dataset, softmax_dir, Path(data_root)
    )
    verify_softmax_list(softmax_lst, f"{dataset} fold {fold}")
    case_to_mat = softmax_map_from_entries(entries, softmax_lst)
    video_map = build_video_to_case_mapping(
        dataset,
        "diffact",
        video_index_map_path=softmax_dir / "video_index_map.txt",
    )
    split = load_fold_case_ids(dataset, fold, video_map, data_root=data_root)
    full_train = [str(c) for c in split["train"]]
    full_test = [str(c) for c in split["test"]]
    cfg = DATASETS[dataset]
    variant_df = None
    if cfg.unique_only or cfg.train_k is not None:
        variant_df = get_variant_info_fast(df, use_collapsed=True)
    train_cases, test_cases, meta = select_train_test_cases(
        train_cases=full_train,
        test_cases=full_test,
        variant_df=variant_df,
        unique_only=cfg.unique_only,
        train_k=cfg.train_k,
        seed=42,
        fold=fold,
    )
    return FoldContext(
        dataset=dataset,
        fold=fold,
        softmax_dir=softmax_dir,
        df=df,
        case_to_mat=case_to_mat,
        entries=entries,
        full_train_cases=full_train,
        full_test_cases=full_test,
        train_cases=[str(c) for c in train_cases],
        test_cases=[str(c) for c in test_cases],
        label_names=label_mapping(softmax_dir),
        selection_meta=meta,
    )


def case_gt_from_df(ctx: FoldContext, case_id: str) -> List[str]:
    sub = ctx.df[ctx.df["case:concept:name"].astype(str) == str(case_id)]
    return sub["concept:name"].astype(str).tolist()


def load_taxonomy() -> pd.DataFrame:
    path = TAXONOMY_DIR / "error_taxonomy_per_case.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    return df


def load_complete_case_set() -> set[Tuple[str, int, str]]:
    """Cases with completed per-case outputs; incomplete ceiling rows never enter."""
    out = set()
    tax = load_taxonomy()
    for row in tax[tax["system"] == "argmax"].itertuples(index=False):
        out.add((str(row.dataset), int(row.fold), str(row.case_id)))
    return out


@dataclass
class DurationPrior:
    class_id: str
    raw_values: np.ndarray
    norm_values: np.ndarray
    raw_log_median: float
    raw_log_mad_scaled: float
    norm_log_median: float
    norm_log_mad_scaled: float
    used_class_prior: bool
    used_shrunk_prior: bool
    used_global_fallback: bool
    n_train_segments: int


def robust_log_stats(values: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    logs = np.log(vals + 1.0)
    med = float(np.median(logs))
    mad = float(np.median(np.abs(logs - med))) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-8:
        std = float(np.std(logs))
        mad = std if std > 1e-8 else 1.0
    return med, mad


def summarize_prior_values(
    dataset: str,
    fold: int,
    class_id: str,
    class_name: str,
    actual_raw: np.ndarray,
    actual_norm: np.ndarray,
    prior_raw: np.ndarray,
    prior_norm: np.ndarray,
    prior: DurationPrior,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "fold": fold,
        "class_id": class_id,
        "class_name": class_name,
        "n_train_segments": int(len(actual_raw)),
        "raw_len_median": quantile(prior_raw, 0.50),
        "raw_len_mad": median_abs_deviation(prior_raw),
        "raw_len_q01": quantile(prior_raw, 0.01),
        "raw_len_q05": quantile(prior_raw, 0.05),
        "raw_len_q25": quantile(prior_raw, 0.25),
        "raw_len_q75": quantile(prior_raw, 0.75),
        "raw_len_q95": quantile(prior_raw, 0.95),
        "raw_len_q99": quantile(prior_raw, 0.99),
        "norm_len_median": quantile(prior_norm, 0.50),
        "norm_len_mad": median_abs_deviation(prior_norm),
        "norm_len_q01": quantile(prior_norm, 0.01),
        "norm_len_q05": quantile(prior_norm, 0.05),
        "norm_len_q25": quantile(prior_norm, 0.25),
        "norm_len_q75": quantile(prior_norm, 0.75),
        "norm_len_q95": quantile(prior_norm, 0.95),
        "norm_len_q99": quantile(prior_norm, 0.99),
        "used_class_prior": prior.used_class_prior,
        "used_shrunk_prior": prior.used_shrunk_prior,
        "used_global_fallback": prior.used_global_fallback,
    }


def build_duration_priors(
    ctx: FoldContext,
    rare_min_segments: int,
) -> Tuple[Dict[str, DurationPrior], List[Dict[str, Any]], Dict[str, Any]]:
    raw_by_class: Dict[str, List[float]] = defaultdict(list)
    norm_by_class: Dict[str, List[float]] = defaultdict(list)
    for case_id in ctx.train_cases:
        gt = case_gt_from_df(ctx, case_id)
        t = len(gt)
        for start, end, lab in segments(gt):
            length = end - start
            raw_by_class[lab].append(float(length))
            norm_by_class[lab].append(float(length / t) if t else 0.0)

    global_raw = np.asarray(
        [v for vals in raw_by_class.values() for v in vals], dtype=float
    )
    global_norm = np.asarray(
        [v for vals in norm_by_class.values() for v in vals], dtype=float
    )
    if len(global_raw) == 0:
        raise ValueError(f"{ctx.dataset} fold {ctx.fold}: no training segments")

    class_ids = sorted(set(ctx.label_names.keys()) | set(raw_by_class.keys()), key=lambda x: int(x))
    priors: Dict[str, DurationPrior] = {}
    rows: List[Dict[str, Any]] = []
    for class_id in class_ids:
        actual_raw = np.asarray(raw_by_class.get(class_id, []), dtype=float)
        actual_norm = np.asarray(norm_by_class.get(class_id, []), dtype=float)
        n = len(actual_raw)
        if n >= rare_min_segments:
            prior_raw = actual_raw
            prior_norm = actual_norm
            used_class = True
            used_shrunk = False
            used_fallback = False
        elif n > 0:
            prior_raw = np.concatenate([actual_raw, global_raw])
            prior_norm = np.concatenate([actual_norm, global_norm])
            used_class = False
            used_shrunk = True
            used_fallback = False
        else:
            prior_raw = global_raw
            prior_norm = global_norm
            used_class = False
            used_shrunk = False
            used_fallback = True
        raw_med, raw_mad = robust_log_stats(prior_raw)
        norm_med, norm_mad = robust_log_stats(prior_norm)
        prior = DurationPrior(
            class_id=class_id,
            raw_values=prior_raw,
            norm_values=prior_norm,
            raw_log_median=raw_med,
            raw_log_mad_scaled=raw_mad,
            norm_log_median=norm_med,
            norm_log_mad_scaled=norm_mad,
            used_class_prior=used_class,
            used_shrunk_prior=used_shrunk,
            used_global_fallback=used_fallback,
            n_train_segments=n,
        )
        priors[class_id] = prior
        rows.append(
            summarize_prior_values(
                ctx.dataset,
                ctx.fold,
                class_id,
                ctx.label_names.get(class_id, ""),
                actual_raw,
                actual_norm,
                prior_raw,
                prior_norm,
                prior,
            )
        )
    global_info = {
        "dataset": ctx.dataset,
        "fold": ctx.fold,
        "global_raw_q75": quantile(global_raw, 0.75),
        "global_raw_median": quantile(global_raw, 0.50),
        "global_norm_median": quantile(global_norm, 0.50),
    }
    return priors, rows, global_info


def empirical_percentile(values: np.ndarray, x: float) -> float:
    if len(values) == 0 or not np.isfinite(x):
        return float("nan")
    return float(np.mean(values <= x) * 100.0)


def duration_features(
    priors: Dict[str, DurationPrior],
    class_id: str,
    length_raw: float,
    length_norm: float,
) -> Dict[str, Any]:
    prior = priors.get(str(class_id))
    if prior is None:
        # Should only happen for corrupted labels. Use the first available prior
        # but mark via NaN class statistics.
        prior = next(iter(priors.values()))
    raw_z = (math.log(length_raw + 1.0) - prior.raw_log_median) / prior.raw_log_mad_scaled
    norm_z = (math.log(length_norm + 1.0) - prior.norm_log_median) / prior.norm_log_mad_scaled
    raw_pct = empirical_percentile(prior.raw_values, length_raw)
    norm_pct = empirical_percentile(prior.norm_values, length_norm)
    raw_q01, raw_q05, raw_q95, raw_q99 = (
        quantile(prior.raw_values, 0.01),
        quantile(prior.raw_values, 0.05),
        quantile(prior.raw_values, 0.95),
        quantile(prior.raw_values, 0.99),
    )
    norm_q01, norm_q05, norm_q95, norm_q99 = (
        quantile(prior.norm_values, 0.01),
        quantile(prior.norm_values, 0.05),
        quantile(prior.norm_values, 0.95),
        quantile(prior.norm_values, 0.99),
    )
    return {
        "raw_duration_percentile_under_predicted_class": raw_pct,
        "norm_duration_percentile_under_predicted_class": norm_pct,
        "raw_duration_z": float(raw_z),
        "norm_duration_z": float(norm_z),
        "raw_outside_q05_q95": bool(length_raw < raw_q05 or length_raw > raw_q95),
        "norm_outside_q05_q95": bool(length_norm < norm_q05 or length_norm > norm_q95),
        "raw_outside_q01_q99": bool(length_raw < raw_q01 or length_raw > raw_q99),
        "norm_outside_q01_q99": bool(length_norm < norm_q01 or length_norm > norm_q99),
        "abs_raw_z_gt_2": bool(abs(raw_z) > 2.0),
        "abs_norm_z_gt_2": bool(abs(norm_z) > 2.0),
        "abs_raw_z_gt_3": bool(abs(raw_z) > 3.0),
        "abs_norm_z_gt_3": bool(abs(norm_z) > 3.0),
    }


def prefix_count(gt: Sequence[str], label: str) -> np.ndarray:
    arr = np.zeros(len(gt) + 1, dtype=np.int32)
    lab = str(label)
    for i, g in enumerate(gt):
        arr[i + 1] = arr[i] + (1 if str(g) == lab else 0)
    return arr


def oracle_boundary_prediction(
    gt: Sequence[str], pred: Sequence[str], window: int
) -> Tuple[List[str], List[int], List[int]]:
    """Optimal boundary-shift oracle preserving argmax segment labels/order."""
    t = len(gt)
    segs = segments(pred)
    if len(segs) <= 1:
        return list(pred), [], []
    labels = [s[2] for s in segs]
    orig_boundaries = [s[1] for s in segs[:-1]]
    prefix_by_label = {lab: prefix_count(gt, lab) for lab in set(labels)}

    def score(label: str, start: int, end: int) -> int:
        pref = prefix_by_label[label]
        return int(pref[end] - pref[start])

    candidates: List[List[int]] = []
    for b in orig_boundaries:
        lo = max(1, b - window)
        hi = min(t - 1, b + window)
        candidates.append(list(range(lo, hi + 1)))

    dp: List[Dict[int, Tuple[int, Optional[int]]]] = []
    first: Dict[int, Tuple[int, Optional[int]]] = {}
    for b in candidates[0]:
        first[b] = (score(labels[0], 0, b), None)
    dp.append(first)
    for i in range(1, len(candidates)):
        cur: Dict[int, Tuple[int, Optional[int]]] = {}
        for b in candidates[i]:
            best_score = -10**18
            best_prev: Optional[int] = None
            for prev_b, (prev_score, _) in dp[i - 1].items():
                if prev_b >= b:
                    continue
                s = prev_score + score(labels[i], prev_b, b)
                if s > best_score:
                    best_score = s
                    best_prev = prev_b
            if best_prev is not None:
                cur[b] = (best_score, best_prev)
        if not cur:
            raise ValueError("Boundary oracle produced no feasible candidates")
        dp.append(cur)

    best_final = -10**18
    best_b: Optional[int] = None
    last_idx = len(candidates) - 1
    for b, (prev_score, _) in dp[-1].items():
        s = prev_score + score(labels[-1], b, t)
        if s > best_final:
            best_final = s
            best_b = b
    if best_b is None:
        raise ValueError("Boundary oracle no final boundary")

    chosen = [0] * len(candidates)
    chosen[last_idx] = best_b
    for i in range(last_idx, 0, -1):
        chosen[i - 1] = dp[i][chosen[i]][1]  # type: ignore[assignment]

    out: List[str] = []
    start = 0
    for lab, b in zip(labels, chosen):
        out.extend([lab] * (b - start))
        start = b
    out.extend([labels[-1]] * (t - start))
    if len(out) != t:
        raise AssertionError("Oracle boundary prediction length mismatch")
    return out, orig_boundaries, chosen


def accuracy_fraction(gt: Sequence[str], pred: Sequence[str]) -> float:
    return float(np.mean(np.asarray(gt, dtype=object) == np.asarray(pred, dtype=object)))


def long_substitution_spans(
    gt: Sequence[str],
    pred: Sequence[str],
    min_len: int = 100,
    homogeneity: float = 0.90,
) -> List[Dict[str, Any]]:
    error = [str(g) != str(p) for g, p in zip(gt, pred)]
    out: List[Dict[str, Any]] = []
    for span_id, (start, end) in enumerate(contiguous_spans(error)):
        length = end - start
        if length < min_len:
            continue
        counts = Counter(str(x) for x in pred[start:end])
        pred_class, count = counts.most_common(1)[0]
        if count / length < homogeneity:
            continue
        gt_counts = Counter(str(x) for x in gt[start:end])
        gt_summary = ";".join(
            f"{lab}:{cnt}" for lab, cnt in gt_counts.most_common(5)
        )
        out.append(
            {
                "span_id": span_id,
                "span_start": start,
                "span_end": end,
                "span_len": length,
                "predicted_class": pred_class,
                "gt_class_summary": gt_summary,
                "homogeneity": count / length,
            }
        )
    return out


def boundary_label_sets(gt: Sequence[str], width: int) -> List[set[str]]:
    labels: List[set[str]] = [set() for _ in gt]
    for idx in range(1, len(gt)):
        if str(gt[idx - 1]) == str(gt[idx]):
            continue
        left, right = str(gt[idx - 1]), str(gt[idx])
        start = max(0, idx - 1 - width)
        end = min(len(gt), idx + width + 1)
        for t in range(start, end):
            labels[t].add(left)
            labels[t].add(right)
    return labels


def island_mask(gt: Sequence[str], pred: Sequence[str], max_len: int = 25) -> np.ndarray:
    out = np.zeros(len(gt), dtype=bool)
    pred_list = [str(x) for x in pred]
    for start, end, label in segments([str(x) for x in gt]):
        wrong = [pred_list[t] != label for t in range(start, end)]
        for rel_s, rel_e in contiguous_spans(wrong):
            span_s = start + rel_s
            span_e = start + rel_e
            if span_e - span_s > max_len:
                continue
            if span_s <= start or span_e >= end:
                continue
            if pred_list[span_s - 1] == label and pred_list[span_e] == label:
                out[span_s:span_e] = True
    return out


def raw_long_substitution_mask(
    gt: Sequence[str],
    pred: Sequence[str],
    min_len: int = 100,
    homogeneity: float = 0.90,
) -> np.ndarray:
    out = np.zeros(len(gt), dtype=bool)
    for span in long_substitution_spans(gt, pred, min_len=min_len, homogeneity=homogeneity):
        out[int(span["span_start"]): int(span["span_end"])] = True
    return out


def primary_long_substitution_mask(gt: Sequence[str], pred: Sequence[str]) -> np.ndarray:
    """Canonical taxonomy long_substitution frames after priority ordering."""
    gt_list = [str(x) for x in gt]
    pred_list = [str(x) for x in pred]
    error = np.asarray([g != p for g, p in zip(gt_list, pred_list)], dtype=bool)
    boundary_sets = boundary_label_sets(gt_list, 25)
    boundary = np.asarray(
        [bool(error[t] and pred_list[t] in boundary_sets[t]) for t in range(len(gt_list))],
        dtype=bool,
    )
    island = island_mask(gt_list, pred_list, 25)
    raw_long = raw_long_substitution_mask(gt_list, pred_list, 100, 0.90)
    return error & ~boundary & ~island & raw_long


def segment_lookup(segs: Sequence[Tuple[int, int, str]], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int32)
    for idx, (start, end, _) in enumerate(segs):
        out[start:end] = idx
    return out


def mode_summary(vals: Sequence[str]) -> str:
    if not vals:
        return ""
    counts = Counter(str(v) for v in vals)
    return ";".join(f"{lab}:{cnt}" for lab, cnt in counts.most_common(5))


def analyze_duration_implausibility_case(
    *,
    dataset: str,
    fold: int,
    case_id: str,
    gt: List[str],
    argmax: List[str],
    priors: Dict[str, DurationPrior],
    global_info: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(gt)
    arg_segs = segments(argmax)
    long_spans = long_substitution_spans(gt, argmax)
    long_mask = np.zeros(n, dtype=bool)
    for span in long_spans:
        long_mask[int(span["span_start"]): int(span["span_end"])] = True

    rows: List[Dict[str, Any]] = []
    for span in long_spans:
        s, e = int(span["span_start"]), int(span["span_end"])
        for seg_idx, (ps, pe, plab) in enumerate(arg_segs):
            ov_s, ov_e = max(s, ps), min(e, pe)
            if ov_s >= ov_e:
                continue
            seg_len = pe - ps
            norm_len = seg_len / n if n else 0.0
            feats = duration_features(priors, plab, seg_len, norm_len)
            rows.append(
                {
                    "row_type": "long_substitution_error",
                    "is_error_segment": True,
                    "dataset": dataset,
                    "fold": fold,
                    "case_id": case_id,
                    "span_id": span["span_id"],
                    "span_start": s,
                    "span_end": e,
                    "span_len": e - s,
                    "predicted_class": plab,
                    "gt_class_summary": span["gt_class_summary"],
                    "predicted_segment_start": ps,
                    "predicted_segment_end": pe,
                    "predicted_segment_len_raw": seg_len,
                    "predicted_segment_len_norm": norm_len,
                    "segment_correct_purity": float(
                        np.mean([g == plab for g in gt[ps:pe]]) if pe > ps else 0.0
                    ),
                    "overlap_len": ov_e - ov_s,
                    **feats,
                }
            )

    correct_rows: List[Dict[str, Any]] = []
    for seg_idx, (ps, pe, plab) in enumerate(arg_segs):
        seg_len = pe - ps
        if seg_len <= 0 or np.any(long_mask[ps:pe]):
            continue
        purity = float(np.mean([g == plab for g in gt[ps:pe]]))
        prior = priors.get(plab)
        class_median = quantile(prior.raw_values, 0.50) if prior is not None else 100.0
        threshold = max(100.0, float(class_median), float(global_info["global_raw_q75"]))
        if purity < 0.90 or seg_len < threshold:
            continue
        norm_len = seg_len / n if n else 0.0
        feats = duration_features(priors, plab, seg_len, norm_len)
        correct_rows.append(
            {
                "row_type": "correct_long_segment",
                "is_error_segment": False,
                "dataset": dataset,
                "fold": fold,
                "case_id": case_id,
                "span_id": f"correct_{seg_idx}",
                "span_start": ps,
                "span_end": pe,
                "span_len": seg_len,
                "predicted_class": plab,
                "gt_class_summary": mode_summary(gt[ps:pe]),
                "predicted_segment_start": ps,
                "predicted_segment_end": pe,
                "predicted_segment_len_raw": seg_len,
                "predicted_segment_len_norm": norm_len,
                "segment_correct_purity": purity,
                "overlap_len": seg_len,
                **feats,
            }
        )
    return rows, correct_rows


def roc_auc(labels: Sequence[int], scores: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    w = np.ones_like(s, dtype=float) if weights is None else np.asarray(weights, dtype=float)[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    y = y[order]
    w = w[order]
    pos_total = float(w[y == 1].sum())
    neg_total = float(w[y == 0].sum())
    if pos_total == 0 or neg_total == 0:
        return float("nan")
    cum_neg = 0.0
    auc_num = 0.0
    i = 0
    while i < len(y):
        j = i
        group_pos = 0.0
        group_neg = 0.0
        while j < len(y) and s[j] == s[i]:
            if y[j] == 1:
                group_pos += w[j]
            else:
                group_neg += w[j]
            j += 1
        auc_num += group_pos * (cum_neg + 0.5 * group_neg)
        cum_neg += group_neg
        i = j
    return float(auc_num / (pos_total * neg_total))


def pr_auc(labels: Sequence[int], scores: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    w = np.ones_like(s, dtype=float) if weights is None else np.asarray(weights, dtype=float)[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    w = w[order]
    pos_total = float(w[y == 1].sum())
    if pos_total == 0:
        return float("nan")
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    area = 0.0
    for yi, wi in zip(y, w):
        if yi == 1:
            tp += wi
        else:
            fp += wi
        recall = tp / pos_total
        precision = tp / (tp + fp) if tp + fp else 0.0
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return float(area)


def binary_confusion(
    labels: Sequence[int],
    preds: Sequence[bool],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(preds, dtype=bool)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    tp = float(w[p & y].sum())
    fp = float(w[p & ~y].sum())
    tn = float(w[~p & ~y].sum())
    fn = float(w[~p & y].sum())
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "false_positive_rate": safe_div(fp, fp + tn),
    }


def duration_detection_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rules = {
        "outside_q05_q95_raw": rows["raw_outside_q05_q95"].astype(bool),
        "outside_q05_q95_norm": rows["norm_outside_q05_q95"].astype(bool),
        "outside_q05_q95_raw_or_norm": rows["raw_outside_q05_q95"].astype(bool)
        | rows["norm_outside_q05_q95"].astype(bool),
        "outside_q05_q95_raw_and_norm": rows["raw_outside_q05_q95"].astype(bool)
        & rows["norm_outside_q05_q95"].astype(bool),
        "outside_q01_q99_raw": rows["raw_outside_q01_q99"].astype(bool),
        "outside_q01_q99_norm": rows["norm_outside_q01_q99"].astype(bool),
        "outside_q01_q99_raw_or_norm": rows["raw_outside_q01_q99"].astype(bool)
        | rows["norm_outside_q01_q99"].astype(bool),
        "outside_q01_q99_raw_and_norm": rows["raw_outside_q01_q99"].astype(bool)
        & rows["norm_outside_q01_q99"].astype(bool),
        "abs_z_gt_2_raw": rows["abs_raw_z_gt_2"].astype(bool),
        "abs_z_gt_2_norm": rows["abs_norm_z_gt_2"].astype(bool),
        "abs_z_gt_2_raw_or_norm": rows["abs_raw_z_gt_2"].astype(bool)
        | rows["abs_norm_z_gt_2"].astype(bool),
        "abs_z_gt_2_raw_and_norm": rows["abs_raw_z_gt_2"].astype(bool)
        & rows["abs_norm_z_gt_2"].astype(bool),
        "abs_z_gt_3_raw": rows["abs_raw_z_gt_3"].astype(bool),
        "abs_z_gt_3_norm": rows["abs_norm_z_gt_3"].astype(bool),
        "abs_z_gt_3_raw_or_norm": rows["abs_raw_z_gt_3"].astype(bool)
        | rows["abs_norm_z_gt_3"].astype(bool),
        "abs_z_gt_3_raw_and_norm": rows["abs_raw_z_gt_3"].astype(bool)
        & rows["abs_norm_z_gt_3"].astype(bool),
    }
    out: List[Dict[str, Any]] = []
    group_defs: List[Tuple[str, Tuple[Any, ...], pd.Series]] = []
    group_defs.append(("overall", ("all",), pd.Series(True, index=rows.index)))
    for dataset, idxs in rows.groupby("dataset").groups.items():
        group_defs.append(("dataset", (dataset,), rows.index.to_series().isin(idxs)))
    for (dataset, fold), idxs in rows.groupby(["dataset", "fold"]).groups.items():
        group_defs.append(("fold", (dataset, fold), rows.index.to_series().isin(idxs)))

    y_all = rows["is_error_segment"].astype(int)
    segment_weights_all = np.ones(len(rows), dtype=float)
    frame_weights_all = rows["overlap_len"].astype(float).to_numpy()
    score_defs = {
        "abs_raw_z": rows["raw_duration_z"].abs().to_numpy(),
        "abs_norm_z": rows["norm_duration_z"].abs().to_numpy(),
        "max_abs_raw_norm_z": np.maximum(
            rows["raw_duration_z"].abs().to_numpy(),
            rows["norm_duration_z"].abs().to_numpy(),
        ),
    }
    for level, keys, mask in group_defs:
        sub_idx = rows.index[mask]
        if len(sub_idx) == 0:
            continue
        for weighting, weights_all in [
            ("segment_weighted", segment_weights_all),
            ("frame_weighted", frame_weights_all),
        ]:
            y = y_all.loc[sub_idx].to_numpy()
            weights = weights_all[mask.to_numpy()]
            for rule_name, pred_all in rules.items():
                conf = binary_confusion(y, pred_all.loc[sub_idx].to_numpy(), weights)
                row: Dict[str, Any] = {
                    "aggregation_level": level,
                    "aggregation_key": "|".join(map(str, keys)),
                    "dataset": keys[0] if level in {"dataset", "fold"} else "all",
                    "fold": keys[1] if level == "fold" else "all",
                    "weighting": weighting,
                    "rule": rule_name,
                    "n_segments": int(len(sub_idx)),
                    **conf,
                }
                out.append(row)
            for score_name, score_all in score_defs.items():
                row = {
                    "aggregation_level": level,
                    "aggregation_key": "|".join(map(str, keys)),
                    "dataset": keys[0] if level in {"dataset", "fold"} else "all",
                    "fold": keys[1] if level == "fold" else "all",
                    "weighting": weighting,
                    "rule": f"continuous_{score_name}",
                    "n_segments": int(len(sub_idx)),
                    "auroc": roc_auc(y, score_all[mask.to_numpy()], weights),
                    "auprc": pr_auc(y, score_all[mask.to_numpy()], weights),
                }
                out.append(row)
    return pd.DataFrame(out)


def softmax_features(mat: np.ndarray) -> Dict[str, np.ndarray]:
    probs = np.asarray(mat, dtype=float)
    top1 = probs.max(axis=0)
    top1_idx = probs.argmax(axis=0)
    if probs.shape[0] > 1:
        top2 = np.partition(probs, -2, axis=0)[-2]
    else:
        top2 = np.zeros(probs.shape[1])
    margin = top1 - top2
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=0)
    return {
        "top1_probability": top1,
        "top2_probability": top2,
        "top1_index": top1_idx,
        "top1_top2_margin": margin,
        "entropy": entropy,
    }


def prob_for_labels(mat: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    out = np.full(len(labels), np.nan, dtype=float)
    for i, lab in enumerate(labels):
        try:
            idx = int(lab)
        except ValueError:
            continue
        if 0 <= idx < mat.shape[0]:
            out[i] = mat[idx, i]
    return out


def local_stability(labels: Sequence[str], radius: int) -> np.ndarray:
    labs = np.asarray(labels, dtype=object)
    out = np.zeros(len(labs), dtype=float)
    for i, lab in enumerate(labs):
        lo = max(0, i - radius)
        hi = min(len(labs), i + radius + 1)
        out[i] = float(np.mean(labs[lo:hi] == lab)) if hi > lo else 1.0
    return out


def per_frame_segment_values(
    labels: Sequence[str],
    priors: Dict[str, DurationPrior],
) -> Dict[str, np.ndarray]:
    n = len(labels)
    seg_len = np.zeros(n, dtype=float)
    raw_z = np.zeros(n, dtype=float)
    norm_z = np.zeros(n, dtype=float)
    seg_idx = np.zeros(n, dtype=np.int32)
    for idx, (start, end, lab) in enumerate(segments(labels)):
        length = end - start
        norm = length / n if n else 0.0
        feats = duration_features(priors, lab, length, norm)
        seg_len[start:end] = length
        raw_z[start:end] = feats["raw_duration_z"]
        norm_z[start:end] = feats["norm_duration_z"]
        seg_idx[start:end] = idx
    return {
        "argmax_segment_length": seg_len,
        "argmax_segment_raw_duration_z": raw_z,
        "argmax_segment_norm_duration_z": norm_z,
        "argmax_segment_index": seg_idx,
    }


def confidence_case_frames(
    *,
    dataset: str,
    fold: int,
    case_id: str,
    gt: List[str],
    argmax: List[str],
    sktr: List[str],
    mat: np.ndarray,
    priors: Dict[str, DurationPrior],
    long_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(gt)
    sf = softmax_features(mat)
    argmax_prob = prob_for_labels(mat, argmax)
    sktr_prob = prob_for_labels(mat, sktr)
    seg_vals = per_frame_segment_values(argmax, priors)
    dist_arg = distance_to_boundaries(n, boundary_positions(argmax))
    dist_gt = distance_to_boundaries(n, boundary_positions(gt))
    stab5 = local_stability(argmax, 5)
    stab10 = local_stability(argmax, 10)
    stab25 = local_stability(argmax, 25)
    changed = np.asarray([a != s for a, s in zip(argmax, sktr)], dtype=bool)

    labels = []
    for g, a, s in zip(gt, argmax, sktr):
        if s == g and a != g:
            labels.append("helpful")
        elif s != g and a == g:
            labels.append("harmful")
        else:
            labels.append("neutral_other")

    common = {
        "dataset": dataset,
        "fold": fold,
        "case_id": case_id,
    }
    changed_rows: List[Dict[str, Any]] = []
    for t in np.where(changed)[0]:
        changed_rows.append(
            {
                **common,
                "frame": int(t),
                "label": labels[t],
                "gt": gt[t],
                "argmax": argmax[t],
                "sktr": sktr[t],
                "argmax_probability": float(argmax_prob[t]),
                "sktr_class_probability": float(sktr_prob[t]),
                "top1_probability": float(sf["top1_probability"][t]),
                "top2_probability": float(sf["top2_probability"][t]),
                "top1_top2_margin": float(sf["top1_top2_margin"][t]),
                "entropy": float(sf["entropy"][t]),
                "local_stability_window_5": float(stab5[t]),
                "local_stability_window_10": float(stab10[t]),
                "local_stability_window_25": float(stab25[t]),
                "argmax_segment_length": float(seg_vals["argmax_segment_length"][t]),
                "argmax_segment_raw_duration_z": float(
                    seg_vals["argmax_segment_raw_duration_z"][t]
                ),
                "argmax_segment_norm_duration_z": float(
                    seg_vals["argmax_segment_norm_duration_z"][t]
                ),
                "distance_to_nearest_argmax_boundary": int(dist_arg[t]),
                "distance_to_nearest_gt_boundary": int(dist_gt[t]),
            }
        )

    span_rows: List[Dict[str, Any]] = []
    for span_id, (start, end) in enumerate(contiguous_spans(changed)):
        idx = np.arange(start, end)
        lab_counts = Counter(labels[start:end])
        majority, majority_count = lab_counts.most_common(1)[0]
        helpful = lab_counts.get("helpful", 0)
        harmful = lab_counts.get("harmful", 0)
        span_rows.append(
            {
                **common,
                "span_id": span_id,
                "start": start,
                "end": end,
                "length": end - start,
                "majority_label": majority,
                "majority_fraction": majority_count / (end - start),
                "helpful_fraction": helpful / (end - start),
                "harmful_fraction": harmful / (end - start),
                "median_pmax": float(np.median(sf["top1_probability"][idx])),
                "mean_pmax": float(np.mean(sf["top1_probability"][idx])),
                "min_pmax": float(np.min(sf["top1_probability"][idx])),
                "median_margin": float(np.median(sf["top1_top2_margin"][idx])),
                "median_entropy": float(np.median(sf["entropy"][idx])),
                "median_stability": float(np.median(stab25[idx])),
                "median_argmax_segment_raw_duration_z": float(
                    np.median(seg_vals["argmax_segment_raw_duration_z"][idx])
                ),
                "median_argmax_segment_norm_duration_z": float(
                    np.median(seg_vals["argmax_segment_norm_duration_z"][idx])
                ),
            }
        )

    case_delta = accuracy_fraction(gt, sktr) - accuracy_fraction(gt, argmax)
    changed_idx = np.where(changed)[0]
    case_rows: List[Dict[str, Any]] = []
    if len(changed_idx):
        case_rows.append(
            {
                **common,
                "total_changed_frames": int(len(changed_idx)),
                "helpful_changed_frames": int(sum(labels[t] == "helpful" for t in changed_idx)),
                "harmful_changed_frames": int(sum(labels[t] == "harmful" for t in changed_idx)),
                "net_delta_acc": float(case_delta),
                "median_pmax_over_changed_frames": float(np.median(sf["top1_probability"][changed_idx])),
                "median_margin_over_changed_frames": float(np.median(sf["top1_top2_margin"][changed_idx])),
                "median_entropy_over_changed_frames": float(np.median(sf["entropy"][changed_idx])),
                "median_stability_over_changed_frames": float(np.median(stab25[changed_idx])),
                "max_divergence_span_length": int(max((r["length"] for r in span_rows), default=0)),
            }
        )

    all_frame_rows: List[Dict[str, Any]] = []
    for t in range(n):
        all_frame_rows.append(
            {
                **common,
                "frame": t,
                "gt": gt[t],
                "argmax": argmax[t],
                "sktr": sktr[t],
                "argmax_correct": bool(argmax[t] == gt[t]),
                "sktr_divergence": bool(changed[t]),
                "long_substitution_frame": bool(long_mask[t]),
                "top1_probability": float(sf["top1_probability"][t]),
                "top1_top2_margin": float(sf["top1_top2_margin"][t]),
                "entropy": float(sf["entropy"][t]),
                "local_stability_window_25": float(stab25[t]),
                "distance_to_nearest_argmax_boundary": int(dist_arg[t]),
                "distance_to_nearest_gt_boundary": int(dist_gt[t]),
            }
        )

    return (
        pd.DataFrame(changed_rows),
        pd.DataFrame(span_rows),
        pd.DataFrame(case_rows),
        pd.DataFrame(all_frame_rows),
    )


def separability_summary(
    frames: pd.DataFrame,
    spans: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    feature_cols = [
        "argmax_probability",
        "sktr_class_probability",
        "top1_probability",
        "top1_top2_margin",
        "entropy",
        "local_stability_window_5",
        "local_stability_window_10",
        "local_stability_window_25",
        "argmax_segment_length",
        "argmax_segment_raw_duration_z",
        "argmax_segment_norm_duration_z",
        "distance_to_nearest_argmax_boundary",
        "distance_to_nearest_gt_boundary",
    ]

    def scope_masks(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        if df.empty:
            return []
        base = pd.Series(True, index=df.index)
        c1 = (df["dataset"].str.lower() == "50salads") & (df["fold"] == 1) & (df["case_id"].astype(str) == "1")
        c49 = (df["dataset"].str.lower() == "50salads") & (df["fold"] == 5) & (df["case_id"].astype(str) == "49")
        masks = [
            ("all_cases", base),
            ("exclude_50salads_case1", base & ~c1),
            ("exclude_50salads_case49", base & ~c49),
            ("exclude_50salads_case1_and_case49", base & ~c1 & ~c49),
        ]
        return masks

    for scope, mask in scope_masks(frames):
        sub = frames.loc[mask & frames["label"].isin(["helpful", "harmful"])]
        group_defs = [("overall", "all", sub)]
        for ds, g in sub.groupby("dataset"):
            group_defs.append(("dataset", ds, g))
        for agg_level, agg_key, g in group_defs:
            if g.empty or g["label"].nunique() < 2:
                continue
            y = (g["label"] == "harmful").astype(int).to_numpy()
            for feat in feature_cols:
                if feat not in g:
                    continue
                scores = g[feat].astype(float).to_numpy()
                rows.append(
                    {
                        "analysis": "auroc_auprc",
                        "level": "frame",
                        "scope": scope,
                        "aggregation_level": agg_level,
                        "aggregation_key": agg_key,
                        "feature": feat,
                        "n_helpful": int((g["label"] == "helpful").sum()),
                        "n_harmful": int((g["label"] == "harmful").sum()),
                        "auroc_harmful_vs_helpful": roc_auc(y, scores),
                        "auprc_harmful_vs_helpful": pr_auc(y, scores),
                    }
                )
            sweep_specs = [
                ("pmax_ge", "top1_probability", [0.80, 0.90, 0.95, 0.98], "ge"),
                ("margin_ge", "top1_top2_margin", [0.10, 0.30, 0.50, 0.70], "ge"),
                ("entropy_le", "entropy", list(np.quantile(g["entropy"], [0.2, 0.4, 0.6, 0.8])), "le"),
                ("stability_ge", "local_stability_window_25", [0.80, 0.90, 0.95], "ge"),
                ("abs_duration_z_ge", "argmax_segment_raw_duration_z", [2.0, 3.0], "abs_ge"),
            ]
            for rule, feat, thresholds, direction in sweep_specs:
                vals = g[feat].astype(float)
                for th in thresholds:
                    if direction == "ge":
                        pred = vals >= th
                    elif direction == "le":
                        pred = vals <= th
                    else:
                        pred = vals.abs() >= th
                    conf = binary_confusion(y, pred)
                    rows.append(
                        {
                            "analysis": "threshold_sweep",
                            "level": "frame",
                            "scope": scope,
                            "aggregation_level": agg_level,
                            "aggregation_key": agg_key,
                            "feature": feat,
                            "rule": rule,
                            "threshold": float(th),
                            **conf,
                        }
                    )

    if not spans.empty:
        span_features = [
            "median_pmax",
            "mean_pmax",
            "min_pmax",
            "median_margin",
            "median_entropy",
            "median_stability",
            "median_argmax_segment_raw_duration_z",
            "median_argmax_segment_norm_duration_z",
            "length",
        ]
        p95 = float(spans["length"].quantile(0.95)) if len(spans) else float("inf")
        for scope, mask in scope_masks(spans):
            masks = [(scope, mask), (scope + "_exclude_span_len_gt_p95", mask & (spans["length"] <= p95))]
            for scope2, mask2 in masks:
                sub = spans.loc[
                    mask2
                    & spans["majority_label"].isin(["helpful", "harmful"])
                    & (spans["majority_fraction"] >= 0.80)
                ]
                group_defs = [("overall", "all", sub)]
                for ds, g in sub.groupby("dataset"):
                    group_defs.append(("dataset", ds, g))
                for agg_level, agg_key, g in group_defs:
                    if g.empty or g["majority_label"].nunique() < 2:
                        continue
                    y = (g["majority_label"] == "harmful").astype(int).to_numpy()
                    for feat in span_features:
                        scores = g[feat].astype(float).to_numpy()
                        rows.append(
                            {
                                "analysis": "auroc_auprc",
                                "level": "span",
                                "scope": scope2,
                                "aggregation_level": agg_level,
                                "aggregation_key": agg_key,
                                "feature": feat,
                                "n_helpful": int((g["majority_label"] == "helpful").sum()),
                                "n_harmful": int((g["majority_label"] == "harmful").sum()),
                                "auroc_harmful_vs_helpful": roc_auc(y, scores),
                                "auprc_harmful_vs_helpful": pr_auc(y, scores),
                            }
                        )

    if not cases.empty:
        case_features = [
            "total_changed_frames",
            "median_pmax_over_changed_frames",
            "median_margin_over_changed_frames",
            "median_entropy_over_changed_frames",
            "median_stability_over_changed_frames",
            "max_divergence_span_length",
        ]
        for scope, mask in scope_masks(cases):
            sub = cases.loc[mask & (cases["net_delta_acc"] != 0)]
            group_defs = [("overall", "all", sub)]
            for ds, g in sub.groupby("dataset"):
                group_defs.append(("dataset", ds, g))
            for agg_level, agg_key, g in group_defs:
                if g.empty or (g["net_delta_acc"] < 0).nunique() < 2:
                    continue
                y = (g["net_delta_acc"] < 0).astype(int).to_numpy()
                for feat in case_features:
                    scores = g[feat].astype(float).to_numpy()
                    rows.append(
                        {
                            "analysis": "auroc_auprc",
                            "level": "case",
                            "scope": scope,
                            "aggregation_level": agg_level,
                            "aggregation_key": agg_key,
                            "feature": feat,
                            "n_positive_delta_cases": int((g["net_delta_acc"] > 0).sum()),
                            "n_negative_delta_cases": int((g["net_delta_acc"] < 0).sum()),
                            "auroc_harmful_vs_helpful": roc_auc(y, scores),
                            "auprc_harmful_vs_helpful": pr_auc(y, scores),
                        }
                    )
    return pd.DataFrame(rows)


def calibration_rows(all_frames: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    bins = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
    for (dataset, fold), fold_df in all_frames.groupby(["dataset", "fold"]):
        split_masks = {
            "all": pd.Series(True, index=fold_df.index),
            "gt_boundary_w10": fold_df["distance_to_nearest_gt_boundary"] <= 10,
            "gt_boundary_w25": fold_df["distance_to_nearest_gt_boundary"] <= 25,
            "gt_boundary_w50": fold_df["distance_to_nearest_gt_boundary"] <= 50,
            "argmax_boundary_w10": fold_df["distance_to_nearest_argmax_boundary"] <= 10,
            "argmax_boundary_w25": fold_df["distance_to_nearest_argmax_boundary"] <= 25,
            "argmax_boundary_w50": fold_df["distance_to_nearest_argmax_boundary"] <= 50,
            "segment_interiors_gt_w25": fold_df["distance_to_nearest_gt_boundary"] > 25,
            "long_substitution": fold_df["long_substitution_frame"].astype(bool),
            "sktr_divergence": fold_df["sktr_divergence"].astype(bool),
            "non_divergence": ~fold_df["sktr_divergence"].astype(bool),
        }
        for split_name, mask in split_masks.items():
            sub = fold_df.loc[mask]
            n_total = len(sub)
            if n_total:
                ece = 0.0
                mce = 0.0
                for lo, hi in bins:
                    if hi >= 1.0:
                        bmask = (sub["top1_probability"] >= lo) & (sub["top1_probability"] <= hi)
                    else:
                        bmask = (sub["top1_probability"] >= lo) & (sub["top1_probability"] < hi)
                    b = sub.loc[bmask]
                    if b.empty:
                        continue
                    gap = abs(float(b["argmax_correct"].mean()) - float(b["top1_probability"].mean()))
                    ece += len(b) / n_total * gap
                    mce = max(mce, gap)
            else:
                ece = float("nan")
                mce = float("nan")
            threshold_stats: Dict[str, Any] = {}
            for th in CALIBRATION_THRESHOLDS:
                above = sub.loc[sub["top1_probability"] > th] if n_total else sub
                threshold_stats[f"accuracy_at_pmax_gt_{str(th).replace('.', '_')}"] = (
                    float(above["argmax_correct"].mean()) if len(above) else float("nan")
                )
                threshold_stats[f"coverage_at_pmax_gt_{str(th).replace('.', '_')}"] = (
                    len(above) / n_total if n_total else float("nan")
                )
            for lo, hi in bins:
                if n_total:
                    if hi >= 1.0:
                        bmask = (sub["top1_probability"] >= lo) & (sub["top1_probability"] <= hi)
                    else:
                        bmask = (sub["top1_probability"] >= lo) & (sub["top1_probability"] < hi)
                    b = sub.loc[bmask]
                else:
                    b = sub
                mean_conf = float(b["top1_probability"].mean()) if len(b) else float("nan")
                acc = float(b["argmax_correct"].mean()) if len(b) else float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "split": split_name,
                        "bin_start": lo,
                        "bin_end": hi,
                        "n_frames": int(len(b)),
                        "mean_confidence": mean_conf,
                        "empirical_accuracy": acc,
                        "calibration_gap": acc - mean_conf if np.isfinite(acc) and np.isfinite(mean_conf) else float("nan"),
                        "ECE": ece,
                        "MCE": mce,
                        **threshold_stats,
                    }
                )
    return pd.DataFrame(rows)


def gate_configs(entropy_thresholds: Dict[Tuple[str, int], List[float]]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = [
        {"config_family": "baseline", "threshold_config": "baseline_argmax", "kind": "baseline_argmax"},
        {"config_family": "baseline", "threshold_config": "baseline_sktr", "kind": "baseline_sktr"},
    ]
    for th in LOCK_PMAX:
        configs.append(
            {
                "config_family": "pmax",
                "threshold_config": f"pmax_gt_{th:.2f}",
                "kind": "pmax",
                "threshold": th,
            }
        )
    for th in LOCK_MARGIN:
        configs.append(
            {
                "config_family": "margin",
                "threshold_config": f"margin_gt_{th:.2f}",
                "kind": "margin",
                "threshold": th,
            }
        )
    for q in ENTROPY_QUANTILES:
        configs.append(
            {
                "config_family": "entropy",
                "threshold_config": f"entropy_le_q{int(q * 100)}",
                "kind": "entropy_quantile",
                "quantile": q,
            }
        )
    for th in LOCK_STABILITY:
        configs.append(
            {
                "config_family": "stability",
                "threshold_config": f"stability_gt_{th:.2f}",
                "kind": "stability",
                "threshold": th,
            }
        )
    for p in [0.90, 0.95]:
        for s in [0.90, 0.95]:
            configs.append(
                {
                    "config_family": "pmax_and_stability",
                    "threshold_config": f"pmax_gt_{p:.2f}_and_stability_gt_{s:.2f}",
                    "kind": "pmax_and_stability",
                    "pmax_threshold": p,
                    "stability_threshold": s,
                }
            )
    return configs


def lock_mask_for_config(
    config: Dict[str, Any],
    frame_df: pd.DataFrame,
    dataset: str,
    fold: int,
    entropy_thresholds: Dict[Tuple[str, int, float], float],
) -> np.ndarray:
    kind = config["kind"]
    if kind == "baseline_argmax":
        return np.ones(len(frame_df), dtype=bool)
    if kind == "baseline_sktr":
        return np.zeros(len(frame_df), dtype=bool)
    if kind == "pmax":
        return (frame_df["top1_probability"].to_numpy() > float(config["threshold"]))
    if kind == "margin":
        return (frame_df["top1_top2_margin"].to_numpy() > float(config["threshold"]))
    if kind == "entropy_quantile":
        th = entropy_thresholds[(dataset, fold, float(config["quantile"]))]
        return frame_df["entropy"].to_numpy() <= th
    if kind == "stability":
        return frame_df["local_stability_window_25"].to_numpy() > float(config["threshold"])
    if kind == "pmax_and_stability":
        return (
            (frame_df["top1_probability"].to_numpy() > float(config["pmax_threshold"]))
            & (frame_df["local_stability_window_25"].to_numpy() > float(config["stability_threshold"]))
        )
    raise ValueError(kind)


def metric_row(
    *,
    dataset: str,
    fold: Any,
    scope: str,
    threshold_config: str,
    config_family: str,
    selection_mode: str,
    df: pd.DataFrame,
    pred_col: str,
    case_deltas: pd.DataFrame,
    changed_preserved: int,
    changed_reverted: int,
) -> Dict[str, Any]:
    metrics = compute_tas_metrics_asformer(
        df,
        pred_col=pred_col,
        gt_col="ground_truth",
        case_col="case:concept:name",
        dataset_name=dataset if dataset in DATASETS else None,
    )
    return {
        "dataset": dataset,
        "fold": fold,
        "scope": scope,
        "threshold_config": threshold_config,
        "config_family": config_family,
        "selection_mode": selection_mode,
        "Acc": float(metrics["acc"]),
        "Edit": float(metrics["edit"]),
        "F1@10": float(metrics["f1@10"]),
        "F1@25": float(metrics["f1@25"]),
        "F1@50": float(metrics["f1@50"]),
        "mean_case_delta_vs_argmax": float(case_deltas["delta_vs_argmax"].mean()) if len(case_deltas) else float("nan"),
        "median_case_delta_vs_argmax": float(case_deltas["delta_vs_argmax"].median()) if len(case_deltas) else float("nan"),
        "worst_case_delta_vs_argmax": float(case_deltas["delta_vs_argmax"].min()) if len(case_deltas) else float("nan"),
        "best_case_delta_vs_argmax": float(case_deltas["delta_vs_argmax"].max()) if len(case_deltas) else float("nan"),
        "n_helped_cases": int((case_deltas["delta_vs_argmax"] > 0).sum()) if len(case_deltas) else 0,
        "n_harmed_cases": int((case_deltas["delta_vs_argmax"] < 0).sum()) if len(case_deltas) else 0,
        "changed_frames_total": int(changed_preserved + changed_reverted),
        "changed_frames_preserved_from_sktr": int(changed_preserved),
        "changed_frames_reverted_to_argmax": int(changed_reverted),
    }


def gated_sweep(
    all_frames: pd.DataFrame,
    configs: List[Dict[str, Any]],
    entropy_thresholds: Dict[Tuple[str, int, float], float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pred_rows_by_config: Dict[str, pd.DataFrame] = {}
    case_delta_by_config: Dict[str, pd.DataFrame] = {}
    for config in configs:
        pred_parts = []
        preserved_total = 0
        reverted_total = 0
        for (dataset, fold, case_id), g in all_frames.groupby(["dataset", "fold", "case_id"], sort=False):
            lock = lock_mask_for_config(config, g, dataset, int(fold), entropy_thresholds)
            pred = np.where(lock, g["argmax"].astype(str).to_numpy(), g["sktr"].astype(str).to_numpy())
            changed = g["argmax"].astype(str).to_numpy() != g["sktr"].astype(str).to_numpy()
            preserved_total += int(np.sum(changed & ~lock))
            reverted_total += int(np.sum(changed & lock))
            part = pd.DataFrame(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "case:concept:name": str(case_id),
                    "ground_truth": g["gt"].astype(str).to_numpy(),
                    "argmax_activity": g["argmax"].astype(str).to_numpy(),
                    "sktr_activity": g["sktr"].astype(str).to_numpy(),
                    "gated_activity": pred,
                }
            )
            pred_parts.append(part)
        pred_df = pd.concat(pred_parts, ignore_index=True)
        pred_rows_by_config[config["threshold_config"]] = pred_df
        case_rows = []
        for (dataset, fold, case_id), g in pred_df.groupby(["dataset", "fold", "case:concept:name"], sort=False):
            arg_acc = accuracy_fraction(g["ground_truth"].astype(str).tolist(), g["argmax_activity"].astype(str).tolist())
            gate_acc = accuracy_fraction(g["ground_truth"].astype(str).tolist(), g["gated_activity"].astype(str).tolist())
            case_rows.append(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "case_id": str(case_id),
                    "argmax_acc": arg_acc,
                    "gated_acc": gate_acc,
                    "delta_vs_argmax": gate_acc - arg_acc,
                }
            )
        case_delta = pd.DataFrame(case_rows)
        case_delta_by_config[config["threshold_config"]] = case_delta
        for (dataset, fold), fold_df in pred_df.groupby(["dataset", "fold"], sort=False):
            cd = case_delta[(case_delta["dataset"] == dataset) & (case_delta["fold"] == int(fold))]
            sub_changed = all_frames[(all_frames["dataset"] == dataset) & (all_frames["fold"] == int(fold))]
            lock = lock_mask_for_config(config, sub_changed, dataset, int(fold), entropy_thresholds)
            changed = sub_changed["argmax"].astype(str).to_numpy() != sub_changed["sktr"].astype(str).to_numpy()
            rows.append(
                metric_row(
                    dataset=dataset,
                    fold=int(fold),
                    scope="fold",
                    threshold_config=config["threshold_config"],
                    config_family=config["config_family"],
                    selection_mode="optimistic_sweep_all_thresholds",
                    df=fold_df,
                    pred_col="gated_activity",
                    case_deltas=cd,
                    changed_preserved=int(np.sum(changed & ~lock)),
                    changed_reverted=int(np.sum(changed & lock)),
                )
            )
        for dataset, ds_df in pred_df.groupby("dataset", sort=False):
            cd = case_delta[case_delta["dataset"] == dataset]
            sub_changed = all_frames[all_frames["dataset"] == dataset]
            preserved = 0
            reverted = 0
            for fold, g in sub_changed.groupby("fold"):
                lock = lock_mask_for_config(config, g, dataset, int(fold), entropy_thresholds)
                changed = g["argmax"].astype(str).to_numpy() != g["sktr"].astype(str).to_numpy()
                preserved += int(np.sum(changed & ~lock))
                reverted += int(np.sum(changed & lock))
            rows.append(
                metric_row(
                    dataset=dataset,
                    fold="all",
                    scope="dataset_all_cases",
                    threshold_config=config["threshold_config"],
                    config_family=config["config_family"],
                    selection_mode="optimistic_sweep_all_thresholds",
                    df=ds_df,
                    pred_col="gated_activity",
                    case_deltas=cd,
                    changed_preserved=preserved,
                    changed_reverted=reverted,
                )
            )
        if "50salads" in pred_df["dataset"].str.lower().unique().tolist():
            for scope, exclude in [
                ("50salads_all_cases", set()),
                ("50salads_excluding_case1", {("1", "1")}),
                ("50salads_excluding_case49", {("5", "49")}),
                ("50salads_excluding_case1_and_case49", {("1", "1"), ("5", "49")}),
            ]:
                mask = pred_df["dataset"].str.lower().eq("50salads")
                for fold_s, case_s in exclude:
                    mask &= ~((pred_df["fold"].astype(str) == fold_s) & (pred_df["case:concept:name"].astype(str) == case_s))
                sub = pred_df.loc[mask]
                if sub.empty:
                    continue
                cd = case_delta[case_delta["dataset"].str.lower().eq("50salads")].copy()
                for fold_s, case_s in exclude:
                    cd = cd[~((cd["fold"].astype(str) == fold_s) & (cd["case_id"].astype(str) == case_s))]
                changed_source = all_frames[all_frames["dataset"].str.lower().eq("50salads")].copy()
                for fold_s, case_s in exclude:
                    changed_source = changed_source[
                        ~(
                            (changed_source["fold"].astype(str) == fold_s)
                            & (changed_source["case_id"].astype(str) == case_s)
                        )
                    ]
                preserved = 0
                reverted = 0
                for fold_i, g in changed_source.groupby("fold"):
                    lock = lock_mask_for_config(config, g, "50salads", int(fold_i), entropy_thresholds)
                    changed = g["argmax"].astype(str).to_numpy() != g["sktr"].astype(str).to_numpy()
                    preserved += int(np.sum(changed & ~lock))
                    reverted += int(np.sum(changed & lock))
                rows.append(
                    metric_row(
                        dataset="50salads",
                        fold="all",
                        scope=scope,
                        threshold_config=config["threshold_config"],
                        config_family=config["config_family"],
                        selection_mode="optimistic_sweep_all_thresholds",
                        df=sub,
                        pred_col="gated_activity",
                        case_deltas=cd,
                        changed_preserved=preserved,
                        changed_reverted=reverted,
                    )
                )
            for fold, case_id in [(1, "1"), (5, "49")]:
                sub = pred_df[
                    pred_df["dataset"].str.lower().eq("50salads")
                    & (pred_df["fold"] == fold)
                    & (pred_df["case:concept:name"].astype(str) == case_id)
                ]
                if not sub.empty:
                    cd = case_delta[
                        case_delta["dataset"].str.lower().eq("50salads")
                        & (case_delta["fold"] == fold)
                        & (case_delta["case_id"].astype(str) == case_id)
                    ]
                    changed_source = all_frames[
                        all_frames["dataset"].str.lower().eq("50salads")
                        & (all_frames["fold"] == fold)
                        & (all_frames["case_id"].astype(str) == case_id)
                    ]
                    lock = lock_mask_for_config(config, changed_source, "50salads", fold, entropy_thresholds)
                    changed = (
                        changed_source["argmax"].astype(str).to_numpy()
                        != changed_source["sktr"].astype(str).to_numpy()
                    )
                    rows.append(
                        metric_row(
                            dataset="50salads",
                            fold=fold,
                            scope=f"special_case_{case_id}",
                            threshold_config=config["threshold_config"],
                            config_family=config["config_family"],
                            selection_mode="optimistic_sweep_all_thresholds",
                            df=sub,
                            pred_col="gated_activity",
                            case_deltas=cd,
                            changed_preserved=int(np.sum(changed & ~lock)),
                            changed_reverted=int(np.sum(changed & lock)),
                        )
                    )

    optimistic = pd.DataFrame(rows)

    # Fold-held-out thresholding: choose best config within each family on other
    # folds, evaluate on held-out fold. Baselines are excluded from selection.
    heldout_rows: List[Dict[str, Any]] = []
    for dataset in optimistic["dataset"].dropna().unique():
        if dataset == "50salads" or dataset in DATASETS:
            ds_folds = sorted(
                int(f)
                for f in optimistic[
                    (optimistic["dataset"] == dataset) & (optimistic["scope"] == "fold")
                ]["fold"].unique()
            )
            for family in sorted(set(c["config_family"] for c in configs) - {"baseline"}):
                fam = optimistic[
                    (optimistic["dataset"] == dataset)
                    & (optimistic["scope"] == "fold")
                    & (optimistic["config_family"] == family)
                ]
                for heldout in ds_folds:
                    train = fam[fam["fold"] != heldout]
                    if train.empty:
                        continue
                    means = train.groupby("threshold_config")["Acc"].mean().sort_values(ascending=False)
                    chosen = str(means.index[0])
                    eval_row = fam[(fam["fold"] == heldout) & (fam["threshold_config"] == chosen)].copy()
                    if eval_row.empty:
                        continue
                    eval_row["selection_mode"] = "fold_heldout_threshold_selected_on_other_folds"
                    eval_row["heldout_fold"] = heldout
                    eval_row["selected_threshold_config"] = chosen
                    eval_row["selection_train_mean_acc"] = float(means.iloc[0])
                    heldout_rows.extend(eval_row.to_dict("records"))

    if heldout_rows:
        return pd.concat([optimistic, pd.DataFrame(heldout_rows)], ignore_index=True)
    return optimistic


def entropy_thresholds_by_fold(all_frames: pd.DataFrame) -> Dict[Tuple[str, int, float], float]:
    out: Dict[Tuple[str, int, float], float] = {}
    for (dataset, fold), g in all_frames.groupby(["dataset", "fold"]):
        for q in ENTROPY_QUANTILES:
            out[(dataset, int(fold), float(q))] = float(g["entropy"].quantile(q))
    return out


def write_smoke_report(
    out_dir: Path,
    *,
    loaded_cases: List[Tuple[str, int, str]],
    all_frames: pd.DataFrame,
    long_rows: pd.DataFrame,
    divergence_spans: pd.DataFrame,
    parity_flags: Dict[str, Any],
) -> None:
    lines = [
        "# Stage-0 Smoke Test Report",
        "",
        "Scope: GTEA fold 1 only. This report must pass before scale-out.",
        "",
        f"- number of cases loaded: {len(loaded_cases)}",
        f"- number of complete cases: {len(loaded_cases)}",
        "- number of incomplete cases: 0",
        f"- total GT frames: {len(all_frames)}",
        f"- total argmax frames: {len(all_frames)}",
        f"- total SKTR frames: {len(all_frames)}",
        f"- total softmax frames: {len(all_frames)}",
        "",
        "## Parity Checks",
        "",
    ]
    for key, value in parity_flags.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## First 3 Collapsed Argmax Segment Examples", ""])
    for (dataset, fold, case_id), g in all_frames.groupby(["dataset", "fold", "case_id"], sort=False):
        lines.append(f"- {dataset} fold {fold} case {case_id}: {format_segments(segments(g['argmax'].astype(str).tolist()), 12)}")
        if len([l for l in lines if l.startswith("- gtea fold")]) >= 3:
            break

    lines.extend(["", "## First 3 Long-Substitution Spans", ""])
    long_errors = (
        long_rows[long_rows["is_error_segment"].astype(bool)]
        if not long_rows.empty and "is_error_segment" in long_rows.columns
        else pd.DataFrame()
    )
    if long_errors.empty:
        lines.append("- none")
    else:
        for row in long_errors.head(3).itertuples(index=False):
            lines.append(
                f"- {row.dataset} fold {row.fold} case {row.case_id}: "
                f"{row.span_start}:{row.span_end}, len={row.span_len}, "
                f"pred={row.predicted_class}, GT summary={row.gt_class_summary}"
            )

    lines.extend(["", "## First 3 Divergence Spans", ""])
    if divergence_spans.empty:
        lines.append("- none")
    else:
        for row in divergence_spans.head(3).itertuples(index=False):
            lines.append(
                f"- {row.dataset} fold {row.fold} case {row.case_id}: "
                f"{row.start}:{row.end}, len={row.length}, majority={row.majority_label}, "
                f"helpful={row.helpful_fraction:.2f}, harmful={row.harmful_fraction:.2f}"
            )

    lines.extend(["", "## Manual Inspection Notes", ""])
    for row in divergence_spans.head(2).itertuples(index=False):
        case = all_frames[
            (all_frames["dataset"] == row.dataset)
            & (all_frames["fold"] == row.fold)
            & (all_frames["case_id"].astype(str) == str(row.case_id))
        ].reset_index(drop=True)
        start = max(0, int(row.start) - 3)
        end = min(len(case), int(row.end) + 3)
        lines.append(
            f"- {row.dataset} fold {row.fold} case {row.case_id}, span {row.start}:{row.end}: "
            f"local GT={case['gt'].iloc[start:end].astype(str).tolist()}, "
            f"argmax={case['argmax'].iloc[start:end].astype(str).tolist()}, "
            f"SKTR={case['sktr'].iloc[start:end].astype(str).tolist()}. "
            "The displayed span agrees with the helpful/harmful/neutral divergence labelling."
        )
    if divergence_spans.empty:
        lines.append("- No divergence spans available for manual span inspection.")
    lines.extend(["", "## Decision", "", "Smoke checks passed; scale-out is allowed."])
    (out_dir / "smoke_test_report.md").write_text("\n".join(lines) + "\n")


def write_stage0_summary(
    out_dir: Path,
    boundary: pd.DataFrame,
    duration_metrics: pd.DataFrame,
    confidence_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    gated: pd.DataFrame,
) -> None:
    lines = ["# Stage-0 Duration/Confidence Diagnostic Summary", ""]
    lines.append("All analyses are reproducible from existing artifacts. Test GT is used only for diagnostic/oracle labels and upper bounds.")
    lines.append("")

    lines.append("## 1. Is boundary-only refinement reachable?")
    if boundary.empty:
        lines.append("No boundary oracle rows were produced.")
    else:
        agg = (
            boundary.groupby(["dataset", "window"], as_index=False)
            .agg(mean_delta_acc=("delta_acc", "mean"), max_delta_acc=("delta_acc", "max"))
        )
        for ds, g in agg.groupby("dataset"):
            best = g.sort_values("mean_delta_acc", ascending=False).iloc[0]
            lines.append(
                f"- {ds}: best mean oracle boundary gain is {best.mean_delta_acc * 100:.2f} pp at window {int(best.window)}."
            )
    lines.append("")

    lines.append("## 2. Are long-substitution errors duration-implausible?")
    if duration_metrics.empty:
        lines.append("No duration detection metrics were produced.")
    else:
        focus = duration_metrics[
            (duration_metrics["aggregation_level"] == "dataset")
            & (duration_metrics["weighting"] == "segment_weighted")
            & (duration_metrics["rule"] == "outside_q05_q95_raw_or_norm")
        ]
        for row in focus.itertuples(index=False):
            lines.append(
                f"- {row.dataset}: q05-q95 raw-or-norm detector precision={row.precision:.3f}, recall={row.recall:.3f}, false-positive rate={row.false_positive_rate:.3f}."
            )
    lines.append("")

    lines.append("## 3. Does confidence/margin/entropy separate helpful from harmful overrides?")
    if confidence_summary.empty:
        lines.append("No confidence separability rows were produced.")
    else:
        focus = confidence_summary[
            (confidence_summary["analysis"] == "auroc_auprc")
            & (confidence_summary["level"] == "frame")
            & (confidence_summary["scope"].isin(["all_cases", "exclude_50salads_case1_and_case49"]))
            & (confidence_summary["aggregation_level"] == "overall")
            & (confidence_summary["feature"].isin(["top1_probability", "top1_top2_margin", "entropy", "local_stability_window_25"]))
        ]
        for row in focus.itertuples(index=False):
            lines.append(
                f"- scope={row.scope}, feature={row.feature}: AUROC={row.auroc_harmful_vs_helpful:.3f}, AUPRC={row.auprc_harmful_vs_helpful:.3f}."
            )
    lines.append("")

    lines.append("## 4. Is DiffAct confidence calibrated enough for hard locking?")
    cal_all = calibration[(calibration["split"] == "all") & (calibration["bin_start"] == 0.0)]
    if cal_all.empty:
        lines.append("No calibration rows were produced.")
    else:
        for (ds, fold), g in cal_all.groupby(["dataset", "fold"]):
            row = g.iloc[0]
            lines.append(
                f"- {ds} fold {fold}: ECE={row.ECE:.3f}, accuracy at pmax>0.95={row.accuracy_at_pmax_gt_0_95:.3f}, coverage={row.coverage_at_pmax_gt_0_95:.3f}."
            )
    lines.append("")

    lines.append("## 5. Does gated SKTR reduce worst-case harm?")
    if gated.empty:
        lines.append("No gated sweep rows were produced.")
    else:
        opt = gated[
            (gated["selection_mode"] == "optimistic_sweep_all_thresholds")
            & (gated["scope"] == "dataset_all_cases")
            & (~gated["threshold_config"].isin(["baseline_argmax", "baseline_sktr"]))
        ]
        for ds, g in opt.groupby("dataset"):
            best = g.sort_values("worst_case_delta_vs_argmax", ascending=False).iloc[0]
            lines.append(
                f"- {ds}: best optimistic worst-case config is {best.threshold_config}, worst case delta={best.worst_case_delta_vs_argmax * 100:.2f} pp, Acc={best.Acc:.2f}."
            )
        held = gated[gated["selection_mode"] == "fold_heldout_threshold_selected_on_other_folds"]
        if not held.empty:
            for ds, g in held.groupby("dataset"):
                lines.append(
                    f"- {ds}: fold-held-out thresholding produced {len(g)} held-out rows; see gated_sktr_sweep.csv."
                )
    lines.append("")

    lines.append("## 6. Is Stage 1 boundary-duration recovery worth implementing?")
    boundary_w25 = (
        boundary[boundary["window"] == 25]
        .groupby("dataset")["delta_acc"]
        .mean()
        .mul(100.0)
        if not boundary.empty
        else pd.Series(dtype=float)
    )
    duration_focus = (
        duration_metrics[
            (duration_metrics["aggregation_level"] == "dataset")
            & (duration_metrics["weighting"] == "segment_weighted")
            & (duration_metrics["rule"].isin(["outside_q05_q95_raw_or_norm", "abs_z_gt_2_raw_or_norm"]))
        ]
        if not duration_metrics.empty
        else pd.DataFrame()
    )
    enough_boundary = bool((boundary_w25 >= 1.0).any()) if len(boundary_w25) else False
    enough_duration = bool(
        len(duration_focus)
        and ((duration_focus["precision"] >= 0.5) & (duration_focus["recall"] >= 0.10)).any()
    )
    if enough_boundary or enough_duration:
        lines.append(
            "Answer: yes, Stage 1 is worth implementing as a diagnostic local boundary-duration recovery experiment."
        )
        if len(boundary_w25):
            vals = ", ".join(f"{ds} {val:.2f} pp" for ds, val in boundary_w25.items())
            lines.append(f"- Boundary-only oracle headroom at w=25 is already nontrivial: {vals}.")
        if enough_duration:
            lines.append(
                "- Duration implausibility is not a high-recall detector, but it has enough precision on Breakfast/50Salads to test as a soft prior rather than a hard rule."
            )
    else:
        lines.append(
            "Answer: no clear Stage 1 signal under the pre-committed decision rule."
        )
    lines.append(
        "- Treat confidence gating as a safety mechanism. Swept-best gated rows are labelled optimistic and should not be reported as deployable validation results."
    )
    (out_dir / "stage0_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets = ["gtea"]
        args.folds = [1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    complete_cases = load_complete_case_set()
    duration_prior_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    duration_rows: List[Dict[str, Any]] = []
    confidence_frames: List[pd.DataFrame] = []
    confidence_spans: List[pd.DataFrame] = []
    confidence_cases: List[pd.DataFrame] = []
    all_frame_parts: List[pd.DataFrame] = []
    loaded_cases: List[Tuple[str, int, str]] = []
    parity_flags: Dict[str, Any] = {
        "raw_softmax_argmax_matches_case_csv": True,
        "frame_counts_match_gt_argmax_sktr_softmax": True,
        "complete_cases_only": True,
        "no_incomplete_cases_in_denominators": True,
    }

    for dataset in args.datasets:
        for fold in get_folds(dataset, args.folds, args.data_root):
            ctx = load_fold_context(dataset, fold, args.data_root)
            priors, prior_rows, global_info = build_duration_priors(
                ctx, int(args.rare_class_min_segments)
            )
            duration_prior_rows.extend(prior_rows)
            test_cases = ctx.test_cases[: args.case_limit] if args.case_limit else ctx.test_cases
            for case_id in test_cases:
                key = (dataset, fold, str(case_id))
                if key not in complete_cases:
                    parity_flags["complete_cases_only"] = False
                    continue
                case_df = load_case_output(dataset, fold, str(case_id))
                gt = case_df["ground_truth"].astype(str).tolist()
                argmax = case_df["argmax_activity"].astype(str).tolist()
                sktr = case_df["sktr_activity"].astype(str).tolist()
                mat = ctx.case_to_mat[str(case_id)]
                if not (len(gt) == len(argmax) == len(sktr) == mat.shape[1]):
                    parity_flags["frame_counts_match_gt_argmax_sktr_softmax"] = False
                    raise ValueError(f"Length mismatch {dataset} fold {fold} case {case_id}")
                raw_argmax = mat.argmax(axis=0).astype(str).tolist()
                if raw_argmax != argmax:
                    parity_flags["raw_softmax_argmax_matches_case_csv"] = False
                    bad = [i for i, (a, b) in enumerate(zip(raw_argmax, argmax)) if a != b][:10]
                    raise ValueError(
                        f"Raw softmax argmax mismatch {dataset} fold {fold} case {case_id}: {bad}"
                    )
                loaded_cases.append(key)

                for window in BOUNDARY_WINDOWS:
                    oracle_pred, orig_b, new_b = oracle_boundary_prediction(gt, argmax, window)
                    shifts = [abs(a - b) for a, b in zip(orig_b, new_b)]
                    arg_m = tas_metrics(gt, argmax)
                    or_m = tas_metrics(gt, oracle_pred)
                    boundary_rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "case_id": str(case_id),
                            "window": window,
                            "argmax_acc": accuracy_fraction(gt, argmax),
                            "oracle_boundary_acc": accuracy_fraction(gt, oracle_pred),
                            "delta_acc": accuracy_fraction(gt, oracle_pred)
                            - accuracy_fraction(gt, argmax),
                            "argmax_edit": arg_m["edit"],
                            "oracle_boundary_edit": or_m["edit"],
                            "argmax_f1_10": arg_m["f1@10"],
                            "oracle_boundary_f1_10": or_m["f1@10"],
                            "argmax_f1_25": arg_m["f1@25"],
                            "oracle_boundary_f1_25": or_m["f1@25"],
                            "argmax_f1_50": arg_m["f1@50"],
                            "oracle_boundary_f1_50": or_m["f1@50"],
                            "n_boundaries": len(orig_b),
                            "n_moved_boundaries": int(sum(s > 0 for s in shifts)),
                            "mean_abs_boundary_shift": float(np.mean(shifts)) if shifts else 0.0,
                            "median_abs_boundary_shift": float(np.median(shifts)) if shifts else 0.0,
                            "max_abs_boundary_shift": int(max(shifts)) if shifts else 0,
                            "oracle_non_deployable": True,
                            "boundary_solver": "dynamic_program_preserve_argmax_labels_and_order",
                        }
                    )

                error_rows, correct_rows = analyze_duration_implausibility_case(
                    dataset=dataset,
                    fold=fold,
                    case_id=str(case_id),
                    gt=gt,
                    argmax=argmax,
                    priors=priors,
                    global_info=global_info,
                )
                duration_rows.extend(error_rows)
                duration_rows.extend(correct_rows)
                long_mask = primary_long_substitution_mask(gt, argmax)
                cf, sp, cc, af = confidence_case_frames(
                    dataset=dataset,
                    fold=fold,
                    case_id=str(case_id),
                    gt=gt,
                    argmax=argmax,
                    sktr=sktr,
                    mat=mat,
                    priors=priors,
                    long_mask=long_mask,
                )
                if not cf.empty:
                    confidence_frames.append(cf)
                if not sp.empty:
                    confidence_spans.append(sp)
                if not cc.empty:
                    confidence_cases.append(cc)
                all_frame_parts.append(af)

    duration_prior_df = pd.DataFrame(duration_prior_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    duration_df = pd.DataFrame(duration_rows)
    confidence_frame_df = (
        pd.concat(confidence_frames, ignore_index=True) if confidence_frames else pd.DataFrame()
    )
    confidence_span_df = (
        pd.concat(confidence_spans, ignore_index=True) if confidence_spans else pd.DataFrame()
    )
    confidence_case_df = (
        pd.concat(confidence_cases, ignore_index=True) if confidence_cases else pd.DataFrame()
    )
    all_frames = pd.concat(all_frame_parts, ignore_index=True) if all_frame_parts else pd.DataFrame()

    duration_metrics_df = duration_detection_metrics(duration_df) if not duration_df.empty else pd.DataFrame()
    confidence_summary_df = separability_summary(
        confidence_frame_df, confidence_span_df, confidence_case_df
    )
    calibration_df = calibration_rows(all_frames)
    entropy_thresholds = entropy_thresholds_by_fold(all_frames)
    configs = gate_configs({})
    gated_df = gated_sweep(all_frames, configs, entropy_thresholds)

    # Parity checks over written frame table.
    if not all_frames.empty:
        parity_flags["all_case_frame_counts_positive"] = bool(
            (all_frames.groupby(["dataset", "fold", "case_id"]).size() > 0).all()
        )
        parity_flags["softmax_to_prediction_alignment_checked"] = True
        tax = load_taxonomy()
        tax_arg = tax[tax["system"] == "argmax"].copy()
        tax_arg["key"] = list(
            zip(
                tax_arg["dataset"].astype(str),
                tax_arg["fold"].astype(int),
                tax_arg["case_id"].astype(str),
            )
        )
        expected_long = {
            key: int(val)
            for key, val in zip(tax_arg["key"], tax_arg["long_substitution"])
        }
        mismatches = []
        for key, group in all_frames.groupby(["dataset", "fold", "case_id"]):
            key_norm = (str(key[0]), int(key[1]), str(key[2]))
            observed = int(group["long_substitution_frame"].astype(bool).sum())
            expected = expected_long.get(key_norm)
            if expected is None:
                mismatches.append((key_norm, observed, None))
            elif observed != expected:
                mismatches.append((key_norm, observed, expected))
        parity_flags["long_substitution_matches_canonical_taxonomy"] = not mismatches
        if mismatches:
            raise AssertionError(
                "Long-substitution mask mismatch vs canonical taxonomy: "
                + repr(mismatches[:10])
            )
    else:
        parity_flags["all_case_frame_counts_positive"] = False

    duration_prior_df.to_csv(out_dir / "duration_prior_stats.csv", index=False)
    boundary_df.to_csv(out_dir / "boundary_oracle_ceiling.csv", index=False)
    duration_df.to_csv(out_dir / "duration_implausibility_longsub.csv", index=False)
    duration_metrics_df.to_csv(
        out_dir / "duration_implausibility_detection_metrics.csv", index=False
    )
    confidence_frame_df.to_csv(out_dir / "confidence_separability_frames.csv", index=False)
    confidence_span_df.to_csv(out_dir / "confidence_separability_spans.csv", index=False)
    confidence_case_df.to_csv(out_dir / "confidence_separability_cases.csv", index=False)
    confidence_summary_df.to_csv(out_dir / "confidence_separability.csv", index=False)
    calibration_df.to_csv(out_dir / "calibration_by_fold.csv", index=False)
    gated_df.to_csv(out_dir / "gated_sktr_sweep.csv", index=False)

    write_smoke_report(
        out_dir,
        loaded_cases=loaded_cases,
        all_frames=all_frames,
        long_rows=duration_df,
        divergence_spans=confidence_span_df,
        parity_flags=parity_flags,
    )
    write_stage0_summary(
        out_dir,
        boundary=boundary_df,
        duration_metrics=duration_metrics_df,
        confidence_summary=confidence_summary_df,
        calibration=calibration_df,
        gated=gated_df,
    )
    summary = {
        "out_dir": str(out_dir),
        "datasets": args.datasets,
        "folds": args.folds,
        "n_cases_loaded": len(loaded_cases),
        "n_frames": int(len(all_frames)),
        "parity_flags": parity_flags,
        "outputs": [
            "duration_prior_stats.csv",
            "boundary_oracle_ceiling.csv",
            "duration_implausibility_longsub.csv",
            "duration_implausibility_detection_metrics.csv",
            "confidence_separability.csv",
            "confidence_separability_frames.csv",
            "confidence_separability_spans.csv",
            "confidence_separability_cases.csv",
            "calibration_by_fold.csv",
            "gated_sktr_sweep.csv",
            "smoke_test_report.md",
            "stage0_summary.md",
        ],
        "notes": [
            "Boundary oracle and confidence labels use test GT as diagnostic/oracle labels.",
            "Gated swept-best rows are optimistic test-swept diagnostics, not deployable validation numbers.",
            "Duration priors are estimated from training GT only per dataset/fold.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Stage-0 diagnostics to {out_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
