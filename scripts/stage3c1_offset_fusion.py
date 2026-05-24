#!/usr/bin/env python3
"""
Stage-3C1: conservative 50Salads offset-span fusion pre-check.

This stage uses only the Stage-3C0 cache. The baseline is official fold-local
DiffAct pred_full. GT is used only for training-fold selection, held-out
evaluation, and oracle diagnostics; deployable candidates use offset
disagreement/probabilities only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DIFFACT_ROOT = REPO_ROOT / "baselines" / "DiffAct"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(DIFFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFACT_ROOT))

import stage3c0_postprocess_reconciliation as s3c0  # noqa: E402
from stage3b_diffusion_ensemble_feasibility import DEFAULT_DATA_ROOT, PUBLISHED  # noqa: E402
from stage3b_gate05_diffusion_ensemble_probe import event_list_for, load_gt  # noqa: E402


DEFAULT_STAGE3C0 = Path("/data1/eli-bogdanov/sktr_runs/stage3c0_postprocess_reconciliation_v1")
DEFAULT_OUT = Path("/data1/eli-bogdanov/sktr_runs/stage3c1_offset_fusion_v1")
WINDOWS = [0, 5, 10, 15, 30, 45, 60]
SPAN_LENGTHS = [25, 50, 100, 200]
ENTROPY_THRESHOLD = 0.35
RNG_SEED = 42
METRIC_KEYS = ["acc", "edit", "f1@10", "f1@25", "f1@50"]


@dataclass
class CaseData:
    dataset: str
    fold: int
    case_id: str
    video: str
    fold_local_idx: int
    full_len: int
    n_classes: int
    sample_rate: int
    n_offsets: int
    official_path: Path
    mean_pre_path: Path
    post_median_path: Path
    gt: np.ndarray
    official: np.ndarray
    mean_pre: Optional[np.ndarray] = None
    offset_probs: Optional[List[np.ndarray]] = None
    offset_preds: Optional[List[np.ndarray]] = None
    offset_confs: Optional[List[np.ndarray]] = None
    fixed_cache: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)
    offset_summary_cache: Optional[Dict[str, np.ndarray]] = None


def case_key(case: CaseData) -> Tuple[str, int, str]:
    return (case.dataset, case.fold, case.case_id)


def df_to_markdown(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    return s3c0.df_to_markdown(df, max_rows=max_rows)


def load_cases(stage3c0_dir: Path, data_root: Path, smoke_fold: Optional[int]) -> List[CaseData]:
    manifest = pd.read_csv(stage3c0_dir / "cache_manifest.csv")
    fs = manifest[(manifest["dataset"] == "50salads") & (manifest["complete_flag"].astype(bool))].copy()
    if smoke_fold is not None:
        fs = fs[fs["fold"].astype(int) == int(smoke_fold)].copy()
    labels = event_list_for(data_root, "50salads")
    cases: List[CaseData] = []
    for _, row in fs.sort_values(["fold", "case_id"]).iterrows():
        gt = load_gt(data_root, "50salads", str(row["video"]), labels).astype(np.int32, copy=False)
        official = np.load(str(row["official_prediction_path"])).astype(np.int32, copy=False)
        if len(gt) != len(official):
            raise ValueError(f"{row['case_id']}: GT/pred length mismatch {len(gt)} != {len(official)}")
        cases.append(
            CaseData(
                dataset="50salads",
                fold=int(row["fold"]),
                case_id=str(row["case_id"]),
                video=str(row["video"]),
                fold_local_idx=int(row["video_idx_fold_local"]),
                full_len=int(row["n_frames_full"]),
                n_classes=int(row["n_classes"]),
                sample_rate=int(row["sample_rate"]),
                n_offsets=int(row["n_offsets"]),
                official_path=Path(str(row["official_prediction_path"])),
                mean_pre_path=Path(str(row["pre_median_mean_softmax_path"])),
                post_median_path=Path(str(row["post_median_softmax_path"])),
                gt=gt,
                official=official,
            )
        )
    return cases


def check_physical_video_leakage(cases: Sequence[CaseData]) -> Dict[str, Any]:
    video_to_folds: Dict[str, set] = {}
    for case in cases:
        video_to_folds.setdefault(case.video, set()).add(case.fold)
    leaked = {v: sorted(folds) for v, folds in video_to_folds.items() if len(folds) > 1}
    return {
        "n_cases": len(cases),
        "n_unique_videos": len(video_to_folds),
        "physical_video_leakage_detected": bool(leaked),
        "leaked_videos": leaked,
    }


def offset_path(stage3c0_dir: Path, case: CaseData, offset: int) -> Path:
    return (
        stage3c0_dir
        / "pre_median_offset_softmax"
        / "50salads"
        / f"fold{case.fold}"
        / f"{case.case_id}_offset{offset}.npy"
    )


def load_mean_pre(case: CaseData) -> np.ndarray:
    if case.mean_pre is None:
        case.mean_pre = np.load(case.mean_pre_path).astype(np.float32, copy=False)
    return case.mean_pre


def restore_offset_pred_and_conf(probs: np.ndarray, case: CaseData, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    right = case.full_len - (offset + case.sample_rate * (probs.shape[1] - 1)) - 1
    if right < 0:
        raise ValueError(f"{case.case_id} offset {offset}: invalid restore geometry")
    pred = s3c0.restore_pred_from_native(
        probs,
        case.full_len,
        case.sample_rate,
        left_offset=offset,
        right_offset=right,
    )
    native_conf = probs.max(axis=0)
    conf = s3c0.restore_full_sequence(
        native_conf,
        case.full_len,
        offset,
        right,
        case.sample_rate,
    ).astype(np.float32, copy=False)
    return pred.astype(np.int32, copy=False), conf


def ensure_offsets(stage3c0_dir: Path, case: CaseData) -> None:
    if case.offset_probs is not None:
        return
    probs: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    for offset in range(case.n_offsets):
        arr = np.load(offset_path(stage3c0_dir, case, offset)).astype(np.float32, copy=False)
        pred, conf = restore_offset_pred_and_conf(arr, case, offset)
        probs.append(arr)
        preds.append(pred)
        confs.append(conf)
    case.offset_probs = probs
    case.offset_preds = preds
    case.offset_confs = confs


def fixed_prediction(stage3c0_dir: Path, case: CaseData, variant: str, window: int = 0) -> np.ndarray:
    key = (variant, int(window))
    if key in case.fixed_cache:
        return case.fixed_cache[key]
    if variant == "official":
        pred = case.official
    elif variant in ("no_median", "mean_then_median"):
        native = load_mean_pre(case)
        if variant == "mean_then_median":
            native = s3c0.median_and_renorm(native, window)
        pred = s3c0.restore_pred_from_native(native, case.full_len, case.sample_rate)
    elif variant == "median_each_offset_then_mean":
        ensure_offsets(stage3c0_dir, case)
        assert case.offset_probs is not None
        min_len = load_mean_pre(case).shape[1]
        filtered = [s3c0.median_and_renorm(p[:, :min_len], window) for p in case.offset_probs]
        native = np.stack(filtered, axis=0).mean(axis=0)
        native = native / np.maximum(native.sum(axis=0, keepdims=True), 1e-12)
        pred = s3c0.restore_pred_from_native(native, case.full_len, case.sample_rate)
    else:
        raise ValueError(variant)
    case.fixed_cache[key] = pred.astype(np.int32, copy=False)
    return case.fixed_cache[key]


def offset_summary(stage3c0_dir: Path, case: CaseData) -> Dict[str, np.ndarray]:
    if case.offset_summary_cache is not None:
        return case.offset_summary_cache
    ensure_offsets(stage3c0_dir, case)
    assert case.offset_preds is not None and case.offset_confs is not None
    preds = np.stack(case.offset_preds, axis=0)
    confs = np.stack(case.offset_confs, axis=0)
    n_offsets, n_frames = preds.shape
    majority = np.empty(n_frames, dtype=np.int32)
    majority_share = np.empty(n_frames, dtype=np.float32)
    entropy = np.empty(n_frames, dtype=np.float32)
    for t in range(n_frames):
        counts = np.bincount(preds[:, t].astype(np.int32), minlength=case.n_classes)
        majority[t] = int(np.argmax(counts))
        probs = counts[counts > 0].astype(np.float32) / float(n_offsets)
        entropy[t] = float(-(probs * np.log(probs)).sum() / np.log(n_offsets))
        majority_share[t] = float(counts[majority[t]] / n_offsets)
    best_idx = np.argmax(confs, axis=0)
    best_conf_pred = preds[best_idx, np.arange(n_frames)].astype(np.int32, copy=False)
    best_conf = confs[best_idx, np.arange(n_frames)].astype(np.float32, copy=False)
    high_mask = (entropy >= ENTROPY_THRESHOLD) | (majority != case.official)
    case.offset_summary_cache = {
        "preds": preds,
        "confs": confs,
        "majority": majority,
        "majority_share": majority_share,
        "entropy": entropy,
        "best_conf_pred": best_conf_pred,
        "best_conf": best_conf,
        "high_mask": high_mask,
    }
    return case.offset_summary_cache


def mask_to_spans(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            if idx - start >= min_len:
                spans.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        spans.append((start, len(mask)))
    return spans


def prediction_with_spans(case: CaseData, spans: Sequence[Tuple[int, int]], replacement: np.ndarray) -> np.ndarray:
    pred = case.official.copy()
    for start, end in spans:
        pred[start:end] = replacement[start:end]
    return pred.astype(np.int32, copy=False)


def case_metrics(data_root: Path, case: CaseData, pred: np.ndarray) -> Dict[str, float]:
    return s3c0.metrics(data_root, "50salads", case.gt, pred)


def aggregate_cases(data_root: Path, cases: Sequence[CaseData], pred_map: Dict[str, np.ndarray]) -> Dict[str, float]:
    specs = [
        s3c0.CaseSpec("50salads", c.fold, c.case_id, c.video, c.fold_local_idx)
        for c in cases
    ]
    gt_map = {("50salads", c.fold, c.case_id): c.gt for c in cases}
    pm = {("50salads", c.fold, c.case_id): pred_map[c.case_id] for c in cases}
    return s3c0.dataset_metrics(data_root, specs, gt_map, pm)


def official_map(cases: Sequence[CaseData]) -> Dict[str, np.ndarray]:
    return {c.case_id: c.official for c in cases}


def metric_delta(row: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {f"delta_{k}": float(row[k] - base[k]) for k in METRIC_KEYS}


def fixed_configs() -> List[Tuple[str, int]]:
    configs = [("official", -1), ("no_median", 0)]
    configs.extend(("mean_then_median", w) for w in WINDOWS)
    configs.extend(("median_each_offset_then_mean", w) for w in WINDOWS)
    # Keep the exact config list explicit; duplicates are useful audit rows.
    return configs


def part1_fixed_sweep(stage3c0_dir: Path, data_root: Path, cases: Sequence[CaseData]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base_all = aggregate_cases(data_root, cases, official_map(cases))
    configs = fixed_configs()
    folds = sorted({c.fold for c in cases})
    for variant, window in configs:
        pred_all = {c.case_id: fixed_prediction(stage3c0_dir, c, variant, window) for c in cases}
        m_all = aggregate_cases(data_root, cases, pred_all)
        row = {
            "selection_mode": "candidate_all",
            "scope": "all",
            "heldout_fold": "",
            "variant": variant,
            "window": window,
            **m_all,
        }
        row.update(metric_delta(m_all, base_all))
        rows.append(row)
        for fold in folds:
            fold_cases = [c for c in cases if c.fold == fold]
            base = aggregate_cases(data_root, fold_cases, official_map(fold_cases))
            pred = {c.case_id: pred_all[c.case_id] for c in fold_cases}
            m = aggregate_cases(data_root, fold_cases, pred)
            r = {
                "selection_mode": "candidate_by_fold",
                "scope": f"fold{fold}",
                "heldout_fold": fold,
                "variant": variant,
                "window": window,
                **m,
            }
            r.update(metric_delta(m, base))
            rows.append(r)
    for heldout in folds:
        train = [c for c in cases if c.fold != heldout]
        held = [c for c in cases if c.fold == heldout]
        train_base = aggregate_cases(data_root, train, official_map(train))
        scored = []
        for variant, window in configs:
            pred = {c.case_id: fixed_prediction(stage3c0_dir, c, variant, window) for c in train}
            m = aggregate_cases(data_root, train, pred)
            scored.append((m["f1@50"], m["edit"], m["acc"], variant, window))
        _, _, _, variant, window = max(scored)
        held_base = aggregate_cases(data_root, held, official_map(held))
        pred = {c.case_id: fixed_prediction(stage3c0_dir, c, variant, window) for c in held}
        m = aggregate_cases(data_root, held, pred)
        r = {
            "selection_mode": "fold_heldout",
            "scope": f"fold{heldout}",
            "heldout_fold": heldout,
            "variant": variant,
            "window": window,
            **m,
        }
        r.update(metric_delta(m, held_base))
        rows.append(r)
    return pd.DataFrame(rows)


def oracle_validity(stage3c0_dir: Path, data_root: Path, cases: Sequence[CaseData]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    best_rows = []
    window_deltas_by_case: List[np.ndarray] = []
    for case in cases:
        base = case_metrics(data_root, case, case.official)
        candidates: List[Tuple[str, str, int, np.ndarray]] = []
        for variant, window in fixed_configs():
            candidates.append(("fixed", variant, window, fixed_prediction(stage3c0_dir, case, variant, window)))
        ensure_offsets(stage3c0_dir, case)
        assert case.offset_preds is not None
        for idx, pred in enumerate(case.offset_preds):
            candidates.append(("offset", f"offset{idx}", idx, pred))
        case_rows = []
        for family, variant, param, pred in candidates:
            m = case_metrics(data_root, case, pred)
            row = {
                "fold": case.fold,
                "case_id": case.case_id,
                "video": case.video,
                "candidate_family": family,
                "variant": variant,
                "param": param,
                **m,
            }
            row.update(metric_delta(m, base))
            rows.append(row)
            case_rows.append(row)
        best_f50 = max(case_rows, key=lambda r: (r["f1@50"], r["edit"], r["acc"]))
        best_f10 = max(case_rows, key=lambda r: (r["f1@10"], r["edit"], r["acc"]))
        best_rows.append(
            {
                "fold": case.fold,
                "case_id": case.case_id,
                "best_f1@50_delta_f1@50": best_f50["delta_f1@50"],
                "best_f1@10_delta_f1@50": best_f10["delta_f1@50"],
                "best_f1@50_variant": best_f50["variant"],
                "best_f1@10_variant": best_f10["variant"],
            }
        )
        window_deltas_by_case.append(
            np.asarray(
                [
                    r["delta_f1@50"]
                    for r in case_rows
                    if r["candidate_family"] == "fixed" and r["variant"] != "official"
                ],
                dtype=np.float64,
            )
        )
    best_df = pd.DataFrame(best_rows)
    mean_oracle = float(best_df["best_f1@50_delta_f1@50"].mean())
    mean_retained = float(best_df["best_f1@10_delta_f1@50"].mean())
    retained_fraction = mean_retained / mean_oracle if abs(mean_oracle) > 1e-12 else float("nan")
    rng = np.random.default_rng(RNG_SEED)
    null_vals = []
    for _ in range(2000):
        per_case = []
        for vals in window_deltas_by_case:
            centered = vals - vals.mean()
            signs = rng.choice([-1.0, 1.0], size=len(centered))
            per_case.append(float(np.max(centered * signs)))
        null_vals.append(float(np.mean(per_case)))
    stats = {
        "oracle_best_f1@50_mean_delta_f1@50": mean_oracle,
        "cross_metric_f1@10_selected_mean_delta_f1@50": mean_retained,
        "cross_metric_retained_fraction": retained_fraction,
        "window_noise_null_mean_best_delta_f1@50_mean": float(np.mean(null_vals)),
        "window_noise_null_mean_best_delta_f1@50_p95": float(np.percentile(null_vals, 95)),
    }
    detailed = pd.DataFrame(rows).merge(best_df, on=["fold", "case_id"], how="left")
    return detailed, stats


def precheck_direction_and_eligibility(stage3c0_dir: Path, cases: Sequence[CaseData]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {
        "long_error_frames": 0.0,
        "long_error_in_high_disagreement_frames": 0.0,
        "high_disagreement_frames": 0.0,
    }
    for method in ["offset_majority", "best_confidence_offset"]:
        totals[f"{method}_helpful_changed_frames"] = 0.0
        totals[f"{method}_harmful_changed_frames"] = 0.0
        totals[f"{method}_neutral_changed_frames"] = 0.0
    for case in cases:
        summary = offset_summary(stage3c0_dir, case)
        high_mask = summary["high_mask"].astype(bool)
        # Precheck uses a permissive L=25 because it gates whether any span-fusion
        # work is justified. Part 2 sweeps stricter lengths if reached.
        span_mask = np.zeros(case.full_len, dtype=bool)
        for start, end in mask_to_spans(high_mask, min_len=25):
            span_mask[start:end] = True
        long_mask = s3c0.long_wrong_mask(case.gt, case.official)
        totals["long_error_frames"] += int(long_mask.sum())
        totals["long_error_in_high_disagreement_frames"] += int((long_mask & span_mask).sum())
        totals["high_disagreement_frames"] += int(span_mask.sum())
        for method, pred in [
            ("offset_majority", summary["majority"]),
            ("best_confidence_offset", summary["best_conf_pred"]),
        ]:
            changed = span_mask & (pred != case.official)
            helpful = changed & (pred == case.gt) & (case.official != case.gt)
            harmful = changed & (pred != case.gt) & (case.official == case.gt)
            neutral = changed & ~(helpful | harmful)
            totals[f"{method}_helpful_changed_frames"] += int(helpful.sum())
            totals[f"{method}_harmful_changed_frames"] += int(harmful.sum())
            totals[f"{method}_neutral_changed_frames"] += int(neutral.sum())
            denom = int(helpful.sum() + harmful.sum())
            rows.append(
                {
                    "fold": case.fold,
                    "case_id": case.case_id,
                    "video": case.video,
                    "method": method,
                    "high_disagreement_frames": int(span_mask.sum()),
                    "long_error_frames": int(long_mask.sum()),
                    "long_error_in_high_disagreement_frames": int((long_mask & span_mask).sum()),
                    "changed_frames": int(changed.sum()),
                    "helpful_changed_frames": int(helpful.sum()),
                    "harmful_changed_frames": int(harmful.sum()),
                    "neutral_changed_frames": int(neutral.sum()),
                    "direction_toward_gt_rate": float(helpful.sum() / denom) if denom else float("nan"),
                }
            )
    stats: Dict[str, float] = {
        "eligible_long_error_mass_share": (
            totals["long_error_in_high_disagreement_frames"] / totals["long_error_frames"]
            if totals["long_error_frames"]
            else float("nan")
        ),
        "high_disagreement_frame_share": totals["high_disagreement_frames"] / sum(c.full_len for c in cases),
    }
    for method in ["offset_majority", "best_confidence_offset"]:
        helpful = totals[f"{method}_helpful_changed_frames"]
        harmful = totals[f"{method}_harmful_changed_frames"]
        stats[f"{method}_direction_toward_gt_rate"] = helpful / (helpful + harmful) if helpful + harmful else float("nan")
        stats[f"{method}_helpful_changed_frames"] = helpful
        stats[f"{method}_harmful_changed_frames"] = harmful
    return pd.DataFrame(rows), stats


def build_prechecks(stage3c0_dir: Path, data_root: Path, out_dir: Path, cases: Sequence[CaseData]) -> Tuple[pd.DataFrame, pd.DataFrame, bool, Dict[str, Any]]:
    leak = check_physical_video_leakage(cases)
    boundary = pd.read_csv(stage3c0_dir / "boundary_headroom_pre_vs_post.csv")
    boundary_50s = boundary[boundary["dataset"] == "50salads"].copy()
    oracle_gap, oracle_stats = oracle_validity(stage3c0_dir, data_root, cases)
    direction_by_case, direction_stats = precheck_direction_and_eligibility(stage3c0_dir, cases)
    oracle_gap.to_csv(out_dir / "stage3c1_oracle_gap.csv", index=False)
    direction_by_case.to_csv(out_dir / "stage3c1_direction_by_case.csv", index=False)

    pre = []
    for _, row in boundary_50s.iterrows():
        pre.append(
            {
                "check": "probe4_boundary_headroom",
                "item": row["prediction_type"],
                "value": row["delta_f1@50"],
                "secondary_value": row["delta_edit"],
                "notes": "delta_f1@50 / delta_edit from Stage-3C0 boundary_headroom_pre_vs_post",
            }
        )
    pre.extend(
        [
            {
                "check": "oracle_validity",
                "item": "best_f1@10_retains_f1@50_oracle",
                "value": oracle_stats["cross_metric_retained_fraction"],
                "secondary_value": oracle_stats["cross_metric_f1@10_selected_mean_delta_f1@50"],
                "notes": f"F1@50-selected mean delta={oracle_stats['oracle_best_f1@50_mean_delta_f1@50']:.4f}",
            },
            {
                "check": "oracle_validity",
                "item": "window_noise_null_p95",
                "value": oracle_stats["window_noise_null_mean_best_delta_f1@50_p95"],
                "secondary_value": oracle_stats["window_noise_null_mean_best_delta_f1@50_mean"],
                "notes": "Sign-flipped centered window-candidate noise-null for per-case best F1@50 gain",
            },
            {
                "check": "direction",
                "item": "offset_majority_toward_gt",
                "value": direction_stats["offset_majority_direction_toward_gt_rate"],
                "secondary_value": direction_stats["offset_majority_helpful_changed_frames"],
                "notes": "Helpful/(helpful+harmful) changed frames inside high-disagreement spans",
            },
            {
                "check": "direction",
                "item": "best_confidence_offset_toward_gt",
                "value": direction_stats["best_confidence_offset_direction_toward_gt_rate"],
                "secondary_value": direction_stats["best_confidence_offset_helpful_changed_frames"],
                "notes": "Helpful/(helpful+harmful) changed frames inside high-disagreement spans",
            },
            {
                "check": "eligibility",
                "item": "long_substitution_error_mass_in_high_disagreement_spans",
                "value": direction_stats["eligible_long_error_mass_share"],
                "secondary_value": direction_stats["high_disagreement_frame_share"],
                "notes": f"High disagreement = vote entropy >= {ENTROPY_THRESHOLD} or offset majority differs from official; spans >=25 frames",
            },
            {
                "check": "leakage",
                "item": "physical_video_leakage_detected",
                "value": float(leak["physical_video_leakage_detected"]),
                "secondary_value": leak["n_unique_videos"],
                "notes": json.dumps(leak["leaked_videos"], sort_keys=True),
            },
        ]
    )
    pre_df = pd.DataFrame(pre)
    pre_df.to_csv(out_dir / "stage3c1_prechecks.csv", index=False)

    direction_ok = max(
        direction_stats["offset_majority_direction_toward_gt_rate"],
        direction_stats["best_confidence_offset_direction_toward_gt_rate"],
    ) > 0.55
    oracle_ok = oracle_stats["cross_metric_retained_fraction"] >= 0.5
    eligible_ok = direction_stats["eligible_long_error_mass_share"] >= 0.10
    leakage_ok = not leak["physical_video_leakage_detected"]
    gate_pass = bool(direction_ok and oracle_ok and eligible_ok and leakage_ok)
    gate = {
        "direction_ok": direction_ok,
        "oracle_validity_ok": oracle_ok,
        "eligibility_ok": eligible_ok,
        "leakage_ok": leakage_ok,
        "gate_pass": gate_pass,
        **oracle_stats,
        **direction_stats,
        **leak,
    }
    return pre_df, direction_by_case, gate_pass, gate


def span_candidates(stage3c0_dir: Path, case: CaseData, min_len: int) -> List[Tuple[int, int]]:
    summary = offset_summary(stage3c0_dir, case)
    high = summary["high_mask"].astype(bool)
    return mask_to_spans(high, min_len=min_len)


def fusion_prediction(stage3c0_dir: Path, case: CaseData, method: str, min_len: int) -> Tuple[np.ndarray, int, int]:
    summary = offset_summary(stage3c0_dir, case)
    spans = span_candidates(stage3c0_dir, case, min_len)
    if method == "no_edit":
        return case.official.copy(), 0, 0
    if method == "offset_majority":
        repl = summary["majority"]
    elif method == "best_confidence_offset":
        repl = summary["best_conf_pred"]
    else:
        raise ValueError(method)
    pred = prediction_with_spans(case, spans, repl)
    modified = int((pred != case.official).sum())
    return pred, len(spans), modified


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000) -> Tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(RNG_SEED)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def part2_fusion(stage3c0_dir: Path, data_root: Path, out_dir: Path, cases: Sequence[CaseData], gate_pass: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    by_case_rows: List[Dict[str, Any]] = []
    heldout_rows: List[Dict[str, Any]] = []
    special_rows: List[Dict[str, Any]] = []
    if not gate_pass:
        skipped = pd.DataFrame(
            [
                {
                    "skipped": True,
                    "reason": "Part 0 pre-check gate failed; no deployable offset-span fusion was evaluated.",
                }
            ]
        )
        special_rows = []
        for case in cases:
            if str(case.case_id) not in {"1", "49"}:
                continue
            summary = offset_summary(stage3c0_dir, case)
            high_mask = summary["high_mask"].astype(bool)
            span_mask = np.zeros(case.full_len, dtype=bool)
            for start, end in mask_to_spans(high_mask, min_len=25):
                span_mask[start:end] = True
            long_mask = s3c0.long_wrong_mask(case.gt, case.official)
            base = case_metrics(data_root, case, case.official)
            special_rows.append(
                {
                    "skipped": True,
                    "reason": "Part 2 skipped by pre-check gate",
                    "fold": case.fold,
                    "case_id": case.case_id,
                    "video": case.video,
                    "official_acc": base["acc"],
                    "official_edit": base["edit"],
                    "official_f1@50": base["f1@50"],
                    "high_disagreement_frames": int(span_mask.sum()),
                    "long_error_frames": int(long_mask.sum()),
                    "long_error_in_high_disagreement_frames": int((long_mask & span_mask).sum()),
                    "eligible_long_error_mass_share_case": float((long_mask & span_mask).sum() / long_mask.sum()) if long_mask.sum() else float("nan"),
                }
            )
        special = pd.DataFrame(special_rows)
        skipped.to_csv(out_dir / "stage3c1_fusion_by_case.csv", index=False)
        skipped.to_csv(out_dir / "stage3c1_fusion_foldheldout.csv", index=False)
        special.to_csv(out_dir / "stage3c1_special_cases.csv", index=False)
        return skipped, skipped, special, {"part2_skipped": True}

    folds = sorted({c.fold for c in cases})
    methods = ["no_edit", "offset_majority", "best_confidence_offset"]
    configs = [(m, l) for m in methods for l in SPAN_LENGTHS]
    selected_by_fold: Dict[int, Tuple[str, int]] = {}
    for heldout in folds:
        train = [c for c in cases if c.fold != heldout]
        scored = []
        for method, min_len in configs:
            pred_map = {c.case_id: fusion_prediction(stage3c0_dir, c, method, min_len)[0] for c in train}
            base = aggregate_cases(data_root, train, official_map(train))
            m = aggregate_cases(data_root, train, pred_map)
            d = metric_delta(m, base)
            score = (d["delta_f1@50"], d["delta_edit"], d["delta_acc"])
            # Conservative Acc-neutral selection.
            if d["delta_acc"] < -0.1:
                score = (-999.0, -999.0, d["delta_acc"])
            scored.append((score, method, min_len))
        _, method, min_len = max(scored)
        selected_by_fold[heldout] = (method, min_len)

    for heldout in folds:
        held = [c for c in cases if c.fold == heldout]
        method, min_len = selected_by_fold[heldout]
        pred_map: Dict[str, np.ndarray] = {}
        spans_total = 0
        frames_total = 0
        for case in held:
            pred, n_spans, n_frames = fusion_prediction(stage3c0_dir, case, method, min_len)
            pred_map[case.case_id] = pred
            spans_total += n_spans
            frames_total += n_frames
            base_case = case_metrics(data_root, case, case.official)
            m_case = case_metrics(data_root, case, pred)
            row = {
                "fold": case.fold,
                "case_id": case.case_id,
                "video": case.video,
                "selected_method": method,
                "selected_min_span_len": min_len,
                "n_modified_spans": n_spans,
                "n_modified_frames": n_frames,
                **m_case,
            }
            row.update(metric_delta(m_case, base_case))
            by_case_rows.append(row)
        base = aggregate_cases(data_root, held, official_map(held))
        m = aggregate_cases(data_root, held, pred_map)
        row = {
            "heldout_fold": heldout,
            "selected_method": method,
            "selected_min_span_len": min_len,
            "n_cases": len(held),
            "n_modified_spans": spans_total,
            "n_modified_frames": frames_total,
            **m,
        }
        row.update(metric_delta(m, base))
        heldout_rows.append(row)

    by_case = pd.DataFrame(by_case_rows)
    heldout = pd.DataFrame(heldout_rows)
    special_ids = {"1", "49"}
    largest_pos = set(by_case.nlargest(3, "delta_f1@50")["case_id"].astype(str))
    largest_neg = set(by_case.nsmallest(3, "delta_f1@50")["case_id"].astype(str))
    special = by_case[by_case["case_id"].astype(str).isin(special_ids | largest_pos | largest_neg)].copy()
    by_case.to_csv(out_dir / "stage3c1_fusion_by_case.csv", index=False)
    heldout.to_csv(out_dir / "stage3c1_fusion_foldheldout.csv", index=False)
    special.to_csv(out_dir / "stage3c1_special_cases.csv", index=False)
    ci_f50 = bootstrap_ci(by_case["delta_f1@50"].to_numpy(dtype=float))
    ci_edit = bootstrap_ci(by_case["delta_edit"].to_numpy(dtype=float))
    summary = {
        "part2_skipped": False,
        "mean_delta_f1@50": float(by_case["delta_f1@50"].mean()),
        "mean_delta_edit": float(by_case["delta_edit"].mean()),
        "mean_delta_acc": float(by_case["delta_acc"].mean()),
        "f1@50_bootstrap_ci_low": ci_f50[0],
        "f1@50_bootstrap_ci_high": ci_f50[1],
        "edit_bootstrap_ci_low": ci_edit[0],
        "edit_bootstrap_ci_high": ci_edit[1],
        "n_helped_cases_f1@50": int((by_case["delta_f1@50"] > 0).sum()),
        "n_harmed_cases_f1@50": int((by_case["delta_f1@50"] < 0).sum()),
        "worst_case_delta_acc": float(by_case["delta_acc"].min()),
        "largest_case_delta_f1@50": float(by_case["delta_f1@50"].max()),
        "sum_positive_delta_f1@50": float(by_case.loc[by_case["delta_f1@50"] > 0, "delta_f1@50"].sum()),
    }
    return by_case, heldout, special, summary


def paper_delta(metrics_row: Dict[str, float]) -> Dict[str, float]:
    return {
        "minus_paper_acc": metrics_row["acc"] - PUBLISHED["50salads"]["Acc"],
        "minus_paper_edit": metrics_row["edit"] - PUBLISHED["50salads"]["Edit"],
        "minus_paper_f1@50": metrics_row["f1@50"] - PUBLISHED["50salads"]["F1@50"],
    }


def write_summary(
    out_dir: Path,
    data_root: Path,
    cases: Sequence[CaseData],
    prechecks: pd.DataFrame,
    fixed: pd.DataFrame,
    gate: Dict[str, Any],
    fusion_by_case: pd.DataFrame,
    fusion_heldout: pd.DataFrame,
    fusion_summary: Dict[str, Any],
) -> None:
    official = aggregate_cases(data_root, cases, official_map(cases))
    official_paper = paper_delta(official)
    held = fixed[fixed["selection_mode"] == "fold_heldout"].copy()
    test_swept = fixed[fixed["selection_mode"] == "candidate_all"].copy()
    best_swept = test_swept.sort_values(["delta_f1@50", "delta_edit", "delta_acc"], ascending=False).head(8)
    lines = [
        "# Stage-3C1 Offset Fusion Summary",
        "",
        "Baseline is official fold-local DiffAct `pred_full` from Stage-3C0. All metric deltas are versus that baseline unless explicitly marked as paper distance.",
        "",
        "## Official Baseline",
        "",
        df_to_markdown(
            pd.DataFrame(
                [
                    {
                        **official,
                        **official_paper,
                        "paper_acc": PUBLISHED["50salads"]["Acc"],
                        "paper_edit": PUBLISHED["50salads"]["Edit"],
                        "paper_f1@50": PUBLISHED["50salads"]["F1@50"],
                    }
                ]
            )
        ),
        "",
        "## Part 0 Pre-Checks",
        "",
        df_to_markdown(prechecks),
        "",
        "Gate criteria: direction-toward-GT > 0.55, cross-metric oracle retention >= 0.5, eligible long-substitution error mass >= 0.10, and no physical-video leakage.",
        "",
        f"Gate decision: **{'PASS' if gate['gate_pass'] else 'NO-GO'}**",
        "",
        "Pre-check interpretation:",
        "- Pre-median boundary headroom is close to post/official headroom, so the thesis that median destroyed substantial boundary information is weak for this cache.",
        "- The cross-metric oracle retains some F1@50 headroom, but the window noise-null is large; this is not a strong deployable selector by itself.",
        "- Offset-disagreement spans rarely point toward GT and cover only a small fraction of long-substitution error mass, so offset fusion is not an eligible/high-precision edit set.",
        "",
        "```json",
        json.dumps({k: v for k, v in gate.items() if k != "leaked_videos"}, indent=2, sort_keys=True),
        "```",
        "",
        "## Part 1 Fixed-Postprocess Sanity",
        "",
        "Best test-swept rows (optimistic, not deployable):",
        "",
        df_to_markdown(best_swept),
        "",
        "Fold-held-out rows:",
        "",
        df_to_markdown(held),
        "",
        "## Part 2 Conservative Offset Fusion",
        "",
    ]
    if fusion_summary.get("part2_skipped"):
        lines.extend(
            [
                "Part 2 was skipped because the Part 0 gate failed. No learned/adaptive offset-fusion method should be built from these cached offset signals.",
                "",
                "Final decision: **NO-GO** for Stage-3C1 offset-span fusion. Stop the metrics-improvement line here unless a new upstream substrate is introduced.",
            ]
        )
    else:
        lines.extend(
            [
                "Fold-held-out fusion rows:",
                "",
                df_to_markdown(fusion_heldout),
                "",
                "Fusion case summary:",
                "",
                df_to_markdown(pd.DataFrame([fusion_summary])),
            ]
        )
        go = (
            (
                fusion_summary["mean_delta_f1@50"] >= 0.3
                or fusion_summary["mean_delta_edit"] >= 0.3
            )
            and fusion_summary["mean_delta_acc"] >= -0.1
            and fusion_summary["f1@50_bootstrap_ci_low"] > 0
            and fusion_summary["n_helped_cases_f1@50"] >= fusion_summary["n_harmed_cases_f1@50"]
            and (
                fusion_summary["sum_positive_delta_f1@50"] == 0
                or fusion_summary["largest_case_delta_f1@50"] / fusion_summary["sum_positive_delta_f1@50"] < 0.5
            )
        )
        lines.append("")
        lines.append(f"Final decision: **{'GO' if go else 'NO-GO'}** for learned/adaptive offset-span fusion.")
    (out_dir / "stage3c1_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3C1 50Salads offset-span fusion pre-check.")
    parser.add_argument("--stage3c0-dir", type=Path, default=DEFAULT_STAGE3C0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke-fold", type=int, default=None, help="Restrict to one fold for smoke testing.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.stage3c0_dir, args.data_root, args.smoke_fold)
    if not cases:
        raise RuntimeError("No complete 50Salads cases found in Stage-3C0 cache.")
    print(f"Loaded {len(cases)} 50Salads cases", flush=True)

    prechecks, _, gate_pass, gate = build_prechecks(args.stage3c0_dir, args.data_root, args.out_dir, cases)
    print(f"Part 0 gate_pass={gate_pass}", flush=True)

    fixed = part1_fixed_sweep(args.stage3c0_dir, args.data_root, cases)
    fixed.to_csv(args.out_dir / "stage3c1_fixed_sweep.csv", index=False)

    by_case, heldout, special, fusion_summary = part2_fusion(args.stage3c0_dir, args.data_root, args.out_dir, cases, gate_pass and args.smoke_fold is None)
    write_summary(args.out_dir, args.data_root, cases, prechecks, fixed, gate, by_case, heldout, fusion_summary)
    final = {
        "n_cases": len(cases),
        "smoke_fold": args.smoke_fold,
        "stage3c0_dir": str(args.stage3c0_dir),
        "out_dir": str(args.out_dir),
        "gate_pass": bool(gate_pass),
        "part2_evaluated": bool(gate_pass and args.smoke_fold is None),
        "fusion_summary": fusion_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Stage-3C1 outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
