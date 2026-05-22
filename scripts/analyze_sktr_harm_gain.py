#!/usr/bin/env python3
"""
Harm/gain and addressability diagnostics for completed DiffAct+SKTR runs.

This script is deliberately read-only over existing artifacts:
  * per-frame case_outputs/*.csv from the completed evaluation
  * all_ceiling_cases.csv from scripts/analyze_sktr_ceiling.py

It does not rerun SKTR or recompute Petri-net fitness. Addressability is
conservative: if the already-computed argmax collapsed alignment has no
log/model moves, argmax frame errors are non-addressable by the order-only
Petri-net prior. If the case does have argmax log/model moves, the script marks
those frame errors as requiring local alignment review rather than declaring
them addressable.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_diffact_sktr_fold1_paper import (  # noqa: E402
    load_diffact_softmax_and_aligned_df,
    resolve_diffact_softmax_dir,
    softmax_map_from_entries,
    verify_softmax_list,
)
from src.cv_utils import DEFAULT_DATA_ROOT  # noqa: E402


DEFAULT_CEILING_CSV = (
    "/data1/eli-bogdanov/sktr_runs/sktr_ceiling_analysis_gtea_skip_all_v4/"
    "all_ceiling_cases.csv"
)
DEFAULT_RUN_DIR = (
    "/data1/eli-bogdanov/sktr_runs/"
    "diffact_gtea_allfolds_resumable_6ba8868_chunk11_w7"
)
DEFAULT_OUT_DIR = (
    "/data1/eli-bogdanov/sktr_runs/"
    "sktr_harm_gain_gtea_tau_completed_v1"
)
DEFAULT_DATA_ROOT = str(DEFAULT_DATA_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SKTR harm/gain on selected completed DiffAct+SKTR cases."
    )
    parser.add_argument("--ceiling-csv", default=DEFAULT_CEILING_CSV)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--exclude-folds",
        nargs="*",
        type=int,
        default=[2],
        help="Folds excluded from this diagnostic. Default excludes GTEA fold 2.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help=(
            "Optional explicit cases to analyze. Tokens may be case_id or "
            "fold:case_id. Use fold:case_id when case IDs are not globally unique."
        ),
    )
    parser.add_argument(
        "--include-non-tau-completed",
        action="store_true",
        help=(
            "Do not require gt_accepted_exact_tau_completed=True. This is for "
            "targeted drills on failure cases; fitness state is reported from "
            "the ceiling CSV rather than recomputed."
        ),
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional prefix limit after filtering, for smoke tests.",
    )
    parser.add_argument("--manual-review-n", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--high-confidence", type=float, default=0.8)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=11,
        help="Chunk size used for frame index modulo diagnostics. Default matches SKTR chunking.",
    )
    return parser.parse_args()


def parse_list_cell(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list cell, got {type(parsed)}: {value!r}")
    return parsed


def label_probability_and_rank(
    *,
    probs_cell: Any,
    activities_cell: Any,
    label: Any,
) -> Tuple[float, Optional[int]]:
    probs = [float(x) for x in parse_list_cell(probs_cell)]
    activities = [str(x) for x in parse_list_cell(activities_cell)]
    label_s = str(label)
    if label_s not in activities:
        return 0.0, None
    idx = activities.index(label_s)
    prob = probs[idx]
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    rank = order.index(idx) + 1
    return prob, rank


def label_probability_and_competition_rank_from_vector(
    probs: np.ndarray,
    label: Any,
) -> Tuple[float, Optional[int]]:
    idx = int(label)
    if idx < 0 or idx >= probs.shape[0]:
        return 0.0, None
    prob = float(probs[idx])
    rank = int(np.count_nonzero(probs > prob) + 1)
    return prob, rank


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


def run_segments(labels: Sequence[Any]) -> List[Tuple[int, int, str]]:
    if not labels:
        return []
    out: List[Tuple[int, int, str]] = []
    start = 0
    current = str(labels[0])
    for idx in range(1, len(labels)):
        label = str(labels[idx])
        if label != current:
            out.append((start, idx, current))
            start = idx
            current = label
    out.append((start, len(labels), current))
    return out


def format_segments(segments: Sequence[Tuple[int, int, str]], max_items: int = 18) -> str:
    items = [f"{label}[{start}:{end}]" for start, end, label in segments[:max_items]]
    if len(segments) > max_items:
        items.append(f"... +{len(segments) - max_items} more")
    return " | ".join(items)


def load_selected_cases(
    ceiling_csv: Path,
    *,
    exclude_folds: Iterable[int],
    case_ids: Optional[Sequence[str]],
    include_non_tau_completed: bool,
    case_limit: Optional[int],
) -> pd.DataFrame:
    ceiling = pd.read_csv(ceiling_csv)
    required = {
        "dataset",
        "fold",
        "case_id",
        "gt_log_moves",
        "gt_model_moves",
        "gt_accepted_exact",
        "gt_accepted_exact_tau_completed",
        "gt_tau_search_truncated",
        "argmax_log_moves",
        "argmax_model_moves",
        "argmax_accepted_exact",
        "argmax_accepted_exact_tau_completed",
        "sktr_log_moves",
        "sktr_model_moves",
        "sktr_accepted_exact",
        "sktr_accepted_exact_tau_completed",
    }
    missing = required.difference(ceiling.columns)
    if missing:
        raise ValueError(f"{ceiling_csv} missing columns: {sorted(missing)}")

    selected = ceiling.copy()
    if case_ids:
        selected = filter_cases(selected, case_ids)
    if not include_non_tau_completed:
        selected = selected[selected["gt_accepted_exact_tau_completed"].astype(bool)]
    selected = selected[
        ~selected["gt_tau_search_truncated"].astype(bool)
        & ~selected["fold"].astype(int).isin(set(exclude_folds))
    ].copy()
    selected = selected.sort_values(["fold", "case_id"]).reset_index(drop=True)
    if case_limit is not None:
        selected = selected.head(case_limit)
    return selected


def filter_cases(ceiling: pd.DataFrame, case_ids: Sequence[str]) -> pd.DataFrame:
    requested_plain = set()
    requested_fold_case = set()
    for token in case_ids:
        token = str(token)
        if ":" in token:
            fold_s, case_s = token.split(":", 1)
            requested_fold_case.add((int(fold_s), str(case_s)))
        else:
            requested_plain.add(str(token))

    fold_case = set(zip(ceiling["fold"].astype(int), ceiling["case_id"].astype(str)))
    plain_cases = set(ceiling["case_id"].astype(str))
    missing_fold_case = sorted(requested_fold_case.difference(fold_case))
    missing_plain = sorted(requested_plain.difference(plain_cases))
    if missing_fold_case or missing_plain:
        raise ValueError(
            "Requested cases missing from ceiling CSV: "
            f"fold_case={missing_fold_case}, case_id={missing_plain}"
        )

    mask = ceiling["case_id"].astype(str).isin(requested_plain)
    if requested_fold_case:
        mask = mask | pd.Series(
            [
                (int(fold), str(case_id)) in requested_fold_case
                for fold, case_id in zip(ceiling["fold"], ceiling["case_id"])
            ],
            index=ceiling.index,
        )
    return ceiling[mask].copy()


def case_output_path(run_dir: Path, dataset: str, fold: int, case_id: str) -> Path:
    return run_dir / "case_outputs" / f"{dataset}_fold{fold}" / f"{case_id}.csv"


def enrich_case_frames(
    case_df: pd.DataFrame,
    ceiling_row: pd.Series,
    raw_softmax: np.ndarray,
) -> pd.DataFrame:
    frame = case_df.copy()
    if raw_softmax.shape[1] != len(frame):
        raise ValueError(
            f"case {ceiling_row['case_id']}: raw softmax T={raw_softmax.shape[1]} "
            f"!= case output rows={len(frame)}"
        )
    raw_argmax = raw_softmax.argmax(axis=0).astype(str)
    csv_argmax = frame["argmax_activity"].astype(str).to_numpy()
    mismatches = np.flatnonzero(raw_argmax != csv_argmax)
    if len(mismatches):
        sample = ", ".join(
            f"t={int(t)} raw={raw_argmax[t]} csv={csv_argmax[t]}"
            for t in mismatches[:10]
        )
        raise ValueError(
            f"case {ceiling_row['case_id']}: raw softmax argmax disagrees with "
            f"case_outputs argmax at {len(mismatches)} frame(s): {sample}"
        )
    gt = frame["ground_truth"].astype(str)
    argmax = frame["argmax_activity"].astype(str)
    sktr = frame["sktr_activity"].astype(str)

    frame["gt"] = gt
    frame["argmax"] = argmax
    frame["sktr"] = sktr
    frame["argmax_correct"] = argmax == gt
    frame["sktr_correct"] = sktr == gt
    frame["helped"] = ~frame["argmax_correct"] & frame["sktr_correct"]
    frame["harmed"] = frame["argmax_correct"] & ~frame["sktr_correct"]
    frame["both_wrong"] = ~frame["argmax_correct"] & ~frame["sktr_correct"]
    frame["both_correct"] = frame["argmax_correct"] & frame["sktr_correct"]
    frame["argmax_sktr_diverge"] = argmax != sktr

    gt_probs: List[float] = []
    gt_ranks: List[Optional[int]] = []
    argmax_probs: List[float] = []
    argmax_ranks: List[Optional[int]] = []
    sktr_probs: List[float] = []
    sktr_ranks: List[Optional[int]] = []
    rounded_gt_probs: List[float] = []
    rounded_gt_ranks: List[Optional[int]] = []
    rounded_argmax_probs: List[float] = []
    rounded_sktr_probs: List[float] = []
    for t, row in enumerate(frame.itertuples(index=False)):
        probs = raw_softmax[:, t]
        gt_prob, gt_rank = label_probability_and_competition_rank_from_vector(
            probs, row.ground_truth
        )
        arg_prob, arg_rank = label_probability_and_competition_rank_from_vector(
            probs, row.argmax_activity
        )
        sk_prob, sk_rank = label_probability_and_competition_rank_from_vector(
            probs, row.sktr_activity
        )
        rounded_gt_prob, rounded_gt_rank = label_probability_and_rank(
            probs_cell=row.all_probs,
            activities_cell=row.all_activities,
            label=row.ground_truth,
        )
        rounded_arg_prob, _ = label_probability_and_rank(
            probs_cell=row.all_probs,
            activities_cell=row.all_activities,
            label=row.argmax_activity,
        )
        rounded_sk_prob, _ = label_probability_and_rank(
            probs_cell=row.all_probs,
            activities_cell=row.all_activities,
            label=row.sktr_activity,
        )
        gt_probs.append(gt_prob)
        gt_ranks.append(gt_rank)
        argmax_probs.append(arg_prob)
        argmax_ranks.append(arg_rank)
        sktr_probs.append(sk_prob)
        sktr_ranks.append(sk_rank)
        rounded_gt_probs.append(rounded_gt_prob)
        rounded_gt_ranks.append(rounded_gt_rank)
        rounded_argmax_probs.append(rounded_arg_prob)
        rounded_sktr_probs.append(rounded_sk_prob)

    frame["gt_prob"] = gt_probs
    frame["gt_rank"] = gt_ranks
    frame["argmax_prob"] = argmax_probs
    frame["argmax_rank"] = argmax_ranks
    frame["sktr_prob"] = sktr_probs
    frame["sktr_rank"] = sktr_ranks
    frame["rounded_gt_prob"] = rounded_gt_probs
    frame["rounded_gt_rank"] = rounded_gt_ranks
    frame["rounded_argmax_prob"] = rounded_argmax_probs
    frame["rounded_sktr_prob"] = rounded_sktr_probs

    argmax_order_violation = (
        int(ceiling_row["argmax_log_moves"]) + int(ceiling_row["argmax_model_moves"])
    ) > 0
    frame["argmax_case_has_order_violation"] = argmax_order_violation
    frame["argmax_error_model_addressable_confirmed"] = False
    frame["argmax_error_requires_local_addressability_review"] = (
        ~frame["argmax_correct"] & argmax_order_violation
    )
    frame["argmax_error_non_addressable_conservative"] = (
        ~frame["argmax_correct"] & ~argmax_order_violation
    )
    frame["sktr_case_tau_completed"] = bool(ceiling_row["sktr_accepted_exact_tau_completed"])
    frame["argmax_case_tau_completed"] = bool(ceiling_row["argmax_accepted_exact_tau_completed"])
    return frame


def summarize_case(frame: pd.DataFrame, ceiling_row: pd.Series, top_k: int, high_conf: float) -> Dict[str, Any]:
    n = len(frame)
    argmax_errors = int((~frame["argmax_correct"]).sum())
    sktr_errors = int((~frame["sktr_correct"]).sum())
    divergence = frame["argmax_sktr_diverge"]
    harmed = frame["harmed"]
    helped = frame["helped"]
    argmax_order_violation = bool(frame["argmax_case_has_order_violation"].iloc[0])
    divergence_spans = contiguous_spans(divergence.astype(bool).tolist())
    harmed_spans = contiguous_spans(harmed.astype(bool).tolist())
    helped_spans = contiguous_spans(helped.astype(bool).tolist())
    divergence_lengths = [end - start for start, end in divergence_spans]
    harmed_lengths = [end - start for start, end in harmed_spans]
    helped_lengths = [end - start for start, end in helped_spans]
    divergence_mod_counts = (
        frame.loc[divergence, "frame_mod_chunk_size"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
        if divergence.any()
        else {}
    )
    harmed_mod_counts = (
        frame.loc[harmed, "frame_mod_chunk_size"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
        if harmed.any()
        else {}
    )
    return {
        "dataset": ceiling_row["dataset"],
        "fold": int(ceiling_row["fold"]),
        "case_id": str(ceiling_row["case_id"]),
        "n_frames": n,
        "argmax_acc": float(frame["argmax_correct"].mean()),
        "sktr_acc": float(frame["sktr_correct"].mean()),
        "sktr_minus_argmax_acc": float(frame["sktr_correct"].mean() - frame["argmax_correct"].mean()),
        "argmax_errors": argmax_errors,
        "sktr_errors": sktr_errors,
        "helped_frames": int(helped.sum()),
        "harmed_frames": int(harmed.sum()),
        "both_wrong_frames": int(frame["both_wrong"].sum()),
        "divergence_frames": int(divergence.sum()),
        "divergence_span_count": len(divergence_spans),
        "mean_divergence_span_len": float(np.mean(divergence_lengths)) if divergence_lengths else np.nan,
        "median_divergence_span_len": float(np.median(divergence_lengths)) if divergence_lengths else np.nan,
        "max_divergence_span_len": int(max(divergence_lengths)) if divergence_lengths else 0,
        "harmed_span_count": len(harmed_spans),
        "mean_harmed_span_len": float(np.mean(harmed_lengths)) if harmed_lengths else np.nan,
        "median_harmed_span_len": float(np.median(harmed_lengths)) if harmed_lengths else np.nan,
        "max_harmed_span_len": int(max(harmed_lengths)) if harmed_lengths else 0,
        "helped_span_count": len(helped_spans),
        "mean_helped_span_len": float(np.mean(helped_lengths)) if helped_lengths else np.nan,
        "median_helped_span_len": float(np.median(helped_lengths)) if helped_lengths else np.nan,
        "max_helped_span_len": int(max(helped_lengths)) if helped_lengths else 0,
        "divergence_mod0_frames": int(
            (divergence & (frame["frame_mod_chunk_size"].astype(int) == 0)).sum()
        ),
        "harmed_mod0_frames": int(
            (harmed & (frame["frame_mod_chunk_size"].astype(int) == 0)).sum()
        ),
        "divergence_mod_histogram": json.dumps({str(k): int(v) for k, v in divergence_mod_counts.items()}),
        "harmed_mod_histogram": json.dumps({str(k): int(v) for k, v in harmed_mod_counts.items()}),
        "gt_log_moves": int(ceiling_row["gt_log_moves"]),
        "gt_model_moves": int(ceiling_row["gt_model_moves"]),
        "gt_accepted_exact": bool(ceiling_row["gt_accepted_exact"]),
        "gt_accepted_exact_tau_completed": bool(ceiling_row["gt_accepted_exact_tau_completed"]),
        "gt_tau_search_truncated": bool(ceiling_row["gt_tau_search_truncated"]),
        "argmax_case_has_order_violation": argmax_order_violation,
        "argmax_log_moves": int(ceiling_row["argmax_log_moves"]),
        "argmax_model_moves": int(ceiling_row["argmax_model_moves"]),
        "argmax_accepted_exact": bool(ceiling_row["argmax_accepted_exact"]),
        "argmax_accepted_exact_tau_completed": bool(
            ceiling_row["argmax_accepted_exact_tau_completed"]
        ),
        "sktr_log_moves": int(ceiling_row["sktr_log_moves"]),
        "sktr_model_moves": int(ceiling_row["sktr_model_moves"]),
        "sktr_accepted_exact": bool(ceiling_row["sktr_accepted_exact"]),
        "sktr_accepted_exact_tau_completed": bool(
            ceiling_row["sktr_accepted_exact_tau_completed"]
        ),
        "argmax_error_model_addressable_confirmed": int(
            frame["argmax_error_model_addressable_confirmed"].sum()
        ),
        "argmax_error_requires_local_addressability_review": int(
            frame["argmax_error_requires_local_addressability_review"].sum()
        ),
        "argmax_error_non_addressable_conservative": int(
            frame["argmax_error_non_addressable_conservative"].sum()
        ),
        "harmed_high_conf_frames": int((harmed & (frame["argmax_prob"] >= high_conf)).sum()),
        "helped_gt_topk_frames": int((helped & (frame["gt_rank"].fillna(10**9) <= top_k)).sum()),
        "divergence_gt_topk_frames": int((divergence & (frame["gt_rank"].fillna(10**9) <= top_k)).sum()),
        "divergence_gt_not_topk_frames": int((divergence & (frame["gt_rank"].fillna(10**9) > top_k)).sum()),
        "model_consistent_harm_frames": int(
            (harmed & frame["sktr_case_tau_completed"]).sum()
        ),
        "mean_harmed_argmax_prob": float(frame.loc[harmed, "argmax_prob"].mean()) if harmed.any() else np.nan,
        "median_harmed_argmax_prob": float(frame.loc[harmed, "argmax_prob"].median()) if harmed.any() else np.nan,
        "p90_harmed_argmax_prob": float(frame.loc[harmed, "argmax_prob"].quantile(0.9)) if harmed.any() else np.nan,
        "max_harmed_argmax_prob": float(frame.loc[harmed, "argmax_prob"].max()) if harmed.any() else np.nan,
        "mean_helped_gt_prob": float(frame.loc[helped, "gt_prob"].mean()) if helped.any() else np.nan,
        "median_divergence_gt_rank": float(frame.loc[divergence, "gt_rank"].median()) if divergence.any() else np.nan,
    }


def make_manual_review(
    *,
    case_frames: Dict[Tuple[int, str], pd.DataFrame],
    case_summaries: pd.DataFrame,
    out_path: Path,
    n_cases: int,
    top_k: int,
    scope: str,
) -> None:
    ranked = case_summaries.copy()
    ranked["interesting"] = (
        ranked["harmed_frames"].astype(int)
        + ranked["helped_frames"].astype(int)
        + ranked["divergence_frames"].astype(int)
    )
    ranked = ranked.sort_values(
        ["interesting", "harmed_frames", "helped_frames"],
        ascending=False,
    ).head(n_cases)

    lines: List[str] = [
        "# Manual Review Smoke",
        "",
        f"Scope: {scope}",
        "Addressability is conservative: if argmax collapsed alignment has no log/model moves,",
        "argmax errors are treated as non-addressable by the order-only Petri-net prior.",
        "If argmax or GT has log/model moves, local interpretation is required; the script",
        "does not turn those moves into automatic addressability claims.",
        "",
    ]

    for row in ranked.itertuples(index=False):
        key = (int(row.fold), str(row.case_id))
        frame = case_frames[key]
        lines.extend(
            [
                f"## Fold {row.fold}, Case {row.case_id}",
                "",
                f"- Frames: {row.n_frames}",
                f"- Argmax acc: {row.argmax_acc:.6f}",
                f"- SKTR acc: {row.sktr_acc:.6f}",
                f"- Helped frames: {row.helped_frames}",
                f"- Harmed frames: {row.harmed_frames}",
                f"- GT fitness state: accepted_exact={row.gt_accepted_exact}, "
                f"tau_completed={row.gt_accepted_exact_tau_completed}, "
                f"log={row.gt_log_moves}, model={row.gt_model_moves}",
                f"- Argmax order violation: {row.argmax_case_has_order_violation} "
                f"(accepted_exact={row.argmax_accepted_exact}, "
                f"tau_completed={row.argmax_accepted_exact_tau_completed}, "
                f"log={row.argmax_log_moves}, model={row.argmax_model_moves})",
                f"- SKTR fitness state: accepted_exact={row.sktr_accepted_exact}, "
                f"tau_completed={row.sktr_accepted_exact_tau_completed}, "
                f"log={row.sktr_log_moves}, model={row.sktr_model_moves}",
                f"- Conservative non-addressable argmax errors: "
                f"{row.argmax_error_non_addressable_conservative}/{row.argmax_errors}",
                f"- Divergence spans: count={row.divergence_span_count}, "
                f"median_len={row.median_divergence_span_len}, max_len={row.max_divergence_span_len}, "
                f"mod0={row.divergence_mod0_frames}/{row.divergence_frames}",
                "",
                "Segments:",
                f"- GT    : {format_segments(run_segments(frame['gt'].tolist()))}",
                f"- Argmax: {format_segments(run_segments(frame['argmax'].tolist()))}",
                f"- SKTR  : {format_segments(run_segments(frame['sktr'].tolist()))}",
                "",
                "Top divergence spans:",
            ]
        )
        spans = contiguous_spans(frame["argmax_sktr_diverge"].astype(bool).tolist())
        if not spans:
            lines.append("- none")
        for start, end in spans[:8]:
            mid = start
            sample = frame.iloc[mid]
            lines.append(
                f"- [{start}:{end}] len={end-start} "
                f"GT={sample['gt']} Argmax={sample['argmax']} SKTR={sample['sktr']} "
                f"gt_rank={sample['gt_rank']} gt_prob={sample['gt_prob']:.4f} "
                f"argmax_prob={sample['argmax_prob']:.4f} sktr_prob={sample['sktr_prob']:.4f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def write_outputs(
    *,
    out_dir: Path,
    frame_rows: List[pd.DataFrame],
    case_rows: List[Dict[str, Any]],
    selected_cases: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    frames = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame()
    cases = pd.DataFrame(case_rows)

    frames.to_csv(out_dir / "frame_diagnostics.csv", index=False)
    cases.to_csv(out_dir / "case_summary.csv", index=False)
    selected_cases.to_csv(out_dir / "selected_cases.csv", index=False)
    selected_cases.to_csv(out_dir / "selected_tau_completed_cases.csv", index=False)

    total_frames = int(cases["n_frames"].sum()) if not cases.empty else 0
    total_argmax_errors = int(cases["argmax_errors"].sum()) if not cases.empty else 0
    total_divergence = int(cases["divergence_frames"].sum()) if not cases.empty else 0
    total_harmed = int(cases["harmed_frames"].sum()) if not cases.empty else 0
    total_helped = int(cases["helped_frames"].sum()) if not cases.empty else 0
    total_non_addressable = (
        int(cases["argmax_error_non_addressable_conservative"].sum())
        if not cases.empty
        else 0
    )
    total_addressable_confirmed = (
        int(cases["argmax_error_model_addressable_confirmed"].sum())
        if not cases.empty
        else 0
    )
    total_requires_review = (
        int(cases["argmax_error_requires_local_addressability_review"].sum())
        if not cases.empty
        else 0
    )
    harmed_argmax_probs = frames.loc[frames["harmed"].astype(bool), "argmax_prob"] if not frames.empty else pd.Series(dtype=float)
    divergence_gt_ranks = frames.loc[
        frames["argmax_sktr_diverge"].astype(bool), "gt_rank"
    ] if not frames.empty else pd.Series(dtype=float)

    summary = {
        "dataset": (
            str(cases["dataset"].iloc[0])
            if not cases.empty and cases["dataset"].nunique() == 1
            else "mixed"
        ),
        "scope": (
            "explicit/non-tau-completed cases allowed"
            if args.include_non_tau_completed
            else "tau-completed GT cases from ceiling CSV"
        ),
        "excluded_folds": args.exclude_folds,
        "requested_case_ids": args.case_ids,
        "include_non_tau_completed": args.include_non_tau_completed,
        "ceiling_csv": args.ceiling_csv,
        "run_dir": args.run_dir,
        "data_root": args.data_root,
        "n_cases": int(len(cases)),
        "n_frames": total_frames,
        "frame_argmax_acc": (
            float(np.average(cases["argmax_acc"], weights=cases["n_frames"]))
            if total_frames
            else None
        ),
        "frame_sktr_acc": (
            float(np.average(cases["sktr_acc"], weights=cases["n_frames"]))
            if total_frames
            else None
        ),
        "helped_frames": total_helped,
        "harmed_frames": total_harmed,
        "net_sktr_minus_argmax_correct_frames": total_helped - total_harmed,
        "argmax_error_frames": total_argmax_errors,
        "argmax_error_model_addressable_confirmed": total_addressable_confirmed,
        "argmax_error_requires_local_addressability_review": total_requires_review,
        "argmax_error_non_addressable_conservative": total_non_addressable,
        "argmax_error_non_addressable_share": (
            total_non_addressable / total_argmax_errors if total_argmax_errors else None
        ),
        "divergence_frames": total_divergence,
        "divergence_gt_topk_frames": int(cases["divergence_gt_topk_frames"].sum()) if not cases.empty else 0,
        "divergence_gt_not_topk_frames": int(cases["divergence_gt_not_topk_frames"].sum()) if not cases.empty else 0,
        "divergence_span_count": int(cases["divergence_span_count"].sum()) if not cases.empty else 0,
        "max_divergence_span_len": int(cases["max_divergence_span_len"].max()) if not cases.empty else 0,
        "median_case_median_divergence_span_len": (
            float(cases["median_divergence_span_len"].median())
            if not cases.empty and cases["median_divergence_span_len"].notna().any()
            else None
        ),
        "divergence_mod0_frames": int(cases["divergence_mod0_frames"].sum()) if not cases.empty else 0,
        "harmed_mod0_frames": int(cases["harmed_mod0_frames"].sum()) if not cases.empty else 0,
        "gt_accepted_exact_cases": int(cases["gt_accepted_exact"].sum()) if not cases.empty else 0,
        "gt_tau_completed_cases": int(cases["gt_accepted_exact_tau_completed"].sum()) if not cases.empty else 0,
        "argmax_accepted_exact_cases": int(cases["argmax_accepted_exact"].sum()) if not cases.empty else 0,
        "argmax_tau_completed_cases": int(cases["argmax_accepted_exact_tau_completed"].sum()) if not cases.empty else 0,
        "sktr_accepted_exact_cases": int(cases["sktr_accepted_exact"].sum()) if not cases.empty else 0,
        "sktr_tau_completed_cases": int(cases["sktr_accepted_exact_tau_completed"].sum()) if not cases.empty else 0,
        "harmed_high_conf_frames": int(cases["harmed_high_conf_frames"].sum()) if not cases.empty else 0,
        "model_consistent_harm_frames": int(cases["model_consistent_harm_frames"].sum()) if not cases.empty else 0,
        "harmed_argmax_prob_mean": float(harmed_argmax_probs.mean()) if len(harmed_argmax_probs) else None,
        "harmed_argmax_prob_median": float(harmed_argmax_probs.median()) if len(harmed_argmax_probs) else None,
        "harmed_argmax_prob_p90": float(harmed_argmax_probs.quantile(0.9)) if len(harmed_argmax_probs) else None,
        "harmed_argmax_prob_max": float(harmed_argmax_probs.max()) if len(harmed_argmax_probs) else None,
        "divergence_gt_rank_median": float(divergence_gt_ranks.median()) if len(divergence_gt_ranks) else None,
        "top_k": args.top_k,
        "high_confidence": args.high_confidence,
        "confidence_source": "raw DiffAct softmax .npy bundle",
        "rank_definition": "competition rank: 1 + count(probability strictly greater)",
        "falsifiable_metrics": {
            "boundary_jitter_amplification": (
                "Many short divergence/harm spans, usually near segment edges, with "
                f"GT in top-{args.top_k}; SKTR changes timing rather than activity order."
            ),
            "model_misdirection": (
                "One or a few long contiguous harm spans where argmax was correct and "
                "confident while SKTR stays on a wrong label."
            ),
            "chunking_pathology": (
                f"Divergence or harm frames cluster at frame_index % {args.chunk_size} == 0."
            ),
            "boundary_repair": (
                "Helped frames appear in short spans near boundaries and reduce "
                "argmax over-segmentation or timing drift."
            ),
            "long_substitution_correction": (
                "Helped frames form long spans where SKTR replaces an extended wrong "
                "argmax activity with the GT label."
            ),
            "case_content_sympathy": (
                "Specific cases show better SKTR behavior despite low tau-completion, "
                "suggesting the discovered net matches those case bodies better than "
                "the fold-level end-marking topology suggests."
            ),
            "penalty_too_strong": (
                "harmed_argmax_prob distribution is high and/or "
                "harmed_high_conf_frames / harmed_frames is high: SKTR overrides "
                "confident correct argmax frames."
            ),
            "non_addressable": (
                "argmax_error_non_addressable_share is high and/or GT is not in "
                f"top-{args.top_k} on divergence frames."
            ),
            "model_misleads": (
                "model_consistent_harm_frames is high: SKTR remains model-consistent "
                "while moving correct argmax frames to wrong labels."
            ),
        },
        "guardrails": {
            "reran_sktr": False,
            "tau_completed_filter_reused_from_ceiling": not args.include_non_tau_completed,
            "non_tau_completed_cases_allowed_by_flag": bool(args.include_non_tau_completed),
            "fold2_excluded_by_default": 2 in set(args.exclude_folds),
            "addressability_requires_local_review_for_violation_cases": True,
            "confidence_uses_raw_softmax": True,
            "raw_softmax_argmax_matches_csv": True,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected_cases(
        Path(args.ceiling_csv),
        exclude_folds=args.exclude_folds,
        case_ids=args.case_ids,
        include_non_tau_completed=args.include_non_tau_completed,
        case_limit=args.case_limit,
    )
    if selected.empty:
        raise ValueError("No cases selected after case/tau/excluded-fold filtering")

    frame_rows: List[pd.DataFrame] = []
    case_rows: List[Dict[str, Any]] = []
    case_frames: Dict[Tuple[int, str], pd.DataFrame] = {}
    run_dir = Path(args.run_dir)
    diffact_root = REPO_ROOT / "baselines" / "DiffAct"
    softmax_by_fold: Dict[int, Dict[str, np.ndarray]] = {}

    for ceiling_row in selected.itertuples(index=False):
        fold = int(ceiling_row.fold)
        case_id = str(ceiling_row.case_id)
        path = case_output_path(run_dir, str(ceiling_row.dataset), fold, case_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        if fold not in softmax_by_fold:
            softmax_dir = resolve_diffact_softmax_dir(
                diffact_root,
                str(ceiling_row.dataset),
                fold,
                disallow_legacy=True,
            )
            _, softmax_lst, entries = load_diffact_softmax_and_aligned_df(
                str(ceiling_row.dataset),
                softmax_dir,
                Path(args.data_root),
            )
            verify_softmax_list(softmax_lst, f"{ceiling_row.dataset} fold {fold}")
            softmax_by_fold[fold] = softmax_map_from_entries(entries, softmax_lst)
        if case_id not in softmax_by_fold[fold]:
            raise KeyError(f"Missing raw softmax for fold {fold}, case {case_id}")
        raw = pd.read_csv(path)
        frame = enrich_case_frames(
            raw,
            pd.Series(ceiling_row._asdict()),
            raw_softmax=softmax_by_fold[fold][case_id],
        )
        frame.insert(0, "dataset", str(ceiling_row.dataset))
        frame.insert(1, "fold", fold)
        frame.insert(2, "case_id", case_id)
        frame["frame_mod_chunk_size"] = np.arange(len(frame), dtype=int) % int(args.chunk_size)
        frame_rows.append(frame)
        case_frames[(fold, case_id)] = frame
        case_rows.append(
            summarize_case(
                frame,
                pd.Series(ceiling_row._asdict()),
                top_k=args.top_k,
                high_conf=args.high_confidence,
            )
        )

    summary = write_outputs(
        out_dir=out_dir,
        frame_rows=frame_rows,
        case_rows=case_rows,
        selected_cases=selected,
        args=args,
    )
    make_manual_review(
        case_frames=case_frames,
        case_summaries=pd.DataFrame(case_rows),
        out_path=out_dir / "manual_review_smoke.md",
        n_cases=args.manual_review_n,
        top_k=args.top_k,
        scope=summary["scope"],
    )

    print(f"Wrote harm/gain diagnostics to {out_dir}")
    print(
        "Scope: "
        f"{summary['n_cases']} cases, {summary['n_frames']} frames, "
        f"excluded folds={summary['excluded_folds']}"
    )
    print(
        "Frames: "
        f"helped={summary['helped_frames']}, harmed={summary['harmed_frames']}, "
        f"net={summary['net_sktr_minus_argmax_correct_frames']}"
    )
    print(
        "Argmax errors: "
        f"addressable_confirmed={summary['argmax_error_model_addressable_confirmed']}, "
        f"requires_local_review={summary['argmax_error_requires_local_addressability_review']}, "
        f"non_addressable_conservative={summary['argmax_error_non_addressable_conservative']}"
    )


if __name__ == "__main__":
    main()
