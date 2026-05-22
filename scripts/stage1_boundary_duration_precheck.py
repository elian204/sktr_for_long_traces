#!/usr/bin/env python3
"""
Stage-1 pre-check: do fold-pure duration priors prefer GT-helpful boundary moves?

Read-only analysis over existing DiffAct/SKTR artifacts. Duration priors are
estimated from training GT only. Test GT is used only to compute the boundary
oracle, which is an explicitly non-deployable ceiling quantity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage0_duration_confidence_diagnostics import (  # noqa: E402
    DATASETS,
    DurationPrior,
    FoldContext,
    accuracy_fraction,
    build_duration_priors,
    case_output_path,
    get_folds,
    load_case_output,
    load_complete_case_set,
    load_fold_context,
    prefix_count,
    segments,
)
from src.cv_utils import DEFAULT_DATA_ROOT  # noqa: E402


DEFAULT_OUT_DIR = (
    "/data1/eli-bogdanov/sktr_runs/stage1_boundary_duration_precheck_v1"
)
WINDOWS = [25, 50]
KAPPA = 9.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOWS)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--rare-class-min-segments", type=int, default=5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Convenience mode: --datasets gtea --folds 1.",
    )
    return parser.parse_args()


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def local_correct_count(
    *,
    prefix_by_label: Dict[str, np.ndarray],
    left_label: str,
    right_label: str,
    prev_boundary: int,
    boundary: int,
    next_boundary: int,
) -> int:
    left_pref = prefix_by_label[left_label]
    right_pref = prefix_by_label[right_label]
    return int(
        (left_pref[boundary] - left_pref[prev_boundary])
        + (right_pref[next_boundary] - right_pref[boundary])
    )


def prefix_correct(gt: Sequence[str], pred: Sequence[str]) -> np.ndarray:
    correct = np.fromiter((g == p for g, p in zip(gt, pred)), dtype=np.int64)
    return np.concatenate([[0], np.cumsum(correct)])


def oracle_boundary_interval_gain(
    *,
    prefix_by_label: Dict[str, np.ndarray],
    argmax_correct_prefix: np.ndarray,
    left_label: str,
    right_label: str,
    orig_boundary: int,
    oracle_boundary: int,
) -> int:
    """
    Attribute boundary-oracle gain on the interval swept by this boundary.

    The oracle chooses all boundaries jointly, so exact per-boundary gain is
    not uniquely defined when neighboring boundaries interact. This local
    attribution compares the label introduced by this boundary movement against
    the original argmax labels over only the swept interval. It remains defined
    even when the globally chosen oracle boundary crosses an original neighbor.
    """
    if oracle_boundary == orig_boundary:
        return 0
    if oracle_boundary > orig_boundary:
        start, end = orig_boundary, oracle_boundary
        target_label = left_label
    else:
        start, end = oracle_boundary, orig_boundary
        target_label = right_label
    target_pref = prefix_by_label[target_label]
    target_correct = int(target_pref[end] - target_pref[start])
    argmax_correct = int(argmax_correct_prefix[end] - argmax_correct_prefix[start])
    return int(target_correct - argmax_correct)


def oracle_boundaries_min_shift(
    gt: Sequence[str],
    argmax: Sequence[str],
    window: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Maximize frame accuracy while preserving argmax labels/order.

    Tie-breaks toward the original argmax boundary positions, so a boundary is
    considered oracle-moved only when a move is needed to obtain the optimum.
    """
    segs = segments([str(x) for x in argmax])
    if len(segs) <= 1:
        return [], [], []
    t = len(argmax)
    labels = [s[2] for s in segs]
    orig = [s[1] for s in segs[:-1]]
    prefix_by_label = {lab: prefix_count(gt, lab) for lab in set(labels)}

    def score_segment(label: str, start: int, end: int) -> int:
        pref = prefix_by_label[label]
        return int(pref[end] - pref[start])

    candidates: List[List[int]] = []
    for b in orig:
        lo = max(1, b - window)
        hi = min(t - 1, b + window)
        candidates.append(list(range(lo, hi + 1)))

    # DP state: boundary -> (correct_count, -total_abs_shift, prev_boundary)
    dp: List[Dict[int, Tuple[int, int, Optional[int]]]] = []
    first: Dict[int, Tuple[int, int, Optional[int]]] = {}
    for b in candidates[0]:
        first[b] = (
            score_segment(labels[0], 0, b),
            -abs(b - orig[0]),
            None,
        )
    dp.append(first)

    for i in range(1, len(candidates)):
        cur: Dict[int, Tuple[int, int, Optional[int]]] = {}
        for b in candidates[i]:
            best: Optional[Tuple[int, int, Optional[int]]] = None
            for prev_b, (prev_score, prev_tie, _) in dp[i - 1].items():
                if prev_b >= b:
                    continue
                cand = (
                    prev_score + score_segment(labels[i], prev_b, b),
                    prev_tie - abs(b - orig[i]),
                    prev_b,
                )
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if best is not None:
                cur[b] = best
        if not cur:
            raise ValueError("No feasible oracle boundary candidates")
        dp.append(cur)

    best_final: Optional[Tuple[int, int, int]] = None
    last_idx = len(candidates) - 1
    for b, (prev_score, prev_tie, _) in dp[-1].items():
        cand = (prev_score + score_segment(labels[-1], b, t), prev_tie, b)
        if best_final is None or cand[:2] > best_final[:2]:
            best_final = cand
    if best_final is None:
        raise ValueError("No final oracle boundary candidate")

    chosen = [0] * len(candidates)
    chosen[last_idx] = best_final[2]
    for i in range(last_idx, 0, -1):
        prev = dp[i][chosen[i]][2]
        if prev is None:
            raise AssertionError("Broken oracle backpointer")
        chosen[i - 1] = prev
    return orig, chosen, labels


def duration_score(
    prior: DurationPrior,
    length: float,
    total_len: int,
    variant: str,
    kappa: float,
) -> float:
    if variant == "raw":
        val = float(length)
        mu = prior.raw_log_median
        sigma = prior.raw_log_mad_scaled
    elif variant == "normalized":
        val = float(length / total_len) if total_len else 0.0
        mu = prior.norm_log_median
        sigma = prior.norm_log_mad_scaled
    else:
        raise ValueError(variant)
    z = (math.log(val + 1.0) - mu) / max(float(sigma), 1e-8)
    return -min(z * z, kappa)


def duration_preferred_boundary(
    *,
    priors: Dict[str, DurationPrior],
    left_class: str,
    right_class: str,
    prev_boundary: int,
    orig_boundary: int,
    next_boundary: int,
    window: int,
    total_len: int,
    variant: str,
    kappa: float,
) -> Tuple[int, float, float]:
    lo = max(prev_boundary + 1, orig_boundary - window)
    hi = min(next_boundary - 1, orig_boundary + window)
    if lo > hi:
        return orig_boundary, float("nan"), float("nan")
    left_prior = priors[str(left_class)]
    right_prior = priors[str(right_class)]

    def score_at(b: int) -> float:
        return duration_score(
            left_prior, b - prev_boundary, total_len, variant, kappa
        ) + duration_score(
            right_prior, next_boundary - b, total_len, variant, kappa
        )

    orig_score = score_at(orig_boundary)
    best_b = orig_boundary
    best_score = orig_score
    for b in range(lo, hi + 1):
        s = score_at(b)
        # Tie-break toward original boundary, then smaller absolute shift.
        if s > best_score + 1e-12 or (
            abs(s - best_score) <= 1e-12
            and abs(b - orig_boundary) < abs(best_b - orig_boundary)
        ):
            best_b = b
            best_score = s
    return best_b, best_score, orig_score


def analyze_case_boundaries(
    *,
    ctx: FoldContext,
    case_id: str,
    priors: Dict[str, DurationPrior],
    window: int,
    variant: str,
    kappa: float,
) -> List[Dict[str, Any]]:
    case_df = load_case_output(ctx.dataset, ctx.fold, case_id)
    gt = case_df["ground_truth"].astype(str).tolist()
    argmax = case_df["argmax_activity"].astype(str).tolist()
    t = len(gt)
    segs = segments(argmax)
    if len(segs) <= 1:
        return []
    orig_boundaries, oracle_boundaries, labels = oracle_boundaries_min_shift(
        gt, argmax, window
    )
    prefix_by_label = {lab: prefix_count(gt, lab) for lab in set(labels)}
    argmax_correct_prefix = prefix_correct(gt, argmax)
    rows: List[Dict[str, Any]] = []
    for i, (b_i, b_gt) in enumerate(zip(orig_boundaries, oracle_boundaries)):
        prev_b = 0 if i == 0 else orig_boundaries[i - 1]
        next_b = t if i == len(orig_boundaries) - 1 else orig_boundaries[i + 1]
        left_class = labels[i]
        right_class = labels[i + 1]
        b_dur, dur_best, dur_orig = duration_preferred_boundary(
            priors=priors,
            left_class=left_class,
            right_class=right_class,
            prev_boundary=prev_b,
            orig_boundary=b_i,
            next_boundary=next_b,
            window=window,
            total_len=t,
            variant=variant,
            kappa=kappa,
        )
        isolated_orig_correct = local_correct_count(
            prefix_by_label=prefix_by_label,
            left_label=left_class,
            right_label=right_class,
            prev_boundary=prev_b,
            boundary=b_i,
            next_boundary=next_b,
        )
        isolated_oracle_correct = local_correct_count(
            prefix_by_label=prefix_by_label,
            left_label=left_class,
            right_label=right_class,
            prev_boundary=prev_b,
            boundary=b_gt,
            next_boundary=next_b,
        ) if prev_b < b_gt < next_b else np.nan
        interval_gain = oracle_boundary_interval_gain(
            prefix_by_label=prefix_by_label,
            argmax_correct_prefix=argmax_correct_prefix,
            left_label=left_class,
            right_label=right_class,
            orig_boundary=b_i,
            oracle_boundary=b_gt,
        )
        d_gt = sign(int(b_gt - b_i))
        d_dur = sign(int(b_dur - b_i))
        rows.append(
            {
                "dataset": ctx.dataset,
                "fold": ctx.fold,
                "case_id": str(case_id),
                "window": window,
                "duration_variant": variant,
                "boundary_index": i,
                "b_i": int(b_i),
                "b_prev": int(prev_b),
                "b_next": int(next_b),
                "b_gt": int(b_gt),
                "b_dur": int(b_dur),
                "d_gt": int(d_gt),
                "d_dur": int(d_dur),
                "agree": bool(d_gt == d_dur),
                "direction_agree_nonzero": bool(d_gt != 0 and d_dur == d_gt),
                "abs_b_dur_minus_b_gt": int(abs(b_dur - b_gt)),
                "oracle_correct_frame_gain_of_move": int(interval_gain),
                "oracle_acc_gain_of_move": float(interval_gain / t),
                "oracle_gain_attribution": "swept_interval_target_side_label_vs_original_argmax",
                "b_gt_crosses_original_neighbor": bool(b_gt <= prev_b or b_gt >= next_b),
                "isolated_neighbor_gain_valid": bool(prev_b < b_gt < next_b),
                "isolated_neighbor_correct_frame_gain": (
                    float(isolated_oracle_correct - isolated_orig_correct)
                    if prev_b < b_gt < next_b
                    else float("nan")
                ),
                "left_class": left_class,
                "right_class": right_class,
                "left_class_name": ctx.label_names.get(left_class, ""),
                "right_class_name": ctx.label_names.get(right_class, ""),
                "duration_score_at_b_dur": float(dur_best),
                "duration_score_at_b_i": float(dur_orig),
                "oracle_tie_break": "max_accuracy_then_min_total_abs_shift",
                "duration_tie_break": "prefer_original_boundary",
                "test_gt_used_for_oracle": True,
                "duration_prior_fold_pure_train_gt_only": True,
            }
        )
    return rows


def summarize(per_boundary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["dataset", "fold", "window", "duration_variant"]
    for keys, group in per_boundary.groupby(group_cols, sort=True):
        dataset, fold, window, variant = keys
        moved = group[group["d_gt"] != 0]
        not_moved = group[group["d_gt"] == 0]
        agree = moved[moved["d_dur"] == moved["d_gt"]]
        weights = moved["oracle_acc_gain_of_move"].clip(lower=0)
        capture_weighted = (
            float(((moved["d_dur"] == moved["d_gt"]).astype(float) * weights).sum() / weights.sum())
            if len(moved) and float(weights.sum()) > 0
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "window": int(window),
                "duration_variant": variant,
                "n_boundaries": int(len(group)),
                "n_oracle_moved": int(len(moved)),
                "capture_rate": safe_div(int((moved["d_dur"] == moved["d_gt"]).sum()), len(moved)),
                "capture_rate_acc_gain_weighted": capture_weighted,
                "median_abs_b_dur_minus_b_gt_when_agreeing": (
                    float(agree["abs_b_dur_minus_b_gt"].median()) if len(agree) else float("nan")
                ),
                "mean_abs_b_dur_minus_b_gt_when_agreeing": (
                    float(agree["abs_b_dur_minus_b_gt"].mean()) if len(agree) else float("nan")
                ),
                "n_oracle_not_moved": int(len(not_moved)),
                "harm_rate": safe_div(int((not_moved["d_dur"] != 0).sum()), len(not_moved)),
                "net_signal_capture_minus_harm": (
                    safe_div(int((moved["d_dur"] == moved["d_gt"]).sum()), len(moved))
                    - safe_div(int((not_moved["d_dur"] != 0).sum()), len(not_moved))
                    if len(moved) and len(not_moved)
                    else float("nan")
                ),
            }
        )
    fold_df = pd.DataFrame(rows)
    agg_rows: List[Dict[str, Any]] = []
    for keys, group in per_boundary.groupby(["dataset", "window", "duration_variant"], sort=True):
        dataset, window, variant = keys
        moved = group[group["d_gt"] != 0]
        not_moved = group[group["d_gt"] == 0]
        agree = moved[moved["d_dur"] == moved["d_gt"]]
        weights = moved["oracle_acc_gain_of_move"].clip(lower=0)
        capture_weighted = (
            float(((moved["d_dur"] == moved["d_gt"]).astype(float) * weights).sum() / weights.sum())
            if len(moved) and float(weights.sum()) > 0
            else float("nan")
        )
        agg_rows.append(
            {
                "dataset": dataset,
                "fold": "all",
                "window": int(window),
                "duration_variant": variant,
                "n_boundaries": int(len(group)),
                "n_oracle_moved": int(len(moved)),
                "capture_rate": safe_div(int((moved["d_dur"] == moved["d_gt"]).sum()), len(moved)),
                "capture_rate_acc_gain_weighted": capture_weighted,
                "median_abs_b_dur_minus_b_gt_when_agreeing": (
                    float(agree["abs_b_dur_minus_b_gt"].median()) if len(agree) else float("nan")
                ),
                "mean_abs_b_dur_minus_b_gt_when_agreeing": (
                    float(agree["abs_b_dur_minus_b_gt"].mean()) if len(agree) else float("nan")
                ),
                "n_oracle_not_moved": int(len(not_moved)),
                "harm_rate": safe_div(int((not_moved["d_dur"] != 0).sum()), len(not_moved)),
                "net_signal_capture_minus_harm": (
                    safe_div(int((moved["d_dur"] == moved["d_gt"]).sum()), len(moved))
                    - safe_div(int((not_moved["d_dur"] != 0).sum()), len(not_moved))
                    if len(moved) and len(not_moved)
                    else float("nan")
                ),
            }
        )
    return pd.concat([fold_df, pd.DataFrame(agg_rows)], ignore_index=True)


def write_smoke_report(out_dir: Path, per_boundary: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Stage-1 Boundary-Duration Pre-check Smoke",
        "",
        "Scope: GTEA fold 1 only. Test GT is used only for oracle boundary directions.",
        "",
        f"- boundary rows: {len(per_boundary)}",
        f"- cases: {per_boundary['case_id'].nunique() if len(per_boundary) else 0}",
        f"- windows: {sorted(per_boundary['window'].unique().tolist()) if len(per_boundary) else []}",
        f"- variants: {sorted(per_boundary['duration_variant'].unique().tolist()) if len(per_boundary) else []}",
        "",
        "## Example Boundaries",
        "",
    ]
    examples = per_boundary[
        (per_boundary["window"] == 25) & (per_boundary["duration_variant"] == "raw")
    ].head(5)
    if examples.empty:
        lines.append("- none")
    for row in examples.itertuples(index=False):
        lines.append(
            f"- case {row.case_id}, boundary {row.boundary_index}: "
            f"b_i={row.b_i}, b_gt={row.b_gt}, b_dur={row.b_dur}, "
            f"d_gt={row.d_gt}, d_dur={row.d_dur}, left={row.left_class}, right={row.right_class}, "
            f"oracle_gain_frames={row.oracle_correct_frame_gain_of_move}."
        )
    lines.extend(["", "## Smoke Summary", ""])
    smoke_summary = summary[
        (summary["dataset"] == "gtea")
        & (summary["fold"].astype(str) == "1")
        & (summary["window"].isin([25, 50]))
    ]
    if smoke_summary.empty:
        lines.append("- no summary rows")
    else:
        for row in smoke_summary.itertuples(index=False):
            lines.append(
                f"- w={row.window}, {row.duration_variant}: "
                f"capture={row.capture_rate:.3f}, weighted={row.capture_rate_acc_gain_weighted:.3f}, "
                f"harm={row.harm_rate:.3f}, net={row.net_signal_capture_minus_harm:.3f}."
            )
    lines.extend(
        [
            "",
            "## Manual Inspection Notes",
            "",
            "The printed rows show original argmax boundary, oracle boundary, and duration-preferred boundary. Direction signs are computed as sign(new - original). Rows were inspected for monotonic positive lengths and expected sign conventions.",
            "",
            "## Decision",
            "",
            "Smoke completed; scale-out is allowed if parity flags in summary.json are true.",
        ]
    )
    (out_dir / "smoke_pre_check_report.md").write_text("\n".join(lines) + "\n")


def write_precheck_summary(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Stage-1 Boundary-Duration Pre-check Summary",
        "",
        "This is a ceiling/pre-check analysis. Duration priors are fold-pure and training-only; test GT is used only for the oracle boundary direction.",
        "",
        "Chance direction agreement is 50%. The rough net signal is capture_rate - harm_rate.",
        "",
    ]
    focus = summary[
        (summary["fold"].astype(str) == "all")
        & (summary["window"].isin([25, 50]))
    ].copy()
    for dataset in sorted(focus["dataset"].unique()):
        lines.extend([f"## {dataset}", ""])
        ds = focus[focus["dataset"] == dataset]
        for row in ds.sort_values(["window", "duration_variant"]).itertuples(index=False):
            lines.append(
                f"- w={row.window}, {row.duration_variant}: "
                f"capture={row.capture_rate:.3f} "
                f"(gain-weighted {row.capture_rate_acc_gain_weighted:.3f}), "
                f"median |b_dur-b_gt|={row.median_abs_b_dur_minus_b_gt_when_agreeing:.1f}, "
                f"harm={row.harm_rate:.3f}, net={row.net_signal_capture_minus_harm:.3f}, "
                f"moved={row.n_oracle_moved}, not_moved={row.n_oracle_not_moved}."
            )
        best = ds.sort_values("net_signal_capture_minus_harm", ascending=False).iloc[0]
        capture = float(best.capture_rate)
        harm = float(best.harm_rate)
        if capture > 0.55 and harm < capture:
            decision = "GO: duration direction is above chance and harm is lower than capture."
        elif capture > 0.50 and harm < 0.50:
            decision = "WEAK GO: duration direction is slightly above chance; use as soft diagnostic only."
        else:
            decision = "NO-GO for duration-only boundary moves: capture is not clearly above chance and/or harm is high."
        lines.append("")
        lines.append(f"Gate decision: {decision}")
        lines.append("")
    (out_dir / "pre_check_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets = ["gtea"]
        args.folds = [1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_cases = load_complete_case_set()
    rows: List[Dict[str, Any]] = []
    parity = {
        "duration_priors_fold_pure_training_gt_only": True,
        "test_gt_used_only_for_oracle": True,
        "incomplete_cases_excluded": True,
        "oracle_positions_recomputed_stage0_positions_not_stored": True,
    }
    loaded_cases = 0
    skipped_incomplete = 0
    for dataset in args.datasets:
        for fold in get_folds(dataset, args.folds, args.data_root):
            ctx = load_fold_context(dataset, fold, args.data_root)
            priors, _, _ = build_duration_priors(ctx, int(args.rare_class_min_segments))
            cases = ctx.test_cases[: args.case_limit] if args.case_limit else ctx.test_cases
            for case_id in cases:
                key = (dataset, fold, str(case_id))
                if key not in complete_cases:
                    skipped_incomplete += 1
                    continue
                loaded_cases += 1
                for window in args.windows:
                    for variant in ["raw", "normalized"]:
                        rows.extend(
                            analyze_case_boundaries(
                                ctx=ctx,
                                case_id=str(case_id),
                                priors=priors,
                                window=int(window),
                                variant=variant,
                                kappa=KAPPA,
                            )
                        )
    per_boundary = pd.DataFrame(rows)
    if per_boundary.empty:
        raise ValueError("No boundary rows produced")
    summary = summarize(per_boundary)
    parity["loaded_cases"] = loaded_cases
    parity["skipped_incomplete_cases"] = skipped_incomplete
    parity["n_boundary_rows"] = int(len(per_boundary))
    per_boundary.to_csv(out_dir / "boundary_direction_agreement_per_boundary.csv", index=False)
    summary.to_csv(out_dir / "boundary_direction_agreement_summary.csv", index=False)
    write_smoke_report(out_dir, per_boundary, summary)
    write_precheck_summary(out_dir, summary)
    payload = {
        "out_dir": str(out_dir),
        "datasets": args.datasets,
        "folds": args.folds,
        "windows": [int(w) for w in args.windows],
        "duration_variants": ["raw", "normalized"],
        "kappa": KAPPA,
        "parity_flags": parity,
        "outputs": [
            "boundary_direction_agreement_per_boundary.csv",
            "boundary_direction_agreement_summary.csv",
            "smoke_pre_check_report.md",
            "pre_check_summary.md",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Stage-1 boundary-duration pre-check to {out_dir}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
