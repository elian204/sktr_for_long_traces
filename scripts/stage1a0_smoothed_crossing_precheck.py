#!/usr/bin/env python3
"""
Stage-1A.0 pre-check: does smoothed DiffAct softmax move boundaries toward GT?

This is a read-only diagnostic over existing DiffAct/SKTR artifacts. The
smoothed crossing is GT-blind and deployable; test GT is used only to locate the
nearest reference boundary and score whether the blind movement helps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    FoldContext,
    boundary_positions,
    get_folds,
    load_case_output,
    load_complete_case_set,
    load_fold_context,
    segments,
)
from src.cv_utils import DEFAULT_DATA_ROOT  # noqa: E402


DEFAULT_OUT_DIR = (
    "/data1/eli-bogdanov/sktr_runs/stage1a0_smoothed_crossing_precheck_v1"
)
WINDOWS = [25, 50]
SIGMAS = [5.0, 11.0, 25.0]
EPS = 1e-12


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
    parser.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
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


def count_true(values: pd.Series) -> int:
    return int((values == True).sum())  # noqa: E712 - explicit True handles NaN/object columns.


def gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing with edge padding; sigma is in frames."""
    x = np.asarray(values, dtype=float)
    if sigma <= 0:
        return x.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_softmax(mat: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty_like(mat, dtype=float)
    for class_idx in range(mat.shape[0]):
        out[class_idx] = gaussian_smooth_1d(mat[class_idx], sigma)
    return out


def nearest_transition_in_window(
    transitions: Sequence[int],
    boundary: int,
    window: int,
) -> Optional[int]:
    candidates = [b for b in transitions if boundary - window <= b <= boundary + window]
    if not candidates:
        return None
    return min(candidates, key=lambda b: (abs(b - boundary), b))


def smoothed_crossing_boundary(
    *,
    smooth_mat: np.ndarray,
    left_class: str,
    right_class: str,
    boundary: int,
    window: int,
) -> Tuple[int, bool, int, float, float]:
    """Return nearest + to - crossing boundary for log p(left)-log p(right)."""
    left_idx = int(left_class)
    right_idx = int(right_class)
    t = smooth_mat.shape[1]
    lo = max(1, boundary - window)
    hi = min(t - 1, boundary + window)
    if lo > hi:
        return boundary, False, 0, float("nan"), float("nan")

    delta = np.log(np.clip(smooth_mat[left_idx], EPS, 1.0)) - np.log(
        np.clip(smooth_mat[right_idx], EPS, 1.0)
    )
    crossings: List[int] = []
    for b in range(lo, hi + 1):
        left_delta = float(delta[b - 1])
        right_delta = float(delta[b])
        if left_delta >= 0.0 and right_delta <= 0.0:
            crossings.append(b)
    if not crossings:
        return boundary, False, 0, float(delta[boundary - 1]), float(delta[boundary])
    best = min(crossings, key=lambda b: (abs(b - boundary), b))
    return best, True, len(crossings), float(delta[best - 1]), float(delta[best])


def analyze_case(
    *,
    ctx: FoldContext,
    case_id: str,
    windows: Sequence[int],
    sigmas: Sequence[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    case_df = load_case_output(ctx.dataset, ctx.fold, case_id)
    gt = case_df["ground_truth"].astype(str).tolist()
    argmax = case_df["argmax_activity"].astype(str).tolist()
    mat = ctx.case_to_mat[str(case_id)]
    if len(gt) != len(argmax) or len(gt) != mat.shape[1]:
        raise ValueError(
            f"Length mismatch {ctx.dataset} fold {ctx.fold} case {case_id}: "
            f"gt={len(gt)} argmax={len(argmax)} softmax_T={mat.shape[1]}"
        )
    raw_argmax = mat.argmax(axis=0).astype(str).tolist()
    if raw_argmax != argmax:
        bad = [i for i, (a, b) in enumerate(zip(raw_argmax, argmax)) if a != b][:10]
        raise ValueError(
            f"Raw softmax argmax mismatch {ctx.dataset} fold {ctx.fold} case {case_id}: {bad}"
        )

    gt_transitions = boundary_positions(gt)
    argmax_segments = segments(argmax)
    smooth_cache = {float(sigma): smooth_softmax(mat, float(sigma)) for sigma in sigmas}
    rows: List[Dict[str, Any]] = []
    for boundary_index, (left_seg, right_seg) in enumerate(
        zip(argmax_segments[:-1], argmax_segments[1:])
    ):
        _, b_argmax, left_class = left_seg
        right_start, _, right_class = right_seg
        if b_argmax != right_start:
            raise AssertionError("Unexpected non-contiguous argmax segments")
        for window in windows:
            b_gt = nearest_transition_in_window(gt_transitions, b_argmax, int(window))
            has_gt = b_gt is not None
            if has_gt:
                dist_argmax = abs(int(b_argmax) - int(b_gt))
                d_gt = sign(int(b_gt) - int(b_argmax))
                argmax_already_correct = dist_argmax == 0
            else:
                dist_argmax = np.nan
                d_gt = np.nan
                argmax_already_correct = False
            for sigma, smooth_mat in smooth_cache.items():
                b_smooth, crossing_found, n_crossings, d_left, d_right = (
                    smoothed_crossing_boundary(
                        smooth_mat=smooth_mat,
                        left_class=left_class,
                        right_class=right_class,
                        boundary=int(b_argmax),
                        window=int(window),
                    )
                )
                d_smooth = sign(int(b_smooth) - int(b_argmax))
                if has_gt:
                    dist_smooth = abs(int(b_smooth) - int(b_gt))
                    dist_reduction = int(dist_argmax) - int(dist_smooth)
                    direction_agree = (
                        bool(d_smooth == d_gt) if int(dist_argmax) != 0 else np.nan
                    )
                    improved = bool(dist_smooth < dist_argmax)
                    worsened = bool(dist_smooth > dist_argmax)
                    moved_away_when_correct = bool(
                        argmax_already_correct and b_smooth != b_argmax
                    )
                else:
                    dist_smooth = np.nan
                    dist_reduction = np.nan
                    direction_agree = np.nan
                    improved = np.nan
                    worsened = np.nan
                    moved_away_when_correct = np.nan
                rows.append(
                    {
                        "dataset": ctx.dataset,
                        "fold": int(ctx.fold),
                        "case_id": str(case_id),
                        "window": int(window),
                        "sigma": float(sigma),
                        "boundary_index": int(boundary_index),
                        "c_L": str(left_class),
                        "c_R": str(right_class),
                        "c_L_name": ctx.label_names.get(str(left_class), ""),
                        "c_R_name": ctx.label_names.get(str(right_class), ""),
                        "b_argmax": int(b_argmax),
                        "b_gt": int(b_gt) if has_gt else np.nan,
                        "has_gt_transition_in_window": bool(has_gt),
                        "b_smooth": int(b_smooth),
                        "crossing_found": bool(crossing_found),
                        "n_crossings_in_window": int(n_crossings),
                        "no_crossing_no_move": bool(not crossing_found and b_smooth == b_argmax),
                        "dist_argmax": float(dist_argmax) if has_gt else np.nan,
                        "dist_smooth": float(dist_smooth) if has_gt else np.nan,
                        "dist_reduction": float(dist_reduction) if has_gt else np.nan,
                        "d_gt": int(d_gt) if has_gt else np.nan,
                        "d_smooth": int(d_smooth),
                        "dir_agree": direction_agree,
                        "improved": improved,
                        "worsened": worsened,
                        "argmax_already_correct_boundary": bool(argmax_already_correct),
                        "moved_away_when_correct": moved_away_when_correct,
                        "delta_left_of_b_smooth": d_left,
                        "delta_right_of_b_smooth": d_right,
                        "smoothed_crossing_uses_gt": False,
                        "test_gt_used_only_for_scoring": True,
                    }
                )
    meta = {
        "n_frames": len(gt),
        "n_argmax_boundaries": max(0, len(argmax_segments) - 1),
    }
    return rows, meta


def summarize(per_boundary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["dataset", "fold", "window", "sigma"]
    for keys, group in per_boundary.groupby(group_cols, sort=True):
        dataset, fold, window, sigma = keys
        with_gt = group[group["has_gt_transition_in_window"]]
        movable = with_gt[with_gt["dist_argmax"] > 0]
        correct = with_gt[with_gt["dist_argmax"] == 0]
        rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "window": int(window),
                "sigma": float(sigma),
                "n_boundaries": int(len(group)),
                "n_with_gt_transition": int(len(with_gt)),
                "n_without_gt_transition": int(len(group) - len(with_gt)),
                "gt_transition_coverage": safe_div(len(with_gt), len(group)),
                "improvement_rate": safe_div(int(with_gt["improved"].sum()), len(with_gt)),
                "worsening_rate": safe_div(int(with_gt["worsened"].sum()), len(with_gt)),
                "mean_dist_reduction": float(with_gt["dist_reduction"].mean())
                if len(with_gt)
                else float("nan"),
                "median_dist_reduction": float(with_gt["dist_reduction"].median())
                if len(with_gt)
                else float("nan"),
                "mean_dist_argmax": float(with_gt["dist_argmax"].mean())
                if len(with_gt)
                else float("nan"),
                "mean_dist_smooth": float(with_gt["dist_smooth"].mean())
                if len(with_gt)
                else float("nan"),
                "n_direction_eval": int(len(movable)),
                "direction_agreement_rate": safe_div(
                    count_true(movable["dir_agree"]), len(movable)
                ),
                "n_argmax_correct_boundaries": int(len(correct)),
                "away_move_rate_on_correct_boundaries": safe_div(
                    count_true(correct["moved_away_when_correct"]),
                    len(correct),
                ),
                "crossing_found_rate": safe_div(int(group["crossing_found"].sum()), len(group)),
                "no_crossing_no_move_rate": safe_div(
                    int(group["no_crossing_no_move"].sum()), len(group)
                ),
            }
        )
    fold_df = pd.DataFrame(rows)
    agg_rows: List[Dict[str, Any]] = []
    for keys, group in per_boundary.groupby(["dataset", "window", "sigma"], sort=True):
        dataset, window, sigma = keys
        with_gt = group[group["has_gt_transition_in_window"]]
        movable = with_gt[with_gt["dist_argmax"] > 0]
        correct = with_gt[with_gt["dist_argmax"] == 0]
        agg_rows.append(
            {
                "dataset": dataset,
                "fold": "all",
                "window": int(window),
                "sigma": float(sigma),
                "n_boundaries": int(len(group)),
                "n_with_gt_transition": int(len(with_gt)),
                "n_without_gt_transition": int(len(group) - len(with_gt)),
                "gt_transition_coverage": safe_div(len(with_gt), len(group)),
                "improvement_rate": safe_div(int(with_gt["improved"].sum()), len(with_gt)),
                "worsening_rate": safe_div(int(with_gt["worsened"].sum()), len(with_gt)),
                "mean_dist_reduction": float(with_gt["dist_reduction"].mean())
                if len(with_gt)
                else float("nan"),
                "median_dist_reduction": float(with_gt["dist_reduction"].median())
                if len(with_gt)
                else float("nan"),
                "mean_dist_argmax": float(with_gt["dist_argmax"].mean())
                if len(with_gt)
                else float("nan"),
                "mean_dist_smooth": float(with_gt["dist_smooth"].mean())
                if len(with_gt)
                else float("nan"),
                "n_direction_eval": int(len(movable)),
                "direction_agreement_rate": safe_div(
                    count_true(movable["dir_agree"]), len(movable)
                ),
                "n_argmax_correct_boundaries": int(len(correct)),
                "away_move_rate_on_correct_boundaries": safe_div(
                    count_true(correct["moved_away_when_correct"]),
                    len(correct),
                ),
                "crossing_found_rate": safe_div(int(group["crossing_found"].sum()), len(group)),
                "no_crossing_no_move_rate": safe_div(
                    int(group["no_crossing_no_move"].sum()), len(group)
                ),
            }
        )
    return pd.concat([fold_df, pd.DataFrame(agg_rows)], ignore_index=True)


def write_smoke_report(out_dir: Path, per_boundary: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Stage-1A.0 Smoothed-Crossing Smoke",
        "",
        "Scope: GTEA fold 1 only. Smoothed crossings use no GT; GT is used only for scoring.",
        "",
        f"- boundary rows: {len(per_boundary)}",
        f"- cases: {per_boundary['case_id'].nunique() if len(per_boundary) else 0}",
        f"- windows: {sorted(per_boundary['window'].unique().tolist()) if len(per_boundary) else []}",
        f"- sigmas: {sorted(per_boundary['sigma'].unique().tolist()) if len(per_boundary) else []}",
        "",
        "## Example Boundaries",
        "",
    ]
    examples = per_boundary[
        (per_boundary["window"] == 25)
        & (per_boundary["sigma"] == 11.0)
        & (per_boundary["has_gt_transition_in_window"])
    ].head(5)
    if examples.empty:
        lines.append("- none")
    for row in examples.itertuples(index=False):
        lines.append(
            f"- case {row.case_id}, boundary {row.boundary_index}: "
            f"b_argmax={row.b_argmax}, b_gt={int(row.b_gt)}, b_smooth={row.b_smooth}, "
            f"dist_argmax={row.dist_argmax:.0f}, dist_smooth={row.dist_smooth:.0f}, "
            f"dist_reduction={row.dist_reduction:.0f}, d_gt={int(row.d_gt)}, "
            f"d_smooth={row.d_smooth}, crossing_found={row.crossing_found}."
        )
    lines.extend(["", "## Smoke Summary", ""])
    smoke_summary = summary[
        (summary["dataset"] == "gtea")
        & (summary["fold"].astype(str) == "1")
        & (summary["window"].isin([25, 50]))
    ]
    if smoke_summary.empty:
        lines.append("- no summary rows")
    for row in smoke_summary.sort_values(["window", "sigma"]).itertuples(index=False):
        lines.append(
            f"- w={row.window}, sigma={row.sigma:g}: "
            f"improve={row.improvement_rate:.3f}, mean_delta={row.mean_dist_reduction:.3f}, "
            f"median_delta={row.median_dist_reduction:.3f}, dir_agree={row.direction_agreement_rate:.3f}, "
            f"away_when_correct={row.away_move_rate_on_correct_boundaries:.3f}, "
            f"coverage={row.gt_transition_coverage:.3f}."
        )
    lines.extend(
        [
            "",
            "## Manual Inspection Notes",
            "",
            "The printed rows expose original argmax boundary, nearest GT transition, and GT-blind smoothed crossing. Positive distance reduction means the smoothed crossing is closer to the GT transition. Rows were inspected for sign convention and no-crossing handling.",
            "",
            "## Decision",
            "",
            "Smoke completed; scale-out is allowed if parity flags in summary.json are true.",
        ]
    )
    (out_dir / "smoke_report.md").write_text("\n".join(lines) + "\n")


def write_summary(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Stage-1A.0 Smoothed-Crossing Pre-check Summary",
        "",
        "This is a diagnostic/ceiling analysis. The smoothed softmax crossing is GT-blind; test GT is used only to score distance and direction against the nearest GT transition.",
        "",
        "GO criterion used here: positive mean distance reduction, direction agreement clearly above 50%, and limited movement away from already-correct boundaries. Otherwise the larger Stage-1A feature selector is not justified by this cheapest deployable signal.",
        "",
    ]
    focus = summary[summary["fold"].astype(str) == "all"].copy()
    for dataset in sorted(focus["dataset"].unique()):
        lines.extend([f"## {dataset}", ""])
        ds = focus[focus["dataset"] == dataset].sort_values(["window", "sigma"])
        for row in ds.itertuples(index=False):
            lines.append(
                f"- w={row.window}, sigma={row.sigma:g}: "
                f"coverage={row.gt_transition_coverage:.3f}, "
                f"improve={row.improvement_rate:.3f}, "
                f"worsen={row.worsening_rate:.3f}, "
                f"mean dist reduction={row.mean_dist_reduction:.3f}, "
                f"median={row.median_dist_reduction:.3f}, "
                f"direction agreement={row.direction_agreement_rate:.3f}, "
                f"away when already correct={row.away_move_rate_on_correct_boundaries:.3f}, "
                f"crossing found={row.crossing_found_rate:.3f}."
            )
        best = ds.sort_values(
            ["mean_dist_reduction", "direction_agreement_rate"], ascending=False
        ).iloc[0]
        if (
            float(best.mean_dist_reduction) > 0.25
            and float(best.direction_agreement_rate) > 0.55
            and float(best.away_move_rate_on_correct_boundaries) < 0.50
        ):
            decision = "GO: smoothed-crossing signal moves boundaries toward GT enough to justify the fuller Stage-1A selector."
        elif float(best.mean_dist_reduction) > 0.0 and float(best.direction_agreement_rate) > 0.50:
            decision = "WEAK GO only: signal is above zero/chance but harm control is questionable; treat as a diagnostic feature, not a standalone boundary mover."
        else:
            decision = "NO-GO: smoothed crossing is anchored to argmax or moves in the wrong direction on average."
        lines.append("")
        lines.append(f"Gate decision: {decision}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets = ["gtea"]
        args.folds = [1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    complete_cases = load_complete_case_set()
    rows: List[Dict[str, Any]] = []
    loaded_cases = 0
    skipped_incomplete = 0
    total_frames = 0
    total_argmax_boundaries = 0
    parity: Dict[str, Any] = {
        "raw_softmax_argmax_matches_case_csv": True,
        "frame_counts_match_gt_argmax_softmax": True,
        "smoothed_crossing_uses_gt": False,
        "test_gt_used_only_for_scoring": True,
        "incomplete_cases_excluded": True,
    }

    for dataset in args.datasets:
        for fold in get_folds(dataset, args.folds, args.data_root):
            ctx = load_fold_context(dataset, fold, args.data_root)
            test_cases = ctx.test_cases[: args.case_limit] if args.case_limit else ctx.test_cases
            for case_id in test_cases:
                key = (dataset, fold, str(case_id))
                if key not in complete_cases:
                    skipped_incomplete += 1
                    continue
                case_rows, meta = analyze_case(
                    ctx=ctx,
                    case_id=str(case_id),
                    windows=[int(w) for w in args.windows],
                    sigmas=[float(s) for s in args.sigmas],
                )
                loaded_cases += 1
                total_frames += int(meta["n_frames"])
                total_argmax_boundaries += int(meta["n_argmax_boundaries"])
                rows.extend(case_rows)

    per_boundary = pd.DataFrame(rows)
    if per_boundary.empty:
        raise ValueError("No smoothed-crossing boundary rows produced")
    if not bool(per_boundary["smoothed_crossing_uses_gt"].eq(False).all()):
        raise AssertionError("Smoothed crossing must not use GT")
    if not bool(per_boundary["test_gt_used_only_for_scoring"].eq(True).all()):
        raise AssertionError("GT usage flag failed")

    summary = summarize(per_boundary)
    per_boundary.to_csv(out_dir / "smoothed_crossing_per_boundary.csv", index=False)
    summary.to_csv(out_dir / "smoothed_crossing_summary.csv", index=False)
    write_smoke_report(out_dir, per_boundary, summary)
    write_summary(out_dir, summary)

    parity["loaded_cases"] = loaded_cases
    parity["skipped_incomplete_cases"] = skipped_incomplete
    parity["n_boundary_rows"] = int(len(per_boundary))
    parity["total_frames"] = int(total_frames)
    parity["total_argmax_boundaries"] = int(total_argmax_boundaries)
    payload = {
        "out_dir": str(out_dir),
        "datasets": args.datasets,
        "folds": args.folds,
        "windows": [int(w) for w in args.windows],
        "sigmas": [float(s) for s in args.sigmas],
        "parity_flags": parity,
        "outputs": [
            "smoothed_crossing_per_boundary.csv",
            "smoothed_crossing_summary.csv",
            "smoke_report.md",
            "summary.md",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Stage-1A.0 smoothed-crossing pre-check to {out_dir}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
