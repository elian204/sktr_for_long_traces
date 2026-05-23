#!/usr/bin/env python3
"""
Stage-3B1 seed-selection feasibility.

This script is intentionally fail-closed. Stage-3B1 requires per-video,
per-seed predictions/probabilities from the K=10 Gate-0.5 run. The current
Gate-0.5 artifacts only persist aggregate seed metrics plus aggregate
oracle/combiner summaries, so the requested per-video rank-stability and
unsupervised seed selectors cannot be computed without rerunning inference.

To honor the "reuse Gate-0.5 outputs, no new inference" guardrail, this script
writes the partial aggregate diagnostics that are supportable and marks the
load-bearing Phase-1/Phase-2 analyses as blocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


DEFAULT_RUN_DIR = Path("/data1/eli-bogdanov/sktr_runs/stage3b_diffusion_ensemble_v1")
METRICS = ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]
REQUESTED_SELECTORS = [
    "max_mean_pmax",
    "min_entropy",
    "closest_to_majority",
    "closest_to_mean_softmax",
    "fewest_short_segments",
    "best_duration_likelihood",
]


def metric_delta(row: pd.Series, base: pd.Series, metric: str) -> float:
    return float(row[metric] - base[metric])


def retained_fraction(selected_delta: float, oracle_delta: float) -> float:
    if abs(oracle_delta) < 1e-12:
        return float("nan")
    return float(selected_delta / oracle_delta)


def bootstrap_best_minus_random(seed_values: np.ndarray, n_boot: int = 20000) -> Dict[str, float]:
    """Aggregate-only null: expected best-minus-random seed within the seed pool."""
    rng = np.random.default_rng(12345)
    vals = np.asarray(seed_values, dtype=float)
    if vals.size == 0:
        return {"null_mean": float("nan"), "null_q05": float("nan"), "null_q95": float("nan")}
    draws = []
    for _ in range(n_boot):
        random_default = vals[rng.integers(0, vals.size)]
        draws.append(float(vals.max() - random_default))
    arr = np.asarray(draws)
    return {
        "null_mean": float(arr.mean()),
        "null_q05": float(np.quantile(arr, 0.05)),
        "null_q95": float(np.quantile(arr, 0.95)),
    }


def build_oracle_validity(seed_spread: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in sorted(seed_spread["dataset"].dropna().unique()):
        ds = seed_spread[seed_spread["dataset"] == dataset]
        base = ds[ds["row_type"] == "current_export_baseline"]
        seeds = ds[ds["row_type"] == "single_global_seed"].copy()
        if base.empty or seeds.empty:
            continue
        base_row = base.iloc[0]
        same_metric_best = seeds.sort_values("F1@50", ascending=False).iloc[0]
        f10_selected = seeds.sort_values("F1@10", ascending=False).iloc[0]

        ens_ds = ensemble[(ensemble["dataset"] == dataset) & (ensemble["K"] == 10)]
        best_seq = ens_ds[ens_ds["method"] == "oracle_best_seed_per_video_f1@50"]
        best_seq_delta = float(best_seq["delta_F1@50"].iloc[0]) if not best_seq.empty else float("nan")

        tau_f10 = kendalltau(seeds["F1@50"], seeds["F1@10"]).statistic
        tau_edit = kendalltau(seeds["F1@50"], seeds["Edit"]).statistic
        tau_acc = kendalltau(seeds["F1@50"], seeds["Acc"]).statistic
        null = bootstrap_best_minus_random(seeds["F1@50"].to_numpy())

        same_delta = metric_delta(same_metric_best, base_row, "F1@50")
        cross_delta = metric_delta(f10_selected, base_row, "F1@50")
        rows.append(
            {
                "dataset": dataset,
                "scope": "aggregate_seed_metrics_only",
                "phase1_valid": False,
                "blocked_reason": (
                    "Gate-0.5 did not persist per-video/per-seed predictions or probabilities; "
                    "per-video rank stability, unsupervised selectors, case 1/49 rows, and "
                    "fold-pure learned selection cannot be computed without new inference."
                ),
                "n_global_seeds": int(len(seeds)),
                "default_Acc": float(base_row["Acc"]),
                "default_Edit": float(base_row["Edit"]),
                "default_F1@10": float(base_row["F1@10"]),
                "default_F1@25": float(base_row["F1@25"]),
                "default_F1@50": float(base_row["F1@50"]),
                "same_metric_best_seed_by_aggregate_F1@50": same_metric_best["seed"],
                "same_metric_best_delta_F1@50_vs_default": same_delta,
                "cross_metric_seed_by_aggregate_F1@10": f10_selected["seed"],
                "cross_metric_selected_delta_F1@50_vs_default": cross_delta,
                "cross_metric_fraction_of_same_metric_best_retained": retained_fraction(cross_delta, same_delta),
                "best_seed_per_video_oracle_delta_F1@50_from_gate05": best_seq_delta,
                "kendall_tau_aggregate_seed_rank_F1@50_vs_F1@10": float(tau_f10),
                "kendall_tau_aggregate_seed_rank_F1@50_vs_Edit": float(tau_edit),
                "kendall_tau_aggregate_seed_rank_F1@50_vs_Acc": float(tau_acc),
                "aggregate_noise_null_best_minus_random_F1@50_mean": null["null_mean"],
                "aggregate_noise_null_best_minus_random_F1@50_q05": null["null_q05"],
                "aggregate_noise_null_best_minus_random_F1@50_q95": null["null_q95"],
                "note": "These are aggregate seed diagnostics only, not the requested per-video Phase-1 gate.",
            }
        )
    return pd.DataFrame(rows)


def build_unsupervised_selector_rows(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in sorted(ensemble["dataset"].dropna().unique()):
        ens = ensemble[(ensemble["dataset"] == dataset) & (ensemble["K"] == 10)]
        for method in ["mean_softmax_argmax", "majority_vote"]:
            match = ens[ens["method"] == method]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row.update(
                {
                    "selector": method,
                    "selector_type": "combiner_from_gate05_not_seed_selector",
                    "phase1_valid_seed_selector": False,
                    "blocked_reason": "Available from Gate-0.5 aggregate combiner outputs, but not one of the requested seed selectors.",
                }
            )
            rows.append(row)

        for selector in REQUESTED_SELECTORS:
            rows.append(
                {
                    "dataset": dataset,
                    "K": 10,
                    "selector": selector,
                    "selector_type": "requested_unsupervised_seed_selector",
                    "phase1_valid_seed_selector": False,
                    "blocked_reason": "Requires per-video/per-seed predictions/probabilities/features not persisted by Gate-0.5.",
                    "Acc": np.nan,
                    "Edit": np.nan,
                    "F1@10": np.nan,
                    "F1@25": np.nan,
                    "F1@50": np.nan,
                    "delta_Acc": np.nan,
                    "delta_Edit": np.nan,
                    "delta_F1@10": np.nan,
                    "delta_F1@25": np.nan,
                    "delta_F1@50": np.nan,
                    "worst_case_harm": np.nan,
                    "fraction_cross_validated_oracle_gain_captured": np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_blocked_features(run_dir: Path, datasets: List[str]) -> pd.DataFrame:
    rows = []
    feature_groups = [
        "mean_pmax",
        "entropy",
        "distance_to_majority",
        "distance_to_mean_softmax",
        "short_segment_count",
        "duration_likelihood",
    ]
    for dataset in datasets:
        for feature in feature_groups:
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "status": "blocked_missing_per_seed_artifacts",
                    "reason": "Feature requires per-video/per-seed probabilities or predictions.",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "stage3b1_features.csv", index=False)
    return df


def write_learned_eval_placeholder(run_dir: Path, datasets: List[str]) -> pd.DataFrame:
    rows = [
        {
            "dataset": dataset,
            "phase2_status": "skipped",
            "reason": "Phase 1 is not valid from persisted artifacts; learned selector would require per-video/per-seed features and labels.",
        }
        for dataset in datasets
    ]
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "stage3b1_learned_eval.csv", index=False)
    return df


def write_summary(
    run_dir: Path,
    oracle_validity: pd.DataFrame,
    unsupervised: pd.DataFrame,
) -> None:
    lines = [
        "# Stage-3B1 Seed-Selection Feasibility",
        "",
        "Status: **BLOCKED / fail-closed**.",
        "",
        "I did not run new DiffAct inference. The existing Gate-0.5 artifacts do not contain the per-video, per-seed predictions/probabilities needed for the requested Phase-1 seed-selection analysis.",
        "",
        "## What Is Available",
        "",
        "- Aggregate single-seed metrics by dataset and seed from `stage3b_gate05_seed_spread.csv`.",
        "- Aggregate mean-softmax/majority/oracle-combiner metrics from `stage3b_gate05_ensemble_ceiling.csv`.",
        "- Per-video diversity summaries, but not the actual per-seed predictions/probabilities.",
        "",
        "## What Cannot Be Computed Without New Inference Or A Persisted Seed Cache",
        "",
        "- Per-video seed rankings and per-video Kendall tau rank stability.",
        "- Cross-validated oracle seed choice per video.",
        "- Requested unsupervised seed selectors: max-mean-pmax, min-entropy, closest-to-majority, closest-to-mean-softmax, fewest-short-segments, best-duration-likelihood.",
        "- 50Salads case 1 and case 49 selected-seed diagnostics.",
        "- Fold-pure learned selector / Phase 2.",
        "",
        "## Partial Aggregate Diagnostics",
        "",
        "| Dataset | Aggregate best-seed ΔF1@50 | Aggregate F1@10-selected ΔF1@50 | Retained Fraction | Aggregate tau F1@50 vs F1@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in oracle_validity.iterrows():
        lines.append(
            "| {dataset} | {same:+.3f} | {cross:+.3f} | {frac:+.3f} | {tau:+.3f} |".format(
                dataset=row["dataset"],
                same=row["same_metric_best_delta_F1@50_vs_default"],
                cross=row["cross_metric_selected_delta_F1@50_vs_default"],
                frac=row["cross_metric_fraction_of_same_metric_best_retained"],
                tau=row["kendall_tau_aggregate_seed_rank_F1@50_vs_F1@10"],
            )
        )

    lines.extend(
        [
            "",
            "These aggregate rows are not a substitute for the requested Phase-1 gate, because the gate is explicitly per-video/fold-pure.",
            "",
            "## Decision",
            "",
            "Do not proceed to Phase 2 from these artifacts. To run Stage-3B1 properly, rerun Gate-0.5 with a persisted seed cache containing, per case and seed, predictions plus either probabilities or selector features. That rerun would be new inference and needs explicit approval because the current instruction said not to run new inference.",
        ]
    )
    (run_dir / "stage3b1_summary.md").write_text("\n".join(lines) + "\n")

    summary = {
        "status": "blocked_missing_per_video_per_seed_artifacts",
        "new_inference_run": False,
        "phase1_valid": False,
        "phase2_run": False,
        "datasets": sorted(oracle_validity["dataset"].tolist()),
        "outputs": {
            "oracle_validity": "stage3b1_oracle_validity.csv",
            "unsupervised_selectors": "stage3b1_unsupervised_selectors.csv",
            "features": "stage3b1_features.csv",
            "learned_eval": "stage3b1_learned_eval.csv",
            "summary": "stage3b1_summary.md",
        },
    }
    (run_dir / "stage3b1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3B1 seed-selection feasibility over persisted Gate-0.5 artifacts.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    seed_spread_path = args.run_dir / "stage3b_gate05_seed_spread.csv"
    ensemble_path = args.run_dir / "stage3b_gate05_ensemble_ceiling.csv"
    if not seed_spread_path.exists() or not ensemble_path.exists():
        raise FileNotFoundError("Missing Gate-0.5 aggregate artifacts.")

    seed_spread = pd.read_csv(seed_spread_path)
    ensemble = pd.read_csv(ensemble_path)
    oracle_validity = build_oracle_validity(seed_spread, ensemble)
    oracle_validity.to_csv(args.run_dir / "stage3b1_oracle_validity.csv", index=False)

    unsupervised = build_unsupervised_selector_rows(ensemble)
    unsupervised.to_csv(args.run_dir / "stage3b1_unsupervised_selectors.csv", index=False)

    datasets = sorted(oracle_validity["dataset"].tolist())
    write_blocked_features(args.run_dir, datasets)
    write_learned_eval_placeholder(args.run_dir, datasets)
    write_summary(args.run_dir, oracle_validity, unsupervised)

    print(f"Wrote blocked Stage-3B1 report to {args.run_dir}")
    print("status=blocked_missing_per_video_per_seed_artifacts; new_inference_run=False")


if __name__ == "__main__":
    main()
