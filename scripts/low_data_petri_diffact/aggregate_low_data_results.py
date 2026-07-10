#!/usr/bin/env python3
"""Aggregate low-data DiffAct + Petri-net ablation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from low_data_common import DEFAULT_EXPERIMENT_DIR, write_csv_header
from study_aggregation import aggregate_study


SUMMARY_BY_SEED_COLS = [
    "dataset",
    "seed",
    "fraction",
    "condition",
    "method",
    "calibrated",
    "metric_name",
    "metric_value",
]

AGG_COLS = [
    "dataset",
    "fraction",
    "condition",
    "method",
    "calibrated",
    "metric_name",
    "mean",
    "std",
    "n_seeds",
    "n_cases",
]

DELTA_COLS = [
    "dataset",
    "seed",
    "fraction",
    "condition",
    "postprocess_method",
    "metric_name",
    "raw_value",
    "post_value",
    "delta",
]

COVERAGE_COLS = [
    "seed",
    "fraction",
    "n_cases",
    "n_variants",
    "variant_coverage",
    "n_activities",
    "activity_coverage",
    "missing_activities",
]

RUNTIME_COLS = [
    "seed",
    "fraction",
    "condition",
    "method",
    "total_seconds",
    "mean_seconds_per_case",
    "median_seconds_per_case",
]

PER_CASE_RUNTIME_COLS = [
    "seed",
    "fraction",
    "condition",
    "method",
    "split",
    "original_case_id",
    "sequential_case_id",
    "n_frames",
    "seconds",
]

RELATIVE_COLS = [
    "dataset",
    "seed",
    "fraction",
    "condition",
    "postprocess_method",
    "metric_name",
    "higher_is_better",
    "raw_value",
    "post_value",
    "absolute_improvement",
    "relative_improvement_pct",
]

PAIRED_COLS = [
    "dataset",
    "seed",
    "fraction",
    "condition",
    "postprocess_method",
    "metric_name",
    "higher_is_better",
    "mean_case_improvement",
    "ci95_low",
    "ci95_high",
    "n_cases",
]

RUN_STATUS_COLS = [
    "dataset",
    "seed",
    "fraction",
    "config_exists",
    "checkpoint_exists",
    "train_log_exists",
    "val_softmax_exists",
    "test_softmax_exists",
    "honest_test_petri_metrics_exists",
    "honest_test_calibrated_petri_metrics_exists",
    "structural_prior_test_petri_metrics_exists",
    "oracle_test_petri_metrics_exists",
    "status",
]

BALANCED_AGGREGATION_COLS = [
    "dataset",
    "condition",
    "method",
    "calibrated",
    "requested_fractions",
    "included_seeds",
    "excluded_partial_seeds",
    "n_included_seeds",
    "n_excluded_partial_seeds",
]

HIGHER_IS_BETTER = {
    "frame_accuracy": True,
    "edit": True,
    "f1@10": True,
    "f1@25": True,
    "f1@50": True,
    "run_collapsed_levenshtein": False,
    "illegal_transition_pct": False,
    "alignment_cost": False,
    "mean_alignment_cost": False,
    "total_alignment_cost": False,
    "mean_transition_penalty": False,
    "total_transition_penalty": False,
}


def load_concat(paths: List[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.is_file()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_metric_files(experiment_dir: Path) -> pd.DataFrame:
    return load_concat(sorted((experiment_dir / "petri").glob("**/metrics.csv")))


def collect_runtime_files(experiment_dir: Path) -> pd.DataFrame:
    return load_concat(sorted((experiment_dir / "petri").glob("**/runtime.csv")))


def collect_per_case_metric_files(experiment_dir: Path) -> pd.DataFrame:
    return load_concat(sorted((experiment_dir / "petri").glob("**/per_case_metrics.csv")))


def collect_per_case_runtime_files(experiment_dir: Path) -> pd.DataFrame:
    return load_concat(sorted((experiment_dir / "petri").glob("**/per_case_runtime.csv")))


def filter_test(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "split" in df.columns:
        return df[df["split"] == "test"].copy()
    return df.copy()


def requested_fractions(metadata: Dict[str, Any], df: pd.DataFrame) -> List[int]:
    values = metadata.get("fractions", [])
    if values:
        return sorted(int(x) for x in values)
    if "fraction" not in df.columns or df.empty:
        return []
    return sorted(int(x) for x in df["fraction"].dropna().unique())


def filter_balanced_seed_sets(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only seeds with every requested fraction for each comparable series.

    While long runs are still in flight, this prevents top-level cross-fraction
    summaries and plots from mixing, for example, 4 seeds at 25% with 3 seeds
    at 50/75/100%. Per-run artifacts remain on disk and are represented in the
    run-status inventory; they enter the top-level summaries once their seed has
    all requested fractions for that method/condition/calibration setting.
    """

    empty_status = pd.DataFrame(columns=BALANCED_AGGREGATION_COLS)
    if df.empty or "seed" not in df.columns or "fraction" not in df.columns:
        return df.copy(), empty_status

    fractions = requested_fractions(metadata, df)
    if not fractions:
        return df.copy(), empty_status
    required = set(fractions)
    group_cols = [
        col
        for col in ["dataset", "condition", "method", "calibrated"]
        if col in df.columns
    ]
    if not group_cols:
        group_cols = ["__all__"]
        df = df.copy()
        df["__all__"] = "all"

    kept: List[pd.DataFrame] = []
    status_rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        seed_to_fractions = (
            group.groupby("seed")["fraction"]
            .apply(lambda values: set(int(x) for x in values.dropna().unique()))
            .to_dict()
        )
        included = sorted(
            int(seed)
            for seed, seen in seed_to_fractions.items()
            if required.issubset(seen)
        )
        partial = sorted(
            int(seed)
            for seed, seen in seed_to_fractions.items()
            if seen and not required.issubset(seen)
        )
        if included:
            kept.append(group[group["seed"].isin(included)])
        status_rows.append(
            {
                "dataset": key_map.get("dataset", ""),
                "condition": key_map.get("condition", ""),
                "method": key_map.get("method", ""),
                "calibrated": key_map.get("calibrated", ""),
                "requested_fractions": " ".join(str(x) for x in fractions),
                "included_seeds": " ".join(str(x) for x in included),
                "excluded_partial_seeds": " ".join(str(x) for x in partial),
                "n_included_seeds": len(included),
                "n_excluded_partial_seeds": len(partial),
            }
        )

    filtered = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0].copy()
    if "__all__" in filtered.columns:
        filtered = filtered.drop(columns=["__all__"])
    status_df = pd.DataFrame(status_rows, columns=BALANCED_AGGREGATION_COLS)
    return filtered, status_df


def improvement_direction(metric_name: str) -> bool:
    return HIGHER_IS_BETTER.get(metric_name, True)


def build_relative_improvement(delta: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if delta.empty:
        return pd.DataFrame(columns=RELATIVE_COLS)
    for _, row in delta.iterrows():
        metric = str(row["metric_name"])
        higher = improvement_direction(metric)
        raw = float(row["raw_value"])
        post = float(row["post_value"])
        absolute_improvement = (post - raw) if higher else (raw - post)
        relative = np.nan
        if abs(raw) > 1e-12:
            relative = 100.0 * absolute_improvement / abs(raw)
        rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "fraction": row["fraction"],
                "condition": row["condition"],
                "postprocess_method": row["postprocess_method"],
                "metric_name": metric,
                "higher_is_better": higher,
                "raw_value": raw,
                "post_value": post,
                "absolute_improvement": absolute_improvement,
                "relative_improvement_pct": relative,
            }
        )
    return pd.DataFrame(rows, columns=RELATIVE_COLS)


def build_paired_case_delta_summary(per_case_df: pd.DataFrame, n_bootstrap: int = 1000) -> pd.DataFrame:
    test_case = filter_test(per_case_df)
    if test_case.empty:
        return pd.DataFrame(columns=PAIRED_COLS)

    raw = test_case[test_case["method"] == "raw_argmax"][
        [
            "dataset",
            "seed",
            "fraction",
            "condition",
            "calibrated",
            "metric_name",
            "original_case_id",
            "metric_value",
        ]
    ].rename(columns={"metric_value": "raw_value"})
    post = test_case[test_case["method"] != "raw_argmax"][
        [
            "dataset",
            "seed",
            "fraction",
            "condition",
            "calibrated",
            "method",
            "metric_name",
            "original_case_id",
            "metric_value",
        ]
    ].rename(columns={"method": "postprocess_method", "metric_value": "post_value"})

    joined = post.merge(
        raw,
        on=["dataset", "seed", "fraction", "condition", "calibrated", "metric_name", "original_case_id"],
        how="inner",
    )
    joined.loc[joined["calibrated"].astype(bool), "postprocess_method"] = (
        joined.loc[joined["calibrated"].astype(bool), "postprocess_method"].astype(str)
        + "_calibrated"
    )
    if joined.empty:
        return pd.DataFrame(columns=PAIRED_COLS)

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(1729)
    group_cols = ["dataset", "seed", "fraction", "condition", "postprocess_method", "metric_name"]
    for keys, group in joined.groupby(group_cols, dropna=False):
        metric = str(keys[-1])
        higher = improvement_direction(metric)
        raw_vals = group["raw_value"].astype(float).to_numpy()
        post_vals = group["post_value"].astype(float).to_numpy()
        improvements = (post_vals - raw_vals) if higher else (raw_vals - post_vals)
        n = len(improvements)
        if n == 0:
            continue
        if n == 1:
            ci_low = ci_high = float(improvements[0])
        else:
            sample_idx = rng.integers(0, n, size=(n_bootstrap, n))
            boot_means = improvements[sample_idx].mean(axis=1)
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5]).tolist()
        rows.append(
            {
                "dataset": keys[0],
                "seed": keys[1],
                "fraction": keys[2],
                "condition": keys[3],
                "postprocess_method": keys[4],
                "metric_name": metric,
                "higher_is_better": higher,
                "mean_case_improvement": float(improvements.mean()),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "n_cases": int(n),
            }
        )
    return pd.DataFrame(rows, columns=PAIRED_COLS)


def write_required_csvs(
    experiment_dir: Path,
    metadata: Dict[str, Any],
    metrics_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    per_case_df: pd.DataFrame,
    per_case_runtime_df: pd.DataFrame,
) -> Dict[str, Path]:
    paths = {
        "results_summary_by_seed": experiment_dir / "results_summary_by_seed.csv",
        "results_summary_aggregated": experiment_dir / "results_summary_aggregated.csv",
        "raw_vs_post_delta": experiment_dir / "raw_vs_post_delta.csv",
        "subset_coverage_summary": experiment_dir / "subset_coverage_summary.csv",
        "runtime_summary": experiment_dir / "runtime_summary.csv",
        "per_case_runtime": experiment_dir / "per_case_runtime.csv",
        "relative_improvement_summary": experiment_dir / "relative_improvement_summary.csv",
        "paired_case_delta_summary": experiment_dir / "paired_case_delta_summary.csv",
        "run_status_inventory": experiment_dir / "run_status_inventory.csv",
        "balanced_aggregation_status": experiment_dir / "balanced_aggregation_status.csv",
    }

    all_test_metrics = filter_test(metrics_df)
    test_metrics, balance_status = filter_balanced_seed_sets(all_test_metrics, metadata)
    if balance_status.empty:
        write_csv_header(paths["balanced_aggregation_status"], BALANCED_AGGREGATION_COLS)
    else:
        balance_status[BALANCED_AGGREGATION_COLS].to_csv(
            paths["balanced_aggregation_status"],
            index=False,
        )
    if test_metrics.empty:
        write_csv_header(paths["results_summary_by_seed"], SUMMARY_BY_SEED_COLS)
        write_csv_header(paths["results_summary_aggregated"], AGG_COLS)
        write_csv_header(paths["raw_vs_post_delta"], DELTA_COLS)
        write_csv_header(paths["relative_improvement_summary"], RELATIVE_COLS)
    else:
        test_metrics[SUMMARY_BY_SEED_COLS].to_csv(paths["results_summary_by_seed"], index=False)
        grouped = (
            test_metrics.groupby(
                ["dataset", "fraction", "condition", "method", "calibrated", "metric_name"],
                dropna=False,
            )
            .agg(
                mean=("metric_value", "mean"),
                std=("metric_value", "std"),
                n_seeds=("seed", "nunique"),
                n_cases=("n_cases", "max"),
            )
            .reset_index()
        )
        grouped[AGG_COLS].to_csv(paths["results_summary_aggregated"], index=False)

        raw = test_metrics[test_metrics["method"] == "raw_argmax"][
            ["dataset", "seed", "fraction", "condition", "calibrated", "metric_name", "metric_value"]
        ].rename(columns={"metric_value": "raw_value"})
        post = test_metrics[test_metrics["method"] != "raw_argmax"][
            ["dataset", "seed", "fraction", "condition", "calibrated", "method", "metric_name", "metric_value"]
        ].rename(columns={"method": "postprocess_method", "metric_value": "post_value"})
        delta = post.merge(
            raw,
            on=["dataset", "seed", "fraction", "condition", "calibrated", "metric_name"],
            how="left",
        )
        delta.loc[delta["calibrated"].astype(bool), "postprocess_method"] = (
            delta.loc[delta["calibrated"].astype(bool), "postprocess_method"].astype(str)
            + "_calibrated"
        )
        delta["delta"] = delta["post_value"] - delta["raw_value"]
        delta[DELTA_COLS].to_csv(paths["raw_vs_post_delta"], index=False)
        build_relative_improvement(delta).to_csv(
            paths["relative_improvement_summary"],
            index=False,
        )

    coverage = pd.read_csv(paths["subset_coverage_summary"]) if paths["subset_coverage_summary"].is_file() else pd.DataFrame()
    if not coverage.empty:
        coverage[[c for c in COVERAGE_COLS if c in coverage.columns]].to_csv(
            paths["subset_coverage_summary"],
            index=False,
        )
    else:
        write_csv_header(paths["subset_coverage_summary"], COVERAGE_COLS)

    test_runtime, _ = filter_balanced_seed_sets(filter_test(runtime_df), metadata)
    if test_runtime.empty:
        write_csv_header(paths["runtime_summary"], RUNTIME_COLS)
    else:
        test_runtime[RUNTIME_COLS].to_csv(paths["runtime_summary"], index=False)

    test_per_case_runtime, _ = filter_balanced_seed_sets(filter_test(per_case_runtime_df), metadata)
    if test_per_case_runtime.empty:
        write_csv_header(paths["per_case_runtime"], PER_CASE_RUNTIME_COLS)
    else:
        test_per_case_runtime[PER_CASE_RUNTIME_COLS].to_csv(
            paths["per_case_runtime"],
            index=False,
        )

    balanced_per_case, _ = filter_balanced_seed_sets(filter_test(per_case_df), metadata)
    paired = build_paired_case_delta_summary(balanced_per_case)
    if paired.empty:
        write_csv_header(paths["paired_case_delta_summary"], PAIRED_COLS)
    else:
        paired[PAIRED_COLS].to_csv(paths["paired_case_delta_summary"], index=False)

    return paths


def completed_checkpoint_exists(run_dir: Path, naming: str, config_path: Path) -> bool:
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
        num_epochs = int(config["num_epochs"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False

    complete_path = run_dir / "train_complete.json"
    if complete_path.is_file():
        try:
            payload = json.loads(complete_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        checkpoint = Path(str(payload.get("final_checkpoint", "")))
        if checkpoint.is_file() and int(payload.get("num_epochs", -1)) == num_epochs:
            return True

    return (run_dir / "training" / naming / f"epoch-{num_epochs - 1}.model").is_file()


def softmax_bundle_exists(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "video_index_map.txt").is_file()
        and any(path.glob("[0-9]*.npy"))
    )


def petri_metrics_exists(
    experiment_dir: Path,
    condition: str,
    seed: int,
    fraction: int,
    split_dir: str,
    *,
    petri_fraction: int | None = None,
    oracle: bool = False,
) -> bool:
    if oracle:
        roots = [
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"diffact_frac_{fraction}"
            / "petri_source_test"
        ]
    else:
        petri_frac = fraction if petri_fraction is None else petri_fraction
        roots = [
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"diffact_frac_{fraction}"
            / f"petri_frac_{petri_frac}",
            experiment_dir / "petri" / condition / f"seed_{seed}" / f"frac_{fraction}",
        ]
    return any(
        path.is_file()
        for root in roots
        for path in root.glob(f"**/{split_dir}/metrics.csv")
    )


def build_run_status_inventory(experiment_dir: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    dataset = metadata.get("dataset", "")
    fold = metadata.get("fold", "")
    seeds = [int(x) for x in metadata.get("seeds", [])]
    fractions = [int(x) for x in metadata.get("fractions", [])]
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for fraction in fractions:
            run_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}"
            config_path = run_dir / "config.json"
            naming = f"lowdata_{dataset}_fold{fold}_seed{seed}_frac{fraction}"
            config_exists = config_path.is_file()
            checkpoint_ok = completed_checkpoint_exists(run_dir, naming, config_path)
            train_log_ok = (run_dir / "train.log").is_file()
            val_softmax_ok = softmax_bundle_exists(run_dir / "softmax_val")
            test_softmax_ok = softmax_bundle_exists(run_dir / "softmax_test")
            honest_metrics_ok = petri_metrics_exists(
                experiment_dir,
                "honest_low_data_petri",
                seed,
                fraction,
                "test",
            )
            honest_calibrated_metrics_ok = petri_metrics_exists(
                experiment_dir,
                "honest_low_data_petri",
                seed,
                fraction,
                "test_calibrated",
            )
            structural_metrics_ok = (
                True
                if fraction >= 100
                else petri_metrics_exists(
                    experiment_dir,
                    "structural_prior_full_train_petri",
                    seed,
                    fraction,
                    "test",
                    petri_fraction=100,
                )
            )
            oracle_metrics_ok = petri_metrics_exists(
                experiment_dir,
                "oracle_test_fold_petri",
                seed,
                fraction,
                "test",
                oracle=True,
            )
            if honest_metrics_ok:
                status = "complete_honest_test"
            elif test_softmax_ok:
                status = "needs_petri_postprocessing"
            elif checkpoint_ok:
                status = "needs_softmax_export"
            elif config_exists:
                status = "needs_training"
            else:
                status = "missing_config"
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fraction": fraction,
                    "config_exists": config_exists,
                    "checkpoint_exists": checkpoint_ok,
                    "train_log_exists": train_log_ok,
                    "val_softmax_exists": val_softmax_ok,
                    "test_softmax_exists": test_softmax_ok,
                    "honest_test_petri_metrics_exists": honest_metrics_ok,
                    "honest_test_calibrated_petri_metrics_exists": honest_calibrated_metrics_ok,
                    "structural_prior_test_petri_metrics_exists": structural_metrics_ok,
                    "oracle_test_petri_metrics_exists": oracle_metrics_ok,
                    "status": status,
                }
            )
    return pd.DataFrame(rows, columns=RUN_STATUS_COLS)


def make_plots(
    experiment_dir: Path,
    metadata: Dict[str, Any],
    metrics_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    per_case_df: pd.DataFrame,
) -> List[Path]:
    plots: List[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: matplotlib unavailable ({exc})")
        return plots

    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    test_metrics, _ = filter_balanced_seed_sets(filter_test(metrics_df), metadata)

    for metric in ["frame_accuracy", "edit", "f1@10", "f1@25", "f1@50"]:
        sub = test_metrics[test_metrics["metric_name"] == metric] if not test_metrics.empty else pd.DataFrame()
        if sub.empty:
            continue
        agg = sub.groupby(["fraction", "method"], as_index=False)["metric_value"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, g in agg.groupby("method"):
            g = g.sort_values("fraction")
            ax.plot(g["fraction"], g["metric_value"], marker="o", label=method)
        ax.set_xlabel("Training cases (%)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs training fraction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = plot_dir / f"metric_{metric.replace('@', '').replace('/', '_')}_vs_fraction.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(path)

    delta_path = experiment_dir / "raw_vs_post_delta.csv"
    if delta_path.is_file():
        delta = pd.read_csv(delta_path)
        delta = delta[delta["metric_name"].isin(["frame_accuracy", "edit", "f1@10", "f1@25", "f1@50"])]
        if not delta.empty:
            agg = delta.groupby(["fraction", "postprocess_method", "metric_name"], as_index=False)["delta"].mean()
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for (method, metric), g in agg.groupby(["postprocess_method", "metric_name"]):
                g = g.sort_values("fraction")
                ax.plot(g["fraction"], g["delta"], marker="o", label=f"{method} {metric}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Training cases (%)")
            ax.set_ylabel("Post - raw")
            ax.set_title("Delta improvement vs training fraction")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            path = plot_dir / "delta_improvement_vs_fraction.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            plots.append(path)

    coverage_path = experiment_dir / "subset_coverage_summary.csv"
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path)
        if not coverage.empty:
            agg = coverage.groupby("fraction", as_index=False)[["variant_coverage", "activity_coverage"]].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(agg["fraction"], agg["variant_coverage"], marker="o", label="variant coverage")
            ax.plot(agg["fraction"], agg["activity_coverage"], marker="o", label="activity coverage")
            ax.set_xlabel("Training cases (%)")
            ax.set_ylabel("Coverage")
            ax.set_ylim(0, 1.05)
            ax.set_title("Activity and variant coverage")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = plot_dir / "coverage_vs_fraction.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            plots.append(path)

    test_runtime, _ = filter_balanced_seed_sets(filter_test(runtime_df), metadata)
    if not test_runtime.empty:
        agg = test_runtime.groupby(["fraction", "method"], as_index=False)["mean_seconds_per_case"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, g in agg.groupby("method"):
            g = g.sort_values("fraction")
            ax.plot(g["fraction"], g["mean_seconds_per_case"], marker="o", label=method)
        ax.set_xlabel("Training cases (%)")
        ax.set_ylabel("Mean seconds per case")
        ax.set_title("Postprocessing runtime")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = plot_dir / "runtime_vs_fraction.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(path)

    test_case, _ = filter_balanced_seed_sets(filter_test(per_case_df), metadata)
    if not test_case.empty:
        raw = test_case[
            (test_case["method"] == "raw_argmax")
            & (test_case["metric_name"] == "frame_accuracy")
        ][["seed", "fraction", "condition", "original_case_id", "metric_value"]].rename(
            columns={"metric_value": "raw_accuracy"}
        )
        post = test_case[
            (test_case["method"] != "raw_argmax")
            & (test_case["metric_name"] == "frame_accuracy")
        ][["seed", "fraction", "condition", "method", "original_case_id", "metric_value"]].rename(
            columns={"metric_value": "post_accuracy"}
        )
        joined = post.merge(
            raw,
            on=["seed", "fraction", "condition", "original_case_id"],
            how="inner",
        )
        if not joined.empty:
            fig, ax = plt.subplots(figsize=(5, 5))
            for (fraction, method), g in joined.groupby(["fraction", "method"]):
                ax.scatter(
                    g["raw_accuracy"],
                    g["post_accuracy"],
                    label=f"{fraction}% {method}",
                    alpha=0.75,
                )
            lim_low = min(joined["raw_accuracy"].min(), joined["post_accuracy"].min(), 0)
            lim_high = max(joined["raw_accuracy"].max(), joined["post_accuracy"].max(), 100)
            ax.plot([lim_low, lim_high], [lim_low, lim_high], color="black", linewidth=0.8)
            ax.set_xlabel("Raw per-case frame accuracy")
            ax.set_ylabel("Petri per-case frame accuracy")
            ax.set_title("Per-case raw vs Petri accuracy")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            path = plot_dir / "per_case_raw_vs_petri_accuracy.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            plots.append(path)

    return plots


def conclusion_from_delta(delta_path: Path) -> str:
    if not delta_path.is_file():
        return "No test deltas are available yet, so the hypothesis has not been tested."
    delta = pd.read_csv(delta_path)
    if delta.empty:
        return "No test deltas are available yet, so the hypothesis has not been tested."
    focus = delta[
        delta["metric_name"].isin(["frame_accuracy", "edit", "f1@10", "f1@25", "f1@50"])
    ]
    if focus.empty:
        return "No Petri postprocessing test deltas are available yet."
    pivot = focus.groupby(
        ["fraction", "postprocess_method", "metric_name"],
        as_index=False,
    )["delta"].mean()
    best = pivot.loc[pivot["delta"].idxmax()]
    worst = pivot.loc[pivot["delta"].idxmin()]
    return (
        "Current aggregated test deltas show the largest mean Petri postprocessing gain at "
        f"{int(best['fraction'])}% using {best['postprocess_method']} on "
        f"{best['metric_name']} ({best['delta']:.3f}) and the largest mean loss at "
        f"{int(worst['fraction'])}% using {worst['postprocess_method']} on "
        f"{worst['metric_name']} ({worst['delta']:.3f}). Interpret this only for "
        "completed seeds/runs."
    )


def write_report(
    experiment_dir: Path,
    paths: Dict[str, Path],
    plots: List[Path],
    metrics_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    per_case_df: pd.DataFrame,
    per_case_runtime_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> None:
    metadata_path = experiment_dir / "experiment_metadata.json"
    metadata: Dict[str, Any] = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    completed_test = filter_test(metrics_df)
    balanced_completed_test, _ = filter_balanced_seed_sets(completed_test, metadata)
    completed_runs = 0
    if not balanced_completed_test.empty:
        completed_runs = balanced_completed_test[["seed", "fraction", "condition"]].drop_duplicates().shape[0]

    plot_lines = "\n".join(f"- {p.relative_to(experiment_dir)}" for p in plots) if plots else "- No metric plots yet; no completed test metrics were found."
    if status_df.empty:
        status_lines = "- Run status inventory is empty."
    else:
        counts = status_df["status"].value_counts().sort_index()
        status_lines = "\n".join(f"- {status}: {count}" for status, count in counts.items())
    balance_note = (
        "Top-level summary CSVs and plots use only balanced seed sets: a seed enters "
        "a method/condition/calibration series only after all requested fractions are "
        "complete for that same series. Partial in-flight seed/fraction outputs remain "
        "on disk and in `run_status_inventory.csv`, but are held out of cross-fraction "
        "tables until balanced."
    )
    balance_status_path = paths.get("balanced_aggregation_status")
    if balance_status_path is not None and balance_status_path.is_file():
        balance_status = pd.read_csv(balance_status_path)
        if not balance_status.empty:
            def display_seed_list(value: Any) -> str:
                if pd.isna(value):
                    return "none"
                rendered = str(value).strip()
                return rendered if rendered else "none"

            balance_lines = []
            for _, row in balance_status.drop_duplicates(
                ["condition", "method", "calibrated", "included_seeds", "excluded_partial_seeds"]
            ).iterrows():
                included = display_seed_list(row.get("included_seeds", ""))
                excluded = display_seed_list(row.get("excluded_partial_seeds", ""))
                balance_lines.append(
                    "- "
                    f"{row.get('condition', '')}/{row.get('method', '')}/"
                    f"calibrated={row.get('calibrated', '')}: "
                    f"included seeds `{included}`, held-out partial seeds `{excluded}`"
                )
            balance_note += "\n\n" + "\n".join(balance_lines)
    validation_hp = (
        "No hyperparameter sweep is run in this pilot. The postprocessing method and "
        "transition-illegal penalty are fixed by command line; validation outputs are "
        "used for validation-only calibration/selection metadata."
    )
    val_metrics = metrics_df[metrics_df.get("split", "") == "val"] if not metrics_df.empty and "split" in metrics_df.columns else pd.DataFrame()
    if not val_metrics.empty:
        validation_hp = (
            "Validation metrics exist under `petri/*/val/`. Test postprocessing writes "
            "`selected_hyperparameters.json` from validation-only metrics before running test. "
            "This pilot does not run a broad hyperparameter sweep; the method and "
            "transition-illegal penalty are fixed, while temperature calibration is fit "
            "on validation only when calibrated test outputs are requested."
        )
    per_case_note = (
        f"Per-case paired metric rows available: {len(per_case_df)}."
        if not per_case_df.empty
        else "No per-case paired metrics are available yet."
    )
    runtime_note = (
        f"Runtime rows available: {len(runtime_df)}."
        if not runtime_df.empty
        else "No runtime rows are available yet."
    )
    per_case_runtime_note = (
        f"Per-case runtime rows available: {len(per_case_runtime_df)}."
        if not per_case_runtime_df.empty
        else "No per-case runtime rows are available yet."
    )
    conformance_note_path = experiment_dir / "CONFORMANCE_RUNTIME_NOTE.md"
    conformance_note = (
        "See `CONFORMANCE_RUNTIME_NOTE.md` for the exact-conformance runtime check "
        "that motivated the scalable `petri_transition_viterbi` pilot method."
        if conformance_note_path.is_file()
        else "No separate exact-conformance runtime note is available."
    )
    report = (
        "# Low-Data DiffAct + Petri-Net Ablation\n\n"
        "## Goal And Hypothesis\n\n"
        "Test whether Petri-net/SKTR postprocessing contributes more when DiffAct is trained "
        "from scratch on fewer training cases/videos/traces: 100%, 75%, 50%, and 25%. "
        "The hypothesis remains unsupported until the fixed-test metrics show it.\n\n"
        "## Pilot Scope Caveat\n\n"
        "This directory contains a 101-epoch pilot/throughput run. The inspected 50salads "
        "baseline configuration uses 5001 epochs, so these pilot metrics must not be "
        "mixed with final baseline-epoch DiffAct results. The prior-baseline match is "
        "not treated as verified unless baseline metrics are regenerated or otherwise "
        "located in a complete comparable artifact.\n\n"
        "## Dataset And Split\n\n"
        f"- Dataset: {metadata.get('dataset', 'unknown')}\n"
        f"- Fold: {metadata.get('fold', 'unknown')}\n"
        f"- Data root: `{metadata.get('data_root', '')}`\n"
        f"- DiffAct repo: `{metadata.get('diffact_repo_path', '')}`\n"
        f"- SKTR code: `{metadata.get('postprocessing_code_path', '')}`\n"
        f"- Baseline config: `{metadata.get('baseline_config_path', '')}`\n"
        f"- Previous checkpoint: `{metadata.get('previous_checkpoint_path', '')}`\n"
        f"- Previous result path: `{metadata.get('previous_result_path', '')}`\n"
        f"- Previous softmax output dirs: `{metadata.get('previous_softmax_output_dirs', [])}`\n"
        f"- Split manifest: `{metadata.get('generated_split_manifest_path', '')}`\n"
        f"- Train pool cases: `{metadata.get('generated_train_pool_cases_path', '')}`\n"
        f"- Validation cases: `{metadata.get('generated_val_cases_path', '')}`\n"
        f"- Test cases: `{metadata.get('generated_test_cases_path', '')}`\n"
        f"- Validation alignment dir: `{metadata.get('validation_alignment_dir', '')}`\n"
        f"- Test alignment dir: `{metadata.get('test_alignment_dir', '')}`\n"
        f"- Fixed train pool / validation / test cases: {metadata.get('n_train_pool', 0)} / "
        f"{metadata.get('n_val', 0)} / {metadata.get('n_test', 0)}\n\n"
        "Validation was carved once from the official training split at the case/video level "
        "because no official validation split was recorded in the inspected DiffAct setup. "
        "This validation carve may share person identity with train-pool videos, so it can be "
        "optimistic; it is used only for validation/calibration, while the fixed official "
        "test split remains held out. "
        "The fixed test split is never used for training, Petri-net discovery, calibration, "
        "hyperparameter tuning, or model selection.\n\n"
        "## Training Subsets\n\n"
        "Subsets are nested per seed, sampled at the trace/case level, and stratified by "
        "run-collapsed activity variants. Main experiment traces are not shortened.\n\n"
        "## Commands\n\n"
        f"- Reproduction script: `{experiment_dir / 'commands_to_reproduce.sh'}`\n"
        f"- Pipeline command log: `{experiment_dir / 'pipeline_commands.sh'}`\n"
        f"- Train commands: `{experiment_dir / 'diffact' / 'train_commands.sh'}`\n"
        f"- Inference commands: `{experiment_dir / 'diffact' / 'infer_commands.sh'}`\n"
        f"- Petri commands: `{experiment_dir / 'petri'}`\n\n"
        "## Output Tables\n\n"
        + "\n".join(f"- {name}: `{path}`" for name, path in paths.items())
        + "\n\n"
        "Additional tables beyond the required five provide relative improvements, "
        "bootstrapped paired per-case deltas, per-case runtime rows, and run completion status.\n\n"
        "## Balanced Aggregation Guard\n\n"
        f"{balance_note}\n\n"
        "## Figures\n\n"
        f"{plot_lines}\n\n"
        "## Validation-Tuned Hyperparameters\n\n"
        f"{validation_hp}\n\n"
        "## Current Test Coverage\n\n"
        f"Balanced seed/fraction/condition test runs included in top-level summaries: {completed_runs}\n\n"
        f"{status_lines}\n\n"
        f"{per_case_note}\n\n"
        f"{runtime_note}\n\n"
        f"{per_case_runtime_note}\n\n"
        "## Exact-Conformance Runtime Check\n\n"
        f"{conformance_note}\n\n"
        "## Answer\n\n"
        f"{conclusion_from_delta(paths['raw_vs_post_delta'])}\n\n"
        "## Limitations And Incomplete Parts\n\n"
        "- Beam search and hybrid postprocessing are not run by this scaffold. Exact alignment conformance is available as `petri_conformance`, but the pilot logs show it is runtime-impractical on full-frame 50salads traces with the current implementation; the default scalable method is `petri_transition_viterbi`.\n"
        "- Validation-temperature calibration is available for test postprocessing once validation softmax outputs exist; it fits on validation only and writes `calibration.json`.\n"
        "- Raw logits are not exported by the current DiffAct exporter; available artifacts are softmax probabilities and raw argmax predictions.\n"
        "- Top-level cross-fraction summaries intentionally exclude partial in-flight seeds until every requested fraction is complete for the relevant condition/method/calibration series.\n"
        "- Any DiffAct training/inference not present under `diffact/seed_*/frac_*/` is not completed.\n"
        "- Exact per-case runtime rows are written when SKTR case outputs include `postprocess_seconds`; otherwise median per-case runtime is left blank instead of inferred from the mean.\n"
        "- Per-case bootstrap confidence intervals are generated once per-case metrics exist; seed-level significance is still pending until enough seeds complete.\n\n"
        "## Next Recommended Experiment\n\n"
        "Let the current five-seed 101-epoch pilot finish, then inspect balanced paired deltas with confidence intervals. "
        "Before treating the result as final, run at least one low-data fraction at the full 5001-epoch baseline schedule "
        "to check whether the observed Petri effect survives a converged DiffAct model.\n"
    )
    (experiment_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=None,
        help="Aggregate every experiment listed in study_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Study-wide output directory (default: <study-dir>/aggregation or experiment dir).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    if args.study_dir is not None:
        study_dir = args.study_dir.resolve()
        study_metadata_path = study_dir / "study_metadata.json"
        if not study_metadata_path.is_file():
            raise FileNotFoundError(f"Missing study metadata: {study_metadata_path}")
        study_metadata = json.loads(study_metadata_path.read_text())
        experiment_dirs = [
            Path(item["experiment_dir"]).resolve()
            for item in study_metadata.get("experiments", [])
        ]
        if not experiment_dirs:
            raise ValueError(f"No experiments listed in {study_metadata_path}")
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else study_dir / "aggregation"
        )
        study_paths = aggregate_study(
            experiment_dirs,
            output_dir,
            n_bootstrap=args.bootstrap_replicates,
        )
        print(f"Study experiments: {len(experiment_dirs)}")
        for name, path in study_paths.items():
            print(f"{name}: {path}")
        return

    experiment_dir = args.experiment_dir.resolve()
    metadata_path = experiment_dir / "experiment_metadata.json"
    metadata: Dict[str, Any] = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    metrics_df = collect_metric_files(experiment_dir)
    runtime_df = collect_runtime_files(experiment_dir)
    per_case_df = collect_per_case_metric_files(experiment_dir)
    per_case_runtime_df = collect_per_case_runtime_files(experiment_dir)
    paths = write_required_csvs(
        experiment_dir,
        metadata,
        metrics_df,
        runtime_df,
        per_case_df,
        per_case_runtime_df,
    )
    status_df = build_run_status_inventory(experiment_dir, metadata)
    status_df.to_csv(paths["run_status_inventory"], index=False)
    plots = make_plots(experiment_dir, metadata, metrics_df, runtime_df, per_case_df)
    write_report(
        experiment_dir,
        paths,
        plots,
        metrics_df,
        runtime_df,
        per_case_df,
        per_case_runtime_df,
        status_df,
    )

    print(f"Aggregated metrics rows: {len(metrics_df)}")
    print(f"Aggregated runtime rows: {len(runtime_df)}")
    print(f"Aggregated per-case metric rows: {len(per_case_df)}")
    print(f"Aggregated per-case runtime rows: {len(per_case_runtime_df)}")
    print(f"Report: {experiment_dir / 'REPORT.md'}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    if plots:
        print("Plots:")
        for path in plots:
            print(f"  {path}")
    study_output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else experiment_dir
    )
    study_paths = aggregate_study(
        [experiment_dir],
        study_output_dir,
        n_bootstrap=args.bootstrap_replicates,
    )
    for name, path in study_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
