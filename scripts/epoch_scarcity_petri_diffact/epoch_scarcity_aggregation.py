#!/usr/bin/env python3
"""Aggregate epoch-scarcity per-case learning curves and paired deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from epoch_scarcity_common import ORACLE_CONDITION, PRIMARY_CONDITION, read_json


CORE_METRICS = ("f1@25", "f1@10", "f1@50", "edit", "frame_accuracy")
INDEX_COLS = [
    "dataset",
    "official_fold",
    "trajectory_seed",
    "checkpoint_epoch_index",
    "completed_epochs",
    "schedule_fraction_pct",
    "training_item_presentations",
    "trainer_step",
    "optimizer_updates",
    "condition",
    "petri_discovery_source",
    "is_oracle_condition",
    "original_case_id",
    "metric",
]
METHOD_ROW_COLS = INDEX_COLS + [
    "method",
    "method_detail",
    "metric_value",
    "n_frames",
    "source_file",
]
DELTA_COLS = INDEX_COLS + [
    "method",
    "method_detail",
    "comparator",
    "comparator_detail",
    "method_value",
    "comparator_value",
    "delta",
    "is_primary_endpoint",
    "n_frames",
]
GLOBAL_INDEX_COLS = [column for column in INDEX_COLS if column != "original_case_id"]
GLOBAL_METHOD_ROW_COLS = GLOBAL_INDEX_COLS + [
    "method",
    "method_detail",
    "metric_value",
    "n_cases",
    "source_file",
]
GLOBAL_DELTA_COLS = GLOBAL_INDEX_COLS + [
    "method",
    "method_detail",
    "comparator",
    "comparator_detail",
    "method_value",
    "comparator_value",
    "delta",
    "is_primary_endpoint",
    "is_secondary_fold_global_endpoint",
    "n_cases",
]


def boolean(value: Any) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def canonical_method(value: Any) -> tuple[str, str]:
    detail = str(value).strip().lower()
    if detail in {"raw", "raw_argmax", "raw_decoder_argmax", "decoder_argmax"} or (
        "raw" in detail and "argmax" in detail
    ):
        return "raw_decoder_argmax", detail
    if detail in {
        "official",
        "official_diffact",
        "official_diffact_output",
        "official_diffact_postprocessed",
        "diffact_postprocessed",
    } or ("official" in detail and "diffact" in detail):
        return "official_diffact", detail
    if detail.startswith("petri_") or detail.startswith("sktr"):
        return "sktr", detail
    return detail, detail


def per_case_paths(study_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in (study_dir / "experiments").glob("**/per_case_metrics.csv")
        if path.is_file()
    )


def global_metric_paths(study_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in (study_dir / "experiments").glob("**/metrics.csv")
        if path.is_file()
    )


def normalize_per_case(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for row in source.to_dict("records"):
        if str(row.get("split", "test")).lower() != "test":
            continue
        method, detail = canonical_method(row.get("method", ""))
        metric = str(row.get("metric_name", row.get("metric", ""))).lower()
        if method not in {"raw_decoder_argmax", "official_diffact", "sktr"} or metric not in CORE_METRICS:
            continue
        condition = str(row["condition"])
        is_oracle = boolean(row.get("is_oracle_condition", row.get("oracle_upper_bound", False)))
        if condition == ORACLE_CONDITION:
            is_oracle = True
        rows.append(
            {
                "dataset": str(row["dataset"]).lower(),
                "official_fold": int(row["official_fold"]),
                "trajectory_seed": int(row.get("trajectory_seed", row.get("seed", 0))),
                "checkpoint_epoch_index": int(row["checkpoint_epoch_index"]),
                "completed_epochs": int(row["completed_epochs"]),
                "schedule_fraction_pct": float(row["schedule_fraction_pct"]),
                "training_item_presentations": int(row["training_item_presentations"]),
                "trainer_step": int(row["trainer_step"]),
                "optimizer_updates": int(row["optimizer_updates"]),
                "condition": condition,
                "petri_discovery_source": str(row["petri_discovery_source"]),
                "is_oracle_condition": is_oracle,
                "original_case_id": str(row["original_case_id"]),
                "metric": metric,
                "method": method,
                "method_detail": detail,
                "metric_value": float(row.get("metric_value", row.get("value"))),
                "n_frames": int(row["n_frames"]),
                "source_file": str(path),
            }
        )
    return pd.DataFrame(rows, columns=METHOD_ROW_COLS)


def load_per_case(study_dir: Path) -> pd.DataFrame:
    frames = [normalize_per_case(path) for path in per_case_paths(study_dir)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=METHOD_ROW_COLS)
    result = pd.concat(frames, ignore_index=True)
    keys = INDEX_COLS + ["method", "method_detail"]
    for key, group in result.groupby(keys, dropna=False):
        values = group["metric_value"].astype(float)
        if len(values) > 1 and float(values.max() - values.min()) > 1e-8:
            raise ValueError(f"Conflicting duplicate per-case rows for {key}: {values.tolist()}")
    return (
        result.groupby(keys, dropna=False, as_index=False)
        .agg(
            metric_value=("metric_value", "mean"),
            n_frames=("n_frames", "max"),
            source_file=("source_file", lambda values: ";".join(sorted(set(map(str, values))))),
        )[METHOD_ROW_COLS]
        .sort_values(INDEX_COLS + ["method", "method_detail"])
        .reset_index(drop=True)
    )


def normalize_global_metrics(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for row in source.to_dict("records"):
        if str(row.get("split", "test")).lower() != "test":
            continue
        method, detail = canonical_method(row.get("method", ""))
        metric = str(row.get("metric_name", row.get("metric", ""))).lower()
        if method not in {"raw_decoder_argmax", "official_diffact", "sktr"} or metric not in CORE_METRICS:
            continue
        condition = str(row["condition"])
        is_oracle = boolean(row.get("is_oracle_condition", row.get("oracle_upper_bound", False)))
        if condition == ORACLE_CONDITION:
            is_oracle = True
        rows.append(
            {
                "dataset": str(row["dataset"]).lower(),
                "official_fold": int(row["official_fold"]),
                "trajectory_seed": int(row.get("trajectory_seed", row.get("seed", 0))),
                "checkpoint_epoch_index": int(row["checkpoint_epoch_index"]),
                "completed_epochs": int(row["completed_epochs"]),
                "schedule_fraction_pct": float(row["schedule_fraction_pct"]),
                "training_item_presentations": int(row["training_item_presentations"]),
                "trainer_step": int(row["trainer_step"]),
                "optimizer_updates": int(row["optimizer_updates"]),
                "condition": condition,
                "petri_discovery_source": str(row["petri_discovery_source"]),
                "is_oracle_condition": is_oracle,
                "metric": metric,
                "method": method,
                "method_detail": detail,
                "metric_value": float(row.get("metric_value", row.get("value"))),
                "n_cases": int(row["n_cases"]),
                "source_file": str(path),
            }
        )
    return pd.DataFrame(rows, columns=GLOBAL_METHOD_ROW_COLS)


def load_global_metrics(study_dir: Path) -> pd.DataFrame:
    frames = [normalize_global_metrics(path) for path in global_metric_paths(study_dir)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=GLOBAL_METHOD_ROW_COLS)
    result = pd.concat(frames, ignore_index=True)
    keys = GLOBAL_INDEX_COLS + ["method", "method_detail"]
    for key, group in result.groupby(keys, dropna=False):
        values = group["metric_value"].astype(float)
        if len(values) > 1 and float(values.max() - values.min()) > 1e-8:
            raise ValueError(f"Conflicting duplicate fold-global rows for {key}: {values.tolist()}")
    return (
        result.groupby(keys, dropna=False, as_index=False)
        .agg(
            metric_value=("metric_value", "mean"),
            n_cases=("n_cases", "max"),
            source_file=("source_file", lambda values: ";".join(sorted(set(map(str, values))))),
        )[GLOBAL_METHOD_ROW_COLS]
        .sort_values(GLOBAL_INDEX_COLS + ["method", "method_detail"])
        .reset_index(drop=True)
    )


def build_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(columns=DELTA_COLS)
    match_cols = INDEX_COLS
    indexed = metrics.set_index(match_cols + ["method"], drop=False)
    for sktr in metrics[metrics["method"] == "sktr"].to_dict("records"):
        base_key = tuple(sktr[column] for column in match_cols)
        for comparator in ("official_diffact", "raw_decoder_argmax"):
            key = (*base_key, comparator)
            try:
                candidates = indexed.loc[key]
            except KeyError:
                continue
            if isinstance(candidates, pd.Series):
                candidates = candidates.to_frame().T
            values = candidates["metric_value"].astype(float)
            if float(values.max() - values.min()) > 1e-8:
                raise ValueError(f"Ambiguous comparator for {key}: {values.tolist()}")
            baseline = candidates.sort_values("method_detail").iloc[0]
            delta = float(sktr["metric_value"] - float(baseline["metric_value"]))
            rows.append(
                {
                    **{column: sktr[column] for column in INDEX_COLS},
                    "method": "sktr",
                    "method_detail": sktr["method_detail"],
                    "comparator": comparator,
                    "comparator_detail": baseline["method_detail"],
                    "method_value": sktr["metric_value"],
                    "comparator_value": float(baseline["metric_value"]),
                    "delta": delta,
                    "is_primary_endpoint": (
                        comparator == "official_diffact"
                        and sktr["metric"] == "f1@25"
                        and sktr["condition"] == PRIMARY_CONDITION
                        and not boolean(sktr["is_oracle_condition"])
                    ),
                    "n_frames": sktr["n_frames"],
                }
            )
    return pd.DataFrame(rows, columns=DELTA_COLS).sort_values(
        INDEX_COLS + ["comparator"]
    ).reset_index(drop=True)


def build_global_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(columns=GLOBAL_DELTA_COLS)
    indexed = metrics.set_index(GLOBAL_INDEX_COLS + ["method"], drop=False)
    for sktr in metrics[metrics["method"] == "sktr"].to_dict("records"):
        base_key = tuple(sktr[column] for column in GLOBAL_INDEX_COLS)
        for comparator in ("official_diffact", "raw_decoder_argmax"):
            key = (*base_key, comparator)
            try:
                candidates = indexed.loc[key]
            except KeyError:
                continue
            if isinstance(candidates, pd.Series):
                candidates = candidates.to_frame().T
            values = candidates["metric_value"].astype(float)
            if float(values.max() - values.min()) > 1e-8:
                raise ValueError(f"Ambiguous fold-global comparator for {key}: {values.tolist()}")
            baseline = candidates.sort_values("method_detail").iloc[0]
            rows.append(
                {
                    **{column: sktr[column] for column in GLOBAL_INDEX_COLS},
                    "method": "sktr",
                    "method_detail": sktr["method_detail"],
                    "comparator": comparator,
                    "comparator_detail": baseline["method_detail"],
                    "method_value": sktr["metric_value"],
                    "comparator_value": float(baseline["metric_value"]),
                    "delta": float(sktr["metric_value"] - float(baseline["metric_value"])),
                    "is_primary_endpoint": False,
                    "is_secondary_fold_global_endpoint": (
                        comparator == "official_diffact"
                        and sktr["metric"] == "f1@25"
                        and sktr["condition"] == PRIMARY_CONDITION
                        and not boolean(sktr["is_oracle_condition"])
                    ),
                    "n_cases": sktr["n_cases"],
                }
            )
    return pd.DataFrame(rows, columns=GLOBAL_DELTA_COLS).sort_values(
        GLOBAL_INDEX_COLS + ["comparator"]
    ).reset_index(drop=True)


def bootstrap_interval(
    fold_estimates: pd.DataFrame,
    *,
    value_col: str,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float]:
    values = fold_estimates[value_col].astype(float).to_numpy()
    if len(values) < 2 or n_bootstrap < 1:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    samples = np.mean(
        rng.choice(values, size=(n_bootstrap, len(values)), replace=True),
        axis=1,
    )
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def summarize_curve(
    frame: pd.DataFrame,
    *,
    value_col: str,
    method_columns: Sequence[str],
    n_bootstrap: int,
    case_column: str | None = "original_case_id",
    paired_unit: str = "test_case_nested_within_official_fold",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_cols = [
        "dataset",
        "trajectory_seed",
        "checkpoint_epoch_index",
        "completed_epochs",
        "schedule_fraction_pct",
        "training_item_presentations",
        "trainer_step",
        "optimizer_updates",
        "condition",
        "petri_discovery_source",
        "is_oracle_condition",
        "metric",
        *method_columns,
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        fold_estimates = (
            group.groupby("official_fold", as_index=False)[value_col]
            .mean()
            .sort_values("official_fold")
        )
        values = fold_estimates[value_col].astype(float)
        digest = hashlib.sha256(repr(keys).encode("utf-8")).hexdigest()
        low, high = bootstrap_interval(
            fold_estimates,
            value_col=value_col,
            n_bootstrap=n_bootstrap,
            random_seed=int(digest[:8], 16),
        )
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "mean": float(values.mean()),
                "std_across_fold_estimates": (
                    float(values.std(ddof=1)) if len(values) > 1 else np.nan
                ),
                "ci95_low": low,
                "ci95_high": high,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_folds": int(len(fold_estimates)),
                "n_cases": (
                    int(group[case_column].nunique())
                    if case_column is not None
                    else int(group.groupby("official_fold")["n_cases"].max().sum())
                ),
                "paired_unit": paired_unit,
                "ci_kind": (
                    "descriptive_fold_cluster_bootstrap" if len(fold_estimates) > 1 else "unavailable"
                ),
                "robust_seed_inference_available": False,
                "inference_note": (
                    "Seed0-only first wave; uncertainty is descriptive across official folds."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def aggregate_study(
    study_dir: Path,
    *,
    n_bootstrap: int = 2000,
) -> Dict[str, Path]:
    study_dir = study_dir.resolve()
    metadata = read_json(study_dir / "study_metadata.json")
    output_dir = study_dir / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_per_case(study_dir)
    deltas = build_deltas(metrics)
    global_metrics = load_global_metrics(study_dir)
    global_deltas = build_global_deltas(global_metrics)
    method_curve = summarize_curve(
        metrics,
        value_col="metric_value",
        method_columns=("method", "method_detail"),
        n_bootstrap=n_bootstrap,
    )
    delta_curve = summarize_curve(
        deltas,
        value_col="delta",
        method_columns=("method", "method_detail", "comparator", "comparator_detail"),
        n_bootstrap=n_bootstrap,
    )
    global_method_curve = summarize_curve(
        global_metrics,
        value_col="metric_value",
        method_columns=("method", "method_detail"),
        n_bootstrap=n_bootstrap,
        case_column=None,
        paired_unit="official_fold_global_tas",
    )
    global_delta_curve = summarize_curve(
        global_deltas,
        value_col="delta",
        method_columns=("method", "method_detail", "comparator", "comparator_detail"),
        n_bootstrap=n_bootstrap,
        case_column=None,
        paired_unit="official_fold_global_tas",
    )
    primary = (
        delta_curve[
            (delta_curve["condition"] == PRIMARY_CONDITION)
            & (~delta_curve["is_oracle_condition"].map(boolean))
            & (delta_curve["metric"] == "f1@25")
            & (delta_curve["comparator"] == "official_diffact")
        ].copy()
        if not delta_curve.empty
        else delta_curve
    )
    oracle = (
        delta_curve[
            (delta_curve["condition"] == ORACLE_CONDITION)
            & (delta_curve["is_oracle_condition"].map(boolean))
        ].copy()
        if not delta_curve.empty
        else delta_curve
    )
    global_primary = (
        global_delta_curve[
            (global_delta_curve["condition"] == PRIMARY_CONDITION)
            & (~global_delta_curve["is_oracle_condition"].map(boolean))
            & (global_delta_curve["metric"] == "f1@25")
            & (global_delta_curve["comparator"] == "official_diffact")
        ].copy()
        if not global_delta_curve.empty
        else global_delta_curve
    )
    global_oracle = (
        global_delta_curve[
            (global_delta_curve["condition"] == ORACLE_CONDITION)
            & (global_delta_curve["is_oracle_condition"].map(boolean))
        ].copy()
        if not global_delta_curve.empty
        else global_delta_curve
    )
    paths = {
        "per_case_metrics": output_dir / "epoch_per_case_metrics.csv",
        "paired_case_deltas": output_dir / "epoch_paired_case_deltas.csv",
        "method_curve": output_dir / "epoch_method_curves.csv",
        "delta_curve": output_dir / "epoch_delta_curves.csv",
        "primary_curve": output_dir / "epoch_primary_curve.csv",
        "oracle_curve": output_dir / "epoch_oracle_curve.csv",
        "fold_global_metrics": output_dir / "epoch_fold_global_metrics.csv",
        "fold_global_deltas": output_dir / "epoch_fold_global_deltas.csv",
        "fold_global_method_curve": output_dir / "epoch_fold_global_method_curves.csv",
        "fold_global_delta_curve": output_dir / "epoch_fold_global_delta_curves.csv",
        "fold_global_primary_curve": output_dir / "epoch_fold_global_primary_curve.csv",
        "fold_global_oracle_curve": output_dir / "epoch_fold_global_oracle_curve.csv",
        "metadata": output_dir / "epoch_aggregation_metadata.json",
    }
    metrics.to_csv(paths["per_case_metrics"], index=False)
    deltas.to_csv(paths["paired_case_deltas"], index=False)
    method_curve.to_csv(paths["method_curve"], index=False)
    delta_curve.to_csv(paths["delta_curve"], index=False)
    primary.to_csv(paths["primary_curve"], index=False)
    oracle.to_csv(paths["oracle_curve"], index=False)
    global_metrics.to_csv(paths["fold_global_metrics"], index=False)
    global_deltas.to_csv(paths["fold_global_deltas"], index=False)
    global_method_curve.to_csv(paths["fold_global_method_curve"], index=False)
    global_delta_curve.to_csv(paths["fold_global_delta_curve"], index=False)
    global_primary.to_csv(paths["fold_global_primary_curve"], index=False)
    global_oracle.to_csv(paths["fold_global_oracle_curve"], index=False)
    payload = {
        "schema_version": 1,
        "study_id": metadata["study_id"],
        "study_type": metadata["study_type"],
        "per_case_metric_rows": len(metrics),
        "paired_case_delta_rows": len(deltas),
        "method_curve_rows": len(method_curve),
        "delta_curve_rows": len(delta_curve),
        "primary_curve_rows": len(primary),
        "oracle_curve_rows": len(oracle),
        "fold_global_metric_rows": len(global_metrics),
        "fold_global_delta_rows": len(global_deltas),
        "fold_global_method_curve_rows": len(global_method_curve),
        "fold_global_delta_curve_rows": len(global_delta_curve),
        "fold_global_primary_curve_rows": len(global_primary),
        "fold_global_oracle_curve_rows": len(global_oracle),
        "checkpoint_grid": metadata["checkpoint_grid"],
        "core_metrics": list(CORE_METRICS),
        "primary_endpoint": {
            "method": "sktr",
            "comparator": "official_diffact",
            "metric": "f1@25",
            "condition": PRIMARY_CONDITION,
            "delta_definition": "per-case SKTR value - per-case official DiffAct value",
            "paired_unit": "test_case_nested_within_official_fold",
            "excludes_oracle": True,
        },
        "secondary_fold_global_endpoint": {
            "method": "sktr",
            "comparator": "official_diffact",
            "metric": "f1@25",
            "condition": PRIMARY_CONDITION,
            "delta_definition": "fold-global SKTR value - fold-global official DiffAct value",
            "paired_unit": "official_fold_global_tas",
            "excludes_oracle": True,
            "role": (
                "Secondary absolute/comparability analysis using the conventional fold-global "
                "TAS calculation; it does not replace the pre-specified per-case primary endpoint."
            ),
        },
        "oracle_policy": (
            "Oracle test-fold Petri curves are written separately and never enter the primary curve."
        ),
        "uncertainty_policy": (
            "Per-case values are averaged within each official fold, then fold estimates are "
            "cluster-bootstrapped. Seed0-only intervals are descriptive and do not support "
            "robust seed inference."
        ),
        "bootstrap_replicates": n_bootstrap,
    }
    paths["metadata"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.n_bootstrap < 0:
        raise ValueError("--n-bootstrap must be non-negative")
    paths = aggregate_study(args.study_dir, n_bootstrap=args.n_bootstrap)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
