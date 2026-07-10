#!/usr/bin/env python3
"""Study-wide, fold/seed-aware aggregation for low-data decoding metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


CORE_METRICS = ("f1@25", "f1@10", "f1@50", "edit", "frame_accuracy")
HONEST_CONDITION = "honest_low_data_petri"
ORACLE_CONDITION = "oracle_test_fold_petri"
KEY_COLS = [
    "dataset",
    "official_fold",
    "subset_seed",
    "diffact_fraction",
    "petri_fraction",
    "condition",
    "petri_discovery_source",
    "is_oracle_condition",
    "method",
    "comparator",
    "metric",
]
METRIC_ROW_COLS = KEY_COLS + [
    "method_detail",
    "metric_value",
    "n_cases",
    "split",
    "artifact_scope",
    "experiment_dir",
    "source_file",
]
DELTA_COLS = KEY_COLS + [
    "method_detail",
    "comparator_detail",
    "method_value",
    "comparator_value",
    "delta",
    "is_primary_endpoint",
    "n_cases",
    "artifact_scope",
    "experiment_dir",
]
SUMMARY_COLS = [
    "dataset",
    "diffact_fraction",
    "petri_fraction",
    "condition",
    "petri_discovery_source",
    "is_oracle_condition",
    "method",
    "comparator",
    "metric",
    "method_detail",
    "mean",
    "std_across_fold_seed_estimates",
    "ci95_low",
    "ci95_high",
    "min",
    "max",
    "n_folds",
    "n_seeds",
    "n_fold_seed_estimates",
    "uncertainty_unit",
    "ci_kind",
    "robust_seed_inference_available",
    "inference_note",
]


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return value
    return default


def _integer(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {field}: {value!r}") from exc


def canonical_method(value: Any) -> tuple[str, str]:
    detail = str(value).strip().lower()
    if detail in {
        "raw",
        "raw_argmax",
        "raw_decoder_argmax",
        "decoder_argmax",
        "pre_postprocess_argmax",
    } or ("raw" in detail and "argmax" in detail):
        return "raw_decoder_argmax", detail
    if detail in {
        "official",
        "official_diffact",
        "official_diffact_output",
        "official_diffact_postprocessed",
        "diffact_postprocessed",
        "official_postprocessed",
    } or ("official" in detail and "diffact" in detail):
        return "official_diffact", detail
    if detail.startswith("petri_") or detail.startswith("sktr"):
        return "sktr", detail
    return detail, detail


def _optional_integer(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan", "<na>"}:
        return pd.NA
    return _integer(value, "petri_fraction")


def _petri_fraction(row: Mapping[str, Any], condition: str, diffact_fraction: int) -> Any:
    if _is_oracle_condition(row, condition):
        return pd.NA
    explicit = _first(row, ("petri_fraction", "petri_data_fraction"))
    if explicit is not None:
        return _optional_integer(explicit)
    if condition in {"structural_prior_full_train_petri", "diffact_f_petri_100"}:
        return 100
    return diffact_fraction


def _petri_discovery_source(row: Mapping[str, Any], condition: str) -> str:
    explicit = _first(row, ("petri_discovery_source",))
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if condition == HONEST_CONDITION:
        return "nested_train_fraction"
    if condition == "structural_prior_full_train_petri":
        return "full_train"
    if condition == ORACLE_CONDITION:
        return "official_test_fold"
    if condition == "full_diffact_low_data_petri":
        return "nested_train_fraction"
    return "unknown"


def _is_oracle_condition(row: Mapping[str, Any], condition: str) -> bool:
    if condition == ORACLE_CONDITION:
        return True
    flag = _first(row, ("is_oracle_condition", "oracle_upper_bound"), False)
    if isinstance(flag, str):
        return flag.strip().lower() in {"1", "true", "yes"}
    return bool(flag)


def _artifact_scope(metadata: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    complete = row.get("complete_evaluation_set")
    if complete is not None and not pd.isna(complete):
        if str(complete).strip().lower() in {"false", "0", "no"}:
            return "runtime_pilot"
    value = _first(
        row,
        ("artifact_scope", "run_scope", "run_type"),
        metadata.get("run_type", "final"),
    )
    value = str(value).strip().lower()
    return "runtime_pilot" if "pilot" in value else "final"


def normalize_metrics(
    frame: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    experiment_dir: Path,
    source_file: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source_row in frame.to_dict("records"):
        split = str(_first(source_row, ("split",), "test")).lower()
        if split != "test":
            continue
        dataset = str(_first(source_row, ("dataset",), metadata.get("dataset", ""))).lower()
        fold = _integer(
            _first(source_row, ("official_fold", "fold"), metadata.get("official_fold", metadata.get("fold"))),
            "official_fold",
        )
        seed = _integer(
            _first(source_row, ("subset_seed", "seed"), metadata.get("subset_seed")),
            "subset_seed",
        )
        diffact_fraction = _integer(
            _first(source_row, ("diffact_fraction", "fraction"), metadata.get("diffact_fraction")),
            "diffact_fraction",
        )
        condition = str(_first(source_row, ("condition",), "primary_same_fraction"))
        method, method_detail = canonical_method(_first(source_row, ("method", "decoder"), ""))
        if not method:
            continue
        metric = str(_first(source_row, ("metric", "metric_name"), "")).lower()
        value = _first(source_row, ("metric_value", "value"))
        if not metric or value is None:
            continue
        if bool(_first(source_row, ("calibrated",), False)):
            method_detail += "_calibrated"
        is_oracle = _is_oracle_condition(source_row, condition)
        rows.append(
            {
                "dataset": dataset,
                "official_fold": fold,
                "subset_seed": seed,
                "diffact_fraction": diffact_fraction,
                "petri_fraction": _petri_fraction(source_row, condition, diffact_fraction),
                "condition": condition,
                "petri_discovery_source": _petri_discovery_source(source_row, condition),
                "is_oracle_condition": is_oracle,
                "method": method,
                "comparator": "none",
                "metric": metric,
                "method_detail": method_detail,
                "metric_value": float(value),
                "n_cases": _first(source_row, ("n_cases",), np.nan),
                "split": split,
                "artifact_scope": _artifact_scope(metadata, source_row),
                "experiment_dir": str(experiment_dir),
                "source_file": str(source_file),
            }
        )
    return pd.DataFrame(rows, columns=METRIC_ROW_COLS)


def metric_paths(experiment_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for path in experiment_dir.glob("**/*metrics.csv"):
        name = path.name.lower()
        if (
            path.is_file()
            and "per_case" not in name
            and "summary" not in name
            and not name.startswith("study_")
        ):
            paths.append(path)
    return sorted(set(paths))


def load_study_metrics(experiment_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for experiment_dir in experiment_dirs:
        experiment_dir = experiment_dir.resolve()
        metadata_path = experiment_dir / "experiment_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        for path in metric_paths(experiment_dir):
            try:
                source = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            frames.append(
                normalize_metrics(
                    source,
                    metadata=metadata,
                    experiment_dir=experiment_dir,
                    source_file=path,
                )
            )
    if not frames:
        return pd.DataFrame(columns=METRIC_ROW_COLS)
    result = pd.concat(frames, ignore_index=True)
    duplicate_keys = [
        "dataset",
        "official_fold",
        "subset_seed",
        "diffact_fraction",
        "petri_fraction",
        "condition",
        "petri_discovery_source",
        "is_oracle_condition",
        "method",
        "metric",
        "method_detail",
        "artifact_scope",
    ]
    for keys, group in result.groupby(duplicate_keys, dropna=False):
        values = group["metric_value"].astype(float)
        if len(values) > 1 and float(values.max() - values.min()) > 1e-8:
            raise ValueError(f"Conflicting duplicate metric rows for {keys}: {values.tolist()}")
    result = (
        result.groupby(duplicate_keys, dropna=False, as_index=False)
        .agg(
            metric_value=("metric_value", "mean"),
            n_cases=("n_cases", "max"),
            split=("split", "first"),
            experiment_dir=("experiment_dir", "first"),
            source_file=("source_file", lambda values: ";".join(sorted(set(map(str, values))))),
        )
    )
    result["comparator"] = "none"
    return result[METRIC_ROW_COLS].sort_values(KEY_COLS + ["method_detail"]).reset_index(drop=True)


def _matching_comparator(sktr_row: Mapping[str, Any], metrics: pd.DataFrame, comparator: str) -> pd.Series | None:
    candidates = metrics[
        (metrics["dataset"] == sktr_row["dataset"])
        & (metrics["official_fold"] == sktr_row["official_fold"])
        & (metrics["subset_seed"] == sktr_row["subset_seed"])
        & (metrics["diffact_fraction"] == sktr_row["diffact_fraction"])
        & (metrics["metric"] == sktr_row["metric"])
        & (metrics["method"] == comparator)
        & (metrics["artifact_scope"] == sktr_row["artifact_scope"])
    ].copy()
    if candidates.empty:
        return None
    candidates["_preference"] = (
        (candidates["condition"] == sktr_row["condition"]).astype(int) * 2
        + (candidates["petri_fraction"] == sktr_row["petri_fraction"]).astype(int)
    )
    best = candidates[candidates["_preference"] == candidates["_preference"].max()]
    if float(best["metric_value"].max() - best["metric_value"].min()) > 1e-8:
        raise ValueError(
            "Ambiguous comparator values for "
            f"{sktr_row['dataset']} fold={sktr_row['official_fold']} "
            f"seed={sktr_row['subset_seed']} fraction={sktr_row['diffact_fraction']} "
            f"metric={sktr_row['metric']} comparator={comparator}"
        )
    return best.sort_values(["condition", "method_detail"]).iloc[0]


def build_paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(columns=DELTA_COLS)
    for sktr_row in metrics[metrics["method"] == "sktr"].to_dict("records"):
        for comparator in ("official_diffact", "raw_decoder_argmax"):
            baseline = _matching_comparator(sktr_row, metrics, comparator)
            if baseline is None:
                continue
            rows.append(
                {
                    "dataset": sktr_row["dataset"],
                    "official_fold": sktr_row["official_fold"],
                    "subset_seed": sktr_row["subset_seed"],
                    "diffact_fraction": sktr_row["diffact_fraction"],
                    "petri_fraction": sktr_row["petri_fraction"],
                    "condition": sktr_row["condition"],
                    "petri_discovery_source": sktr_row["petri_discovery_source"],
                    "is_oracle_condition": bool(sktr_row["is_oracle_condition"]),
                    "method": "sktr",
                    "comparator": comparator,
                    "metric": sktr_row["metric"],
                    "method_detail": sktr_row["method_detail"],
                    "comparator_detail": baseline["method_detail"],
                    "method_value": sktr_row["metric_value"],
                    "comparator_value": float(baseline["metric_value"]),
                    "delta": float(sktr_row["metric_value"] - baseline["metric_value"]),
                    "is_primary_endpoint": (
                        comparator == "official_diffact"
                        and sktr_row["metric"] == "f1@25"
                        and sktr_row["condition"] == HONEST_CONDITION
                        and not bool(sktr_row["is_oracle_condition"])
                    ),
                    "n_cases": sktr_row["n_cases"],
                    "artifact_scope": sktr_row["artifact_scope"],
                    "experiment_dir": sktr_row["experiment_dir"],
                }
            )
    return pd.DataFrame(rows, columns=DELTA_COLS).sort_values(KEY_COLS + ["method_detail"]).reset_index(drop=True)


def _hierarchical_interval(
    clusters: pd.DataFrame,
    value_col: str,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float]:
    folds = sorted(clusters["official_fold"].unique())
    if len(clusters) < 2 or n_bootstrap < 1:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_folds = rng.choice(folds, size=len(folds), replace=True)
        values: List[float] = []
        for fold in sampled_folds:
            fold_values = clusters.loc[
                clusters["official_fold"] == fold, value_col
            ].astype(float).to_numpy()
            sampled = rng.choice(fold_values, size=len(fold_values), replace=True)
            values.extend(sampled.tolist())
        samples[index] = float(np.mean(values))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def summarize_estimates(
    frame: pd.DataFrame,
    *,
    value_col: str,
    n_bootstrap: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)
    group_cols = [
        "dataset",
        "diffact_fraction",
        "petri_fraction",
        "condition",
        "petri_discovery_source",
        "is_oracle_condition",
        "method",
        "comparator",
        "metric",
        "method_detail",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        clusters = (
            group.groupby(["official_fold", "subset_seed"], as_index=False)[value_col]
            .mean()
            .sort_values(["official_fold", "subset_seed"])
        )
        values = clusters[value_col].astype(float)
        n_folds = int(clusters["official_fold"].nunique())
        n_seeds = int(clusters["subset_seed"].nunique())
        digest = hashlib.sha256(repr(keys).encode("utf-8")).hexdigest()
        low, high = _hierarchical_interval(
            clusters,
            value_col,
            n_bootstrap=n_bootstrap,
            random_seed=int(digest[:8], 16),
        )
        if n_seeds < 2:
            robust = False
            ci_kind = "descriptive_fold_cluster_bootstrap" if n_folds > 1 else "unavailable"
            note = (
                "Seed0-only pilot: descriptive aggregation across official folds; "
                "robust seed inference is unavailable."
            )
            uncertainty_unit = "official_fold"
        elif n_folds < 2:
            robust = False
            ci_kind = "descriptive_seed_bootstrap"
            note = "Only one official fold is available; cross-fold inference is unavailable."
            uncertainty_unit = "subset_seed"
        else:
            robust = True
            ci_kind = "hierarchical_fold_seed_bootstrap"
            note = "Fold/seed-aware hierarchical bootstrap; frames are not independent units."
            uncertainty_unit = "official_fold_and_subset_seed"
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "mean": float(values.mean()),
                "std_across_fold_seed_estimates": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_folds": n_folds,
                "n_seeds": n_seeds,
                "n_fold_seed_estimates": int(len(clusters)),
                "uncertainty_unit": uncertainty_unit,
                "ci_kind": ci_kind,
                "robust_seed_inference_available": robust,
                "inference_note": note,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLS).sort_values(group_cols).reset_index(drop=True)


def aggregate_study(
    experiment_dirs: Sequence[Path],
    output_dir: Path,
    *,
    n_bootstrap: int = 2000,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_study_metrics(experiment_dirs)
    deltas = build_paired_deltas(metrics)
    final_metrics = metrics[metrics["artifact_scope"] == "final"].copy()
    final_deltas = deltas[deltas["artifact_scope"] == "final"].copy()
    method_summary = summarize_estimates(
        final_metrics,
        value_col="metric_value",
        n_bootstrap=n_bootstrap,
    )
    delta_summary = summarize_estimates(
        final_deltas,
        value_col="delta",
        n_bootstrap=n_bootstrap,
    )
    paths = {
        "metric_rows": output_dir / "study_metric_rows.csv",
        "paired_deltas": output_dir / "study_paired_deltas.csv",
        "bound_paired_deltas": output_dir / "study_bound_paired_deltas.csv",
        "bound_curves": output_dir / "study_bound_curves.csv",
        "method_summary": output_dir / "study_method_summary.csv",
        "paired_delta_summary": output_dir / "study_paired_delta_summary.csv",
        "aggregation_metadata": output_dir / "study_aggregation_metadata.json",
    }
    metrics.to_csv(paths["metric_rows"], index=False)
    deltas.to_csv(paths["paired_deltas"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    delta_summary.to_csv(paths["paired_delta_summary"], index=False)
    primary = deltas[deltas["is_primary_endpoint"].astype(bool)] if not deltas.empty else deltas
    final_primary = primary[primary["artifact_scope"] == "final"].copy()
    bound_deltas = (
        final_deltas[final_deltas["condition"] != HONEST_CONDITION].copy()
        if not final_deltas.empty
        else final_deltas
    )
    bound_curves = (
        delta_summary[delta_summary["condition"] != HONEST_CONDITION].copy()
        if not delta_summary.empty
        else delta_summary
    )
    bound_deltas.to_csv(paths["bound_paired_deltas"], index=False)
    bound_curves.to_csv(paths["bound_curves"], index=False)
    payload = {
        "schema_version": 1,
        "experiment_dirs": [str(path.resolve()) for path in experiment_dirs],
        "metric_rows": len(metrics),
        "final_metric_rows_in_summaries": len(final_metrics),
        "paired_delta_rows": len(deltas),
        "final_paired_delta_rows_in_summaries": len(final_deltas),
        "primary_f1_at_25_sktr_minus_official_rows": len(primary),
        "final_primary_f1_at_25_sktr_minus_official_rows": len(final_primary),
        "final_bound_delta_rows": len(bound_deltas),
        "bound_curve_summary_rows": len(bound_curves),
        "core_metrics": list(CORE_METRICS),
        "primary_endpoint": {
            "method": "sktr",
            "comparator": "official_diffact",
            "metric": "f1@25",
            "condition": HONEST_CONDITION,
            "delta_definition": "method_value - comparator_value",
            "excludes_oracle": True,
            "excludes_structural_and_process_bounds_from_primary_claims": True,
        },
        "bound_curves": {
            "structural_prior_full_train_petri": "practical upper bound; reported separately",
            "oracle_test_fold_petri": "leaky theoretic ceiling; never enters primary claims",
            "full_diffact_low_data_petri": "process-scarcity control; reported separately",
        },
        "uncertainty_policy": (
            "Aggregate fold/seed estimates; never treat frames as independent. "
            "Seed0-only outputs are descriptive and explicitly marked as lacking robust seed inference. "
            "Runtime-only partial pilots remain in row-level files but are excluded from final summaries. "
            "Oracle rows are tagged is_oracle_condition and excluded from primary endpoint claims."
        ),
        "bootstrap_replicates": n_bootstrap,
    }
    paths["aggregation_metadata"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
