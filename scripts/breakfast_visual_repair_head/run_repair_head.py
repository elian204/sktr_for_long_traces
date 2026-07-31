#!/usr/bin/env python3
"""Run the pre-registered CPU-only Breakfast visual repair-head study."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    CLASSIFIER_MAX_ITER,
    OUTER_FOLDS,
    PRIMARY_BUDGET,
    PRIMARY_THRESHOLD,
    PROTOCOL_VERSION,
    SENSITIVITY_THRESHOLDS,
    WORKSPACE_ROOT,
    atomic_write_json,
    candidate_columns,
    canonical_digest,
    canonical_feature_view,
    evaluate_primary_gates,
    feature_orientation,
    file_sha256,
    load_json,
    pool_span_mean_std,
    stable_uint64,
    verify_source_provenance,
)

sys.path.insert(0, str(WORKSPACE_ROOT))
from src.evaluation import _edit_score_asformer, _segmental_f1_counts_asformer  # noqa: E402


@dataclass(frozen=True)
class CaseData:
    outer_fold: int
    case_id: str
    gt: np.ndarray
    baseline: np.ndarray


METRICS = ("acc", "edit", "f1@10", "f1@25", "f1@50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--verify-workers", type=int, default=4)
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _verified_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_hash = file_sha256(path)
    if actual_bytes != int(entry["bytes"]) or actual_hash != entry["sha256"]:
        raise RuntimeError(f"Input hash mismatch: {path}")
    result = {"path": str(path), "sha256": actual_hash, "bytes": actual_bytes}
    if "shape" in entry:
        array = np.load(path, mmap_mode="r")
        orientation, frames = feature_orientation(array.shape)
        if [int(value) for value in array.shape] != list(entry["shape"]):
            raise RuntimeError(f"Feature shape drift: {path}")
        if orientation != entry["orientation"] or frames != int(entry["time_frames"]):
            raise RuntimeError(f"Feature orientation/time drift: {path}")
        if str(array.dtype) != entry["dtype"]:
            raise RuntimeError(f"Feature dtype drift: {path}")
        result.update({"shape": list(array.shape), "orientation": orientation})
    return result


def verify_inputs(manifest: Mapping[str, Any], workers: int) -> Dict[str, Any]:
    expected_digest = manifest["manifest_digest"]
    without_digest = dict(manifest)
    without_digest.pop("manifest_digest")
    if canonical_digest(without_digest) != expected_digest:
        raise RuntimeError("Input manifest self-digest is invalid")
    entries = [
        *manifest["fixed_artifacts"],
        *manifest["ground_truth"],
        *manifest["features"],
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        verified = list(executor.map(_verified_entry, entries))
    return {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_digest": expected_digest,
        "files_verified": len(verified),
        "bytes_verified": int(sum(row["bytes"] for row in verified)),
        "all_hashes_match": True,
    }


def read_mapping(path: Path) -> tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    for line in path.read_text().splitlines():
        index, label = line.split(maxsplit=1)
        id_to_name[int(index)] = label
    if sorted(id_to_name) != list(range(48)):
        raise ValueError("Breakfast mapping must contain contiguous IDs 0..47")
    return id_to_name, {name: index for index, name in id_to_name.items()}


def load_ground_truth(
    manifest: Mapping[str, Any], name_to_id: Mapping[str, int]
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for entry in manifest["ground_truth"]:
        labels = Path(entry["path"]).read_text().splitlines()
        try:
            values = np.asarray([name_to_id[label] for label in labels], dtype=np.int16)
        except KeyError as error:
            raise ValueError(f"Unknown ground-truth label in {entry['path']}") from error
        if len(values) != int(entry["time_frames"]):
            raise ValueError(f"Ground-truth length drift: {entry['case_id']}")
        result[str(entry["case_id"])] = values
    return result


def validate_and_build_cases(
    outer: pd.DataFrame, ground_truth: Mapping[str, np.ndarray]
) -> list[CaseData]:
    cases: list[CaseData] = []
    for (fold, case_id), group in outer.groupby(["outer_fold", "case_id"], sort=True):
        ordered = group.sort_values(["segment_index", "start"], kind="mergesort")
        gt = ground_truth[str(case_id)]
        starts = ordered["start"].to_numpy(dtype=int)
        ends = ordered["end"].to_numpy(dtype=int)
        if starts[0] != 0 or ends[-1] != len(gt) or np.any(starts[1:] != ends[:-1]):
            raise ValueError(f"Outer segment partition mismatch: fold={fold}/{case_id}")
        pred = np.empty(len(gt), dtype=np.int16)
        for row in ordered.itertuples():
            pred[int(row.start) : int(row.end)] = int(row.predicted_label)
            span_gt = gt[int(row.start) : int(row.end)]
            counts = np.bincount(span_gt, minlength=48)
            frozen_label = int(row.correct_label)
            if counts[frozen_label] != counts.max():
                raise ValueError(
                    f"Frozen majority target is invalid: fold={fold}/{case_id}/"
                    f"segment={row.segment_index}"
                )
        cases.append(CaseData(int(fold), str(case_id), gt, pred))
    return cases


def validate_training_targets(
    corpus: pd.DataFrame, ground_truth: Mapping[str, np.ndarray]
) -> None:
    for row in corpus.itertuples():
        gt = ground_truth[str(row.case_id)][int(row.start) : int(row.end)]
        counts = np.bincount(gt, minlength=48)
        target = int(row.correct_label)
        if counts[target] != counts.max():
            raise ValueError(
                f"Invalid OOF majority label: outer={row.outer_fold}/"
                f"{row.case_id}/segment={row.segment_index}"
            )
        expected_fraction = counts[target] / len(gt)
        if abs(float(row.correct_label_fraction) - expected_fraction) > 1e-9:
            raise ValueError(f"OOF majority fraction drift: segment_id={row.segment_id}")


def select_budget_rows(segments: pd.DataFrame, total_frames: int) -> pd.DataFrame:
    """Exact frozen selector order: whole spans and at most one centered cutoff."""
    requested = max(1, int(round(PRIMARY_BUDGET * int(total_frames))))
    scoped = segments.copy()
    scoped["tie_key"] = [
        int(
            stable_uint64(
                row.scope,
                row.mode,
                row.inner_fold,
                row.case_id,
                row.segment_index,
            )
        )
        for row in scoped.itertuples()
    ]
    scoped = scoped.sort_values(
        ["base_score", "tie_key"], ascending=[False, True], kind="mergesort"
    )
    remaining = requested
    rows: list[Dict[str, Any]] = []
    cutoff_count = 0
    for row in scoped.itertuples():
        if remaining <= 0:
            break
        start, end = int(row.start), int(row.end)
        length = end - start
        selected = min(length, remaining)
        if selected == length:
            selected_start = start
        else:
            selected_start = start + (length - selected) // 2
            cutoff_count += 1
        rows.append(
            {
                "segment_id": int(row.segment_id),
                "outer_fold": int(row.outer_fold),
                "case_id": str(row.case_id),
                "segment_index": int(row.segment_index),
                "segment_start": start,
                "segment_end": end,
                "selected_start": selected_start,
                "selected_end": selected_start + selected,
                "selected_frames": selected,
                "is_partial_cutoff": selected != length,
                "base_score": float(row.base_score),
            }
        )
        remaining -= selected
    if remaining or cutoff_count > 1:
        raise AssertionError(
            f"Budget selection failed: remaining={remaining}, cutoffs={cutoff_count}"
        )
    result = pd.DataFrame(rows)
    if int(result["selected_frames"].sum()) != requested:
        raise AssertionError("Selected frames do not equal requested 5% budget")
    return result


def aggregate_metrics(cases: Sequence[CaseData], predictions: Mapping[str, np.ndarray]) -> Dict[str, float]:
    frames = 0
    correct = 0
    edits: list[float] = []
    counts = {0.10: [0, 0, 0], 0.25: [0, 0, 0], 0.50: [0, 0, 0]}
    for case in cases:
        pred = np.asarray(predictions[case.case_id], dtype=str)
        gt = np.asarray(case.gt, dtype=str)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction length mismatch: {case.case_id}")
        frames += len(gt)
        correct += int(np.sum(pred == gt))
        edits.append(float(_edit_score_asformer(pred.tolist(), gt.tolist(), ["background"])))
        for threshold in counts:
            tp, fp, fn = _segmental_f1_counts_asformer(
                gt.tolist(), pred.tolist(), threshold, None
            )
            counts[threshold][0] += int(tp)
            counts[threshold][1] += int(fp)
            counts[threshold][2] += int(fn)
    result = {"acc": 100.0 * safe_div(correct, frames), "edit": float(np.mean(edits))}
    for threshold, (tp, fp, fn) in counts.items():
        result[f"f1@{int(100 * threshold)}"] = 100.0 * safe_div(
            2 * tp, 2 * tp + fp + fn
        )
    return result


def reconcile_baselines(
    cases: Sequence[CaseData], upstream: pd.DataFrame
) -> tuple[Dict[int | str, Dict[str, float]], list[Dict[str, Any]]]:
    baselines: Dict[int | str, Dict[str, float]] = {}
    checks: list[Dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        scoped = [case for case in cases if case.outer_fold == fold]
        actual = aggregate_metrics(scoped, {case.case_id: case.baseline for case in scoped})
        expected_rows = upstream[
            (upstream["scope"] == "official_outer_test")
            & (upstream["mode"] == "official")
            & (upstream["fold"] == f"outer_{fold}")
        ]
        if len(expected_rows) != 1:
            raise ValueError(f"Missing upstream baseline for outer_{fold}")
        expected = expected_rows.iloc[0]
        for metric in METRICS:
            delta = abs(actual[metric] - float(expected[metric]))
            checks.append(
                {
                    "check": f"baseline_reconciliation_outer_{fold}_{metric}",
                    "pass": delta <= 1e-9,
                    "detail": f"absolute_delta={delta:.12g}",
                }
            )
            if delta > 1e-9:
                raise RuntimeError(f"Baseline metric drift: outer_{fold}/{metric}={delta}")
        baselines[fold] = actual
    actual_pooled = aggregate_metrics(cases, {case.case_id: case.baseline for case in cases})
    expected_rows = upstream[
        (upstream["scope"] == "pooled_outer_test")
        & (upstream["mode"] == "official")
        & (upstream["fold"] == "pooled")
    ]
    if len(expected_rows) != 1:
        raise ValueError("Missing upstream pooled baseline")
    expected = expected_rows.iloc[0]
    for metric in METRICS:
        delta = abs(actual_pooled[metric] - float(expected[metric]))
        checks.append(
            {
                "check": f"baseline_reconciliation_pooled_{metric}",
                "pass": delta <= 1e-9,
                "detail": f"absolute_delta={delta:.12g}",
            }
        )
        if delta > 1e-9:
            raise RuntimeError(f"Pooled baseline metric drift: {metric}={delta}")
    baselines["pooled"] = actual_pooled
    return baselines, checks


def reconcile_budget(
    selections: Mapping[int, pd.DataFrame],
    cases: Sequence[CaseData],
    upstream: pd.DataFrame,
) -> list[Dict[str, Any]]:
    checks: list[Dict[str, Any]] = []
    case_map = {(case.outer_fold, case.case_id): case for case in cases}
    for fold, selected in selections.items():
        errors = 0
        for row in selected.itertuples():
            case = case_map[(fold, str(row.case_id))]
            mask_gt = case.gt[int(row.selected_start) : int(row.selected_end)]
            mask_pred = case.baseline[int(row.selected_start) : int(row.selected_end)]
            errors += int(np.sum(mask_gt != mask_pred))
        expected_rows = upstream[
            (upstream["scope"] == "official_outer_test")
            & (upstream["mode"] == "official")
            & (upstream["fold"] == f"outer_{fold}")
            & (upstream["variant"] == "base")
            & (np.isclose(upstream["requested_budget"], PRIMARY_BUDGET))
        ]
        if len(expected_rows) != 1:
            raise ValueError(f"Missing frozen budget row for outer_{fold}")
        expected = expected_rows.iloc[0]
        actual_frames = int(selected["selected_frames"].sum())
        expected_frames = int(expected["flagged_frames"])
        expected_errors = int(expected["flagged_error_frames"])
        passed = actual_frames == expected_frames and errors == expected_errors
        checks.append(
            {
                "check": f"selector_mask_reconciliation_outer_{fold}",
                "pass": passed,
                "detail": (
                    f"frames={actual_frames}/{expected_frames}; "
                    f"error_frames={errors}/{expected_errors}"
                ),
            }
        )
        if not passed:
            raise RuntimeError(f"Frozen selector mask mismatch on outer_{fold}")
    return checks


def pool_all_features(
    records: pd.DataFrame, feature_entries: Mapping[str, Mapping[str, Any]]
) -> np.ndarray:
    output = np.empty((len(records), 4096), dtype=np.float32)
    for case_id, positions in records.groupby("case_id", sort=True).groups.items():
        entry = feature_entries[str(case_id)]
        array = np.load(entry["path"], mmap_mode="r")
        canonical = canonical_feature_view(array, int(entry["time_frames"]))
        cache: Dict[tuple[int, int], np.ndarray] = {}
        for position in positions:
            row = records.loc[position]
            span = (int(row.start), int(row.end))
            if span not in cache:
                cache[span] = pool_span_mean_std(
                    canonical, span[0], span[1], int(entry["time_frames"])
                )
            output[int(position)] = cache[span]
    if not np.isfinite(output).all():
        raise ValueError("Non-finite pooled visual features")
    return output


def fit_length_weighted_classifier(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> LogisticRegression:
    """Fit the frozen prototype classifier and reject every non-converged fit."""
    classifier = LogisticRegression(C=1.0, max_iter=CLASSIFIER_MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(features, targets, sample_weight=weights)
    convergence = [
        item for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence:
        messages = "; ".join(str(item.message) for item in convergence)
        raise RuntimeError(
            "Primary multinomial repair-head fit emitted ConvergenceWarning; "
            f"aborting without accepting the fit: {messages}"
        )
    return classifier


def fit_models(
    records: pd.DataFrame,
    pooled: np.ndarray,
    study_dir: Path,
) -> tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
    models: Dict[int, Dict[str, Any]] = {}
    audits: list[Dict[str, Any]] = []
    model_dir = study_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in OUTER_FOLDS:
        train_mask = (records["record_scope"] == "oof_train") & (
            records["outer_fold"].astype(int) == fold
        )
        test_mask = (records["record_scope"] == "outer_test") & (
            records["outer_fold"].astype(int) == fold
        )
        train_positions = np.flatnonzero(train_mask.to_numpy())
        test_positions = np.flatnonzero(test_mask.to_numpy())
        train = records.iloc[train_positions]
        test = records.iloc[test_positions]
        if set(train["case_id"]) & set(test["case_id"]):
            raise ValueError(f"Train/test case overlap in outer fold {fold}")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(pooled[train_positions])
        x_test = scaler.transform(pooled[test_positions])
        weights = train["length"].to_numpy(dtype=float)
        targets = train["correct_label"].to_numpy(dtype=int)
        classifier = fit_length_weighted_classifier(x_train, targets, weights)
        probabilities = classifier.predict_proba(x_test)
        model_path = model_dir / f"outer_fold_{fold}.joblib"
        joblib.dump({"scaler": scaler, "classifier": classifier}, model_path)
        models[fold] = {
            "classifier": classifier,
            "test_positions": test_positions,
            "probabilities": probabilities,
            "probability_by_segment_id": {
                int(segment_id): probabilities[index]
                for index, segment_id in enumerate(test["segment_id"].astype(int))
            },
        }
        params = classifier.get_params(deep=False)
        audits.append(
            {
                "outer_fold": fold,
                "train_segments": len(train),
                "train_cases": train["case_id"].nunique(),
                "test_segments": len(test),
                "test_cases": test["case_id"].nunique(),
                "visual_feature_count": x_train.shape[1],
                "target_class_count": len(classifier.classes_),
                "train_frame_weight": int(weights.sum()),
                "scaler_sample_weighted": False,
                "classifier_C": params["C"],
                "classifier_solver": params["solver"],
                "classifier_max_iter": params["max_iter"],
                "classifier_multi_class": params["multi_class"],
                "max_n_iter": int(np.max(classifier.n_iter_)),
                "convergence_warning_policy": "hard_failure",
                "convergence_assertion_passed": True,
                "model_path": str(model_path),
                "model_sha256": file_sha256(model_path),
                "uses_metadata": False,
                "uses_process_features": False,
                "uses_shape_features": False,
                "uses_selector_features": False,
            }
        )
    return models, pd.DataFrame(audits)


def proposal_for_segment(
    segment: Mapping[str, Any],
    classifier: LogisticRegression,
    probabilities: np.ndarray,
    variant: str,
) -> tuple[int | None, float, str]:
    classes = classifier.classes_.astype(int)
    if variant == "free_choice":
        index = int(np.argmax(probabilities))
        return int(classes[index]), float(probabilities[index]), "head_top1"
    if variant != "top5_restricted":
        raise ValueError(variant)
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    candidates: list[tuple[int, float]] = []
    for rank in range(1, 6):
        class_column, _, _ = candidate_columns(rank)
        label = int(segment[class_column])
        if label in class_to_index:
            candidates.append((label, float(probabilities[class_to_index[label]])))
    if not candidates:
        return None, float("nan"), "no_trained_class_in_top5"
    label, probability = max(candidates, key=lambda pair: pair[1])
    return label, probability, "highest_unrenormalized_head_probability_in_top5"


def apply_repairs(
    cases: Sequence[CaseData],
    outer: pd.DataFrame,
    selections: Mapping[int, pd.DataFrame],
    models: Mapping[int, Mapping[str, Any]],
    *,
    threshold: float,
    variant: str,
    include_ledger: bool,
) -> tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, int]]:
    predictions = {case.case_id: case.baseline.copy() for case in cases}
    case_map = {(case.outer_fold, case.case_id): case for case in cases}
    segment_lookup = outer.set_index("segment_id", drop=False)
    ledger: list[Dict[str, Any]] = []
    accepted = 0
    abstained = 0
    for fold in OUTER_FOLDS:
        classifier = models[fold]["classifier"]
        probability_lookup = models[fold]["probability_by_segment_id"]
        for selected in selections[fold].itertuples(index=False):
            segment = segment_lookup.loc[int(selected.segment_id)]
            probability_vector = probability_lookup[int(selected.segment_id)]
            proposal, confidence, proposal_rule = proposal_for_segment(
                segment, classifier, probability_vector, variant
            )
            current = int(segment["predicted_label"])
            if proposal is None:
                decision = "abstain_no_candidate"
            elif confidence < float(threshold):
                decision = "abstain_below_threshold"
            elif proposal == current:
                decision = "abstain_same_label"
            else:
                decision = "accept"
            case = case_map[(fold, str(selected.case_id))]
            start, end = int(selected.selected_start), int(selected.selected_end)
            before = case.baseline[start:end]
            gt = case.gt[start:end]
            if decision == "accept":
                predictions[case.case_id][start:end] = int(proposal)
                after = predictions[case.case_id][start:end]
                accepted += 1
            else:
                after = before
                abstained += 1
            if include_ledger:
                fixed = int(np.sum((before != gt) & (after == gt)))
                broken = int(np.sum((before == gt) & (after != gt)))
                lateral = int(np.sum((before != gt) & (after != gt) & (before != after)))
                net = fixed - broken
                outcome = "helped" if net > 0 else "hurt" if net < 0 else "lateral"
                if decision != "accept":
                    outcome = "abstained"
                ledger.append(
                    {
                        **selected._asdict(),
                        "variant": variant,
                        "threshold": threshold,
                        "current_label": current,
                        "gt_majority_label": int(segment["correct_label"]),
                        "majority_status": (
                            "right_majority"
                            if current == int(segment["correct_label"])
                            else "wrong_majority"
                        ),
                        "proposal_label": proposal,
                        "proposal_probability": confidence,
                        "proposal_rule": proposal_rule,
                        "decision": decision,
                        "outcome": outcome,
                        "fixed_frames": fixed,
                        "broken_frames": broken,
                        "lateral_changed_wrong_frames": lateral,
                        "net_correct_frames": net,
                    }
                )
    return predictions, pd.DataFrame(ledger), {"accepted_spans": accepted, "abstained_spans": abstained}


def metric_comparison_row(
    scope: str,
    fold: str,
    cases: Sequence[CaseData],
    baseline: Mapping[str, float],
    repaired_predictions: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
    *,
    threshold: float,
    variant: str,
) -> Dict[str, Any]:
    repaired = aggregate_metrics(cases, repaired_predictions)
    row: Dict[str, Any] = {
        "scope": scope,
        "fold": fold,
        "variant": variant,
        "threshold": threshold,
        "n_cases": len(cases),
        "n_frames": int(sum(len(case.gt) for case in cases)),
        **counts,
    }
    for metric in METRICS:
        row[f"baseline_{metric}"] = baseline[metric]
        row[f"repair_{metric}"] = repaired[metric]
        row[f"delta_{metric}"] = repaired[metric] - baseline[metric]
    return row


def evaluate_scenarios(
    cases: Sequence[CaseData],
    outer: pd.DataFrame,
    selections: Mapping[int, pd.DataFrame],
    models: Mapping[int, Mapping[str, Any]],
    baselines: Mapping[int | str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    rows: list[Dict[str, Any]] = []
    primary_ledger = pd.DataFrame()
    primary_predictions: Dict[str, np.ndarray] = {}
    for variant in ("free_choice", "top5_restricted"):
        for threshold in SENSITIVITY_THRESHOLDS:
            is_primary = variant == "free_choice" and threshold == PRIMARY_THRESHOLD
            predictions, ledger, counts = apply_repairs(
                cases,
                outer,
                selections,
                models,
                threshold=threshold,
                variant=variant,
                include_ledger=is_primary,
            )
            if is_primary:
                primary_ledger = ledger
                primary_predictions = predictions
            for fold in OUTER_FOLDS:
                scoped = [case for case in cases if case.outer_fold == fold]
                scoped_predictions = {case.case_id: predictions[case.case_id] for case in scoped}
                fold_counts = {
                    "accepted_spans": int(
                        sum(
                            1
                            for row in selections[fold].itertuples()
                            if not np.array_equal(
                                scoped_predictions[str(row.case_id)][
                                    int(row.selected_start) : int(row.selected_end)
                                ],
                                next(
                                    case.baseline[
                                        int(row.selected_start) : int(row.selected_end)
                                    ]
                                    for case in scoped
                                    if case.case_id == str(row.case_id)
                                ),
                            )
                        )
                    )
                }
                fold_counts["abstained_spans"] = len(selections[fold]) - fold_counts["accepted_spans"]
                rows.append(
                    metric_comparison_row(
                        "outer_fold",
                        f"outer_{fold}",
                        scoped,
                        baselines[fold],
                        scoped_predictions,
                        fold_counts,
                        threshold=threshold,
                        variant=variant,
                    )
                )
            rows.append(
                metric_comparison_row(
                    "pooled",
                    "pooled",
                    cases,
                    baselines["pooled"],
                    predictions,
                    counts,
                    threshold=threshold,
                    variant=variant,
                )
            )
    return pd.DataFrame(rows), primary_ledger, primary_predictions


def build_video_metrics(
    cases: Sequence[CaseData], predictions: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for case in cases:
        baseline_correct = int(np.sum(case.baseline == case.gt))
        repair_correct = int(np.sum(predictions[case.case_id] == case.gt))
        rows.append(
            {
                "outer_fold": case.outer_fold,
                "case_id": case.case_id,
                "n_frames": len(case.gt),
                "baseline_correct_frames": baseline_correct,
                "repair_correct_frames": repair_correct,
                "net_correct_frames": repair_correct - baseline_correct,
                "baseline_acc": 100.0 * baseline_correct / len(case.gt),
                "repair_acc": 100.0 * repair_correct / len(case.gt),
                "delta_acc": 100.0 * (repair_correct - baseline_correct) / len(case.gt),
            }
        )
    result = pd.DataFrame(rows)
    total_net = int(result["net_correct_frames"].sum())
    result["positive_gain_fraction"] = np.where(
        (result["net_correct_frames"] > 0) & (total_net > 0),
        result["net_correct_frames"] / total_net,
        0.0,
    )
    return result


def build_ledger_summaries(
    ledger: pd.DataFrame, id_to_name: Mapping[int, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = ledger[ledger["decision"] == "accept"].copy()
    split_rows: list[Dict[str, Any]] = []
    for status, group in ledger.groupby("majority_status", sort=True):
        accepted_group = group[group["decision"] == "accept"]
        split_rows.append(
            {
                "majority_status": status,
                "flagged_spans": len(group),
                "flagged_selected_frames": int(group["selected_frames"].sum()),
                "accepted_spans": len(accepted_group),
                "abstained_spans": len(group) - len(accepted_group),
                "accepted_selected_frames": int(accepted_group["selected_frames"].sum()),
                "fixed_frames": int(accepted_group["fixed_frames"].sum()),
                "broken_frames": int(accepted_group["broken_frames"].sum()),
                "lateral_changed_wrong_frames": int(
                    accepted_group["lateral_changed_wrong_frames"].sum()
                ),
                "net_correct_frames": int(accepted_group["net_correct_frames"].sum()),
            }
        )
    split = pd.DataFrame(split_rows)
    if accepted.empty:
        return split, pd.DataFrame()
    class_rows = []
    for basis, column in (
        ("proposal", "proposal_label"),
        ("gt_majority", "gt_majority_label"),
        ("current", "current_label"),
    ):
        grouped = accepted.groupby(column, as_index=False).agg(
            accepted_spans=("segment_id", "size"),
            selected_frames=("selected_frames", "sum"),
            fixed_frames=("fixed_frames", "sum"),
            broken_frames=("broken_frames", "sum"),
            lateral_changed_wrong_frames=("lateral_changed_wrong_frames", "sum"),
            net_correct_frames=("net_correct_frames", "sum"),
        )
        for row in grouped.itertuples(index=False):
            values = row._asdict()
            label = int(values.pop(column))
            class_rows.append(
                {"basis": basis, "class_id": label, "class_name": id_to_name[label], **values}
            )
    return split, pd.DataFrame(class_rows)


def save_primary_predictions(
    cases: Sequence[CaseData], predictions: Mapping[str, np.ndarray], study_dir: Path
) -> list[Dict[str, Any]]:
    records = []
    for fold in OUTER_FOLDS:
        path = study_dir / "predictions" / f"outer_fold_{fold}_primary.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        scoped = sorted((case for case in cases if case.outer_fold == fold), key=lambda case: case.case_id)
        np.savez_compressed(path, **{case.case_id: predictions[case.case_id] for case in scoped})
        records.append({"outer_fold": fold, "path": str(path), "sha256": file_sha256(path)})
    return records


def write_findings(
    path: Path, final_metrics: pd.DataFrame, decision: Mapping[str, Any]
) -> None:
    pooled = final_metrics[final_metrics["scope"] == "pooled"].iloc[0]
    lines = [
        "# Breakfast visual repair-head findings",
        "",
        f"**Pre-registered verdict: {'PASS' if decision['pass'] else 'NO-GO'}.**",
        "",
        (
            f"At the frozen 5% selector budget and tau=0.5, pooled realized gains "
            f"were {pooled.delta_acc:+.3f} Accuracy, {pooled.delta_edit:+.3f} Edit, "
            f"and {pooled['delta_f1@25']:+.3f} F1@25 points."
        ),
        "",
        "| Fold | Delta Acc | Delta Edit | Delta F1@10 | Delta F1@25 | Delta F1@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in final_metrics.iterrows():
        lines.append(
            f"| {row['fold']} | {row['delta_acc']:+.3f} | {row['delta_edit']:+.3f} | "
            f"{row['delta_f1@10']:+.3f} | {row['delta_f1@25']:+.3f} | "
            f"{row['delta_f1@50']:+.3f} |"
        )
    lines.extend(["", "## Frozen gate checks", ""])
    for check, passed in decision["checks"].items():
        lines.append(f"- `{check}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "Threshold sensitivity and top-five restriction are analysis-only and do not alter this verdict.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def execute(study_dir: Path, verify_workers: int) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    if metadata["protocol_version"] != PROTOCOL_VERSION or config["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("Protocol version mismatch")
    if canonical_digest({key: value for key, value in config.items() if key != "config_digest"}) != config["config_digest"]:
        raise RuntimeError("Study config digest is invalid")
    verify_source_provenance(metadata["source_provenance"])

    results_dir = study_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "run_complete.json").exists():
        raise FileExistsError("Study is already complete")
    if (results_dir / "run_failed.json").exists():
        raise FileExistsError("Study previously failed; stage a fresh immutable directory")
    atomic_write_json(
        results_dir / "run_status.json",
        {
            "status": "verifying_inputs",
            "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )
    verification = verify_inputs(manifest, verify_workers)
    atomic_write_json(results_dir / "input_verification.json", verification)

    fixed_by_role = {row["role"]: Path(row["path"]) for row in manifest["fixed_artifacts"]}
    scores = pd.read_csv(fixed_by_role["selector_analysis/segment_scores.csv"])
    corpus = pd.read_csv(fixed_by_role["selector_analysis/repair_training_corpus_segments.csv"])
    upstream_baselines = pd.read_csv(fixed_by_role["selector_analysis/baseline_metrics.csv"])
    upstream_budgets = pd.read_csv(fixed_by_role["selector_analysis/selector_budget_metrics.csv"])
    mapping_path = fixed_by_role["label_mapping"]
    id_to_name, name_to_id = read_mapping(mapping_path)
    ground_truth = load_ground_truth(manifest, name_to_id)

    outer = scores[(scores["scope"] == "outer_test") & (scores["mode"] == "official")].copy()
    outer = outer.sort_values(["outer_fold", "case_id", "segment_index"], kind="mergesort")
    corpus = corpus.sort_values(["outer_fold", "case_id", "segment_index"], kind="mergesort")
    cases = validate_and_build_cases(outer, ground_truth)
    validate_training_targets(corpus, ground_truth)
    baselines, validation_checks = reconcile_baselines(cases, upstream_baselines)

    selections: Dict[int, pd.DataFrame] = {}
    for fold in OUTER_FOLDS:
        fold_segments = outer[outer["outer_fold"].astype(int) == fold]
        total_frames = sum(len(case.gt) for case in cases if case.outer_fold == fold)
        selections[fold] = select_budget_rows(fold_segments, total_frames)
    validation_checks.extend(reconcile_budget(selections, cases, upstream_budgets))

    atomic_write_json(
        results_dir / "run_status.json",
        {
            "status": "pooling_visual_features",
            "pid": os.getpid(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    train_records = corpus.copy()
    train_records["record_scope"] = "oof_train"
    test_records = outer.copy()
    test_records["record_scope"] = "outer_test"
    records = pd.concat([train_records, test_records], ignore_index=True, sort=False)
    records.index = np.arange(len(records))
    feature_entries = {str(row["case_id"]): row for row in manifest["features"]}
    pooled_features = pool_all_features(records, feature_entries)

    atomic_write_json(
        results_dir / "run_status.json",
        {
            "status": "fitting_fold_models",
            "pid": os.getpid(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    models, model_audit = fit_models(records, pooled_features, study_dir)
    model_audit.to_csv(results_dir / "model_audit.csv", index=False)

    sensitivity, ledger, primary_predictions = evaluate_scenarios(
        cases, outer, selections, models, baselines
    )
    sensitivity.to_csv(results_dir / "threshold_sensitivity.csv", index=False)
    final_metrics = sensitivity[
        (sensitivity["variant"] == "free_choice")
        & np.isclose(sensitivity["threshold"], PRIMARY_THRESHOLD)
    ].copy()
    final_metrics.to_csv(results_dir / "final_metrics.csv", index=False)
    ledger.to_csv(results_dir / "helped_hurt_lateral_ledger.csv", index=False)

    video_metrics = build_video_metrics(cases, primary_predictions)
    video_metrics.to_csv(results_dir / "video_metrics.csv", index=False)
    majority_split, class_contributions = build_ledger_summaries(ledger, id_to_name)
    majority_split.to_csv(results_dir / "majority_status_contributions.csv", index=False)
    class_contributions.to_csv(results_dir / "class_contributions.csv", index=False)
    prediction_artifacts = save_primary_predictions(cases, primary_predictions, study_dir)

    pooled = final_metrics[final_metrics["scope"] == "pooled"].iloc[0]
    per_fold = final_metrics[final_metrics["scope"] == "outer_fold"].sort_values("fold")
    pooled_delta = {metric: float(pooled[f"delta_{metric}"]) for metric in METRICS}
    decision = evaluate_primary_gates(
        pooled_delta,
        per_fold["delta_acc"].to_numpy(dtype=float),
        video_metrics["delta_acc"].to_numpy(dtype=float),
        video_metrics["net_correct_frames"].to_numpy(dtype=int),
        config["gates"],
    )
    decision.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "criteria_were_pre_registered": True,
            "primary_variant": "free_choice",
            "primary_threshold": PRIMARY_THRESHOLD,
            "primary_budget": PRIMARY_BUDGET,
            "pooled_delta": pooled_delta,
            "per_fold_acc_deltas": {
                str(row.fold): float(row.delta_acc) for row in per_fold.itertuples()
            },
            "gpu_extensions_unlocked": bool(decision["pass"]),
            "gpu_extensions_launched": False,
        }
    )
    atomic_write_json(results_dir / "decision.json", decision)
    pd.DataFrame(validation_checks).to_csv(results_dir / "validation_checks.csv", index=False)
    write_findings(results_dir / "findings.md", final_metrics, decision)

    output_paths = [
        *sorted(results_dir.glob("*.csv")),
        *sorted(results_dir.glob("*.json")),
        results_dir / "findings.md",
        *sorted((study_dir / "models").glob("*.joblib")),
        *sorted((study_dir / "predictions").glob("*.npz")),
    ]
    output_paths = [
        path
        for path in output_paths
        if path.name not in {"run_status.json", "run_complete.json", "run_failed.json"}
    ]
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "decision_pass": bool(decision["pass"]),
        "input_manifest_digest": manifest["manifest_digest"],
        "config_digest": config["config_digest"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "prediction_artifacts": prediction_artifacts,
        "output_sha256": {
            str(path.relative_to(study_dir)): file_sha256(path) for path in output_paths
        },
    }
    atomic_write_json(results_dir / "run_complete.json", complete)
    (results_dir / "run_status.json").unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    lock_path = study_dir / ".run.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another repair-head process holds the study lock") from error
        try:
            execute(study_dir, args.verify_workers)
        except Exception as error:
            results_dir = study_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                results_dir / "run_failed.json",
                {
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
