#!/usr/bin/env python3
"""Execute the approved, single-shot outer evaluation for repair-head v3."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from common import (
    OUTER_FOLDS,
    PRIMARY_BUDGET,
    PROTOCOL_VERSION,
    SENSITIVITY_THRESHOLDS,
    atomic_write_json,
    file_sha256,
    load_json,
    verify_self_digest,
    verify_source_provenance,
)
from core import (
    METRICS,
    CaseData,
    aggregate_metrics,
    apply_configuration,
    baseline_metrics,
    build_cases,
    fit_agreement_ensemble_probabilities,
    fixed_paths,
    load_ground_truth,
    pool_all_features,
    primary_gate_decision,
    read_mapping,
    select_budget_rows,
    validate_segment_table,
    verify_manifest_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--verify-workers", type=int, default=4)
    return parser.parse_args()


def _status(path: Path, status: str, **extra: Any) -> None:
    atomic_write_json(
        path,
        {
            "status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            **extra,
        },
    )


def _metric_row(
    scope: str,
    fold: str,
    cases: Sequence[CaseData],
    predictions: Mapping[Any, np.ndarray],
    *,
    readout: str,
) -> Dict[str, Any]:
    baseline = baseline_metrics(cases)
    repaired = aggregate_metrics(cases, predictions)
    row: Dict[str, Any] = {
        "readout": readout,
        "scope": scope,
        "fold": fold,
        "n_cases": len(cases),
        "n_frames": int(sum(len(case.gt) for case in cases)),
    }
    for metric in METRICS:
        row[f"baseline_{metric}"] = baseline[metric]
        row[f"repair_{metric}"] = repaired[metric]
        row[f"delta_{metric}"] = repaired[metric] - baseline[metric]
    return row


def _reconcile_baseline(
    cases: Sequence[CaseData], upstream: pd.DataFrame
) -> list[Dict[str, Any]]:
    checks = []
    for fold in OUTER_FOLDS:
        scoped = [case for case in cases if case.outer_fold == fold]
        actual = baseline_metrics(scoped)
        expected_rows = upstream[
            (upstream["scope"] == "official_outer_test")
            & (upstream["mode"] == "official")
            & (upstream["fold"] == f"outer_{fold}")
        ]
        if len(expected_rows) != 1:
            raise ValueError(f"Missing baseline outer_{fold}")
        expected = expected_rows.iloc[0]
        for metric in METRICS:
            delta = abs(actual[metric] - float(expected[metric]))
            if delta > 1e-9:
                raise RuntimeError(f"Baseline drift outer_{fold}/{metric}: {delta}")
            checks.append(
                {
                    "check": f"baseline_outer_{fold}_{metric}",
                    "pass": True,
                    "detail": f"absolute_delta={delta:.12g}",
                }
            )
    actual = baseline_metrics(cases)
    expected = upstream[
        (upstream["scope"] == "pooled_outer_test")
        & (upstream["mode"] == "official")
        & (upstream["fold"] == "pooled")
    ].iloc[0]
    for metric in METRICS:
        delta = abs(actual[metric] - float(expected[metric]))
        if delta > 1e-9:
            raise RuntimeError(f"Pooled baseline drift/{metric}: {delta}")
        checks.append(
            {
                "check": f"baseline_pooled_{metric}",
                "pass": True,
                "detail": f"absolute_delta={delta:.12g}",
            }
        )
    return checks


def _reconcile_primary_masks(
    cases: Sequence[CaseData],
    selections: Mapping[tuple[int, float], pd.DataFrame],
    upstream: pd.DataFrame,
) -> list[Dict[str, Any]]:
    lookup = {(case.outer_fold, case.case_id): case for case in cases}
    checks = []
    for fold in OUTER_FOLDS:
        selected = selections[(fold, PRIMARY_BUDGET)]
        errors = 0
        for row in selected.itertuples(index=False):
            case = lookup[(fold, str(row.case_id))]
            start, end = int(row.selected_start), int(row.selected_end)
            errors += int(np.sum(case.baseline[start:end] != case.gt[start:end]))
        expected_rows = upstream[
            (upstream["scope"] == "official_outer_test")
            & (upstream["mode"] == "official")
            & (upstream["fold"] == f"outer_{fold}")
            & (upstream["variant"] == "base")
            & np.isclose(upstream["requested_budget"], PRIMARY_BUDGET)
        ]
        if len(expected_rows) != 1:
            raise ValueError(f"Missing upstream mask audit outer_{fold}")
        expected = expected_rows.iloc[0]
        actual_frames = int(selected["selected_frames"].sum())
        passed = (
            actual_frames == int(expected["flagged_frames"])
            and errors == int(expected["flagged_error_frames"])
        )
        if not passed:
            raise RuntimeError(f"Frozen 5% selector mask drift outer_{fold}")
        checks.append(
            {
                "check": f"primary_selector_mask_outer_{fold}",
                "pass": True,
                "detail": f"frames={actual_frames}; error_frames={errors}",
            }
        )
    return checks


def _anonymize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    result["segment_key_sha256"] = [
        hashlib.sha256(f"outer={fold}|segment={segment}".encode()).hexdigest()
        for fold, segment in zip(result["outer_fold"], result["segment_id"])
    ]
    return result.drop(
        columns=[column for column in ("case_id", "case_key", "segment_id") if column in result],
        errors="ignore",
    )


def _class_contributions(
    ledger: pd.DataFrame, id_to_name: Mapping[int, str]
) -> pd.DataFrame:
    accepted = ledger[ledger["accepted"]]
    rows = []
    for basis, column in (
        ("proposal", "proposal_label"),
        ("incumbent", "incumbent_label"),
    ):
        for label, group in accepted.groupby(column):
            rows.append(
                {
                    "basis": basis,
                    "class_id": int(label),
                    "class_name": id_to_name[int(label)],
                    "accepted_spans": len(group),
                    "relabelled_frames": int(group["relabelled_frames"].sum()),
                    "fixed_frames": int(group["fixed_frames"].sum()),
                    "broken_frames": int(group["broken_frames"].sum()),
                    "net_correct_frames": int(group["net_correct_frames"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _write_findings(path: Path, metrics: pd.DataFrame, decision: Mapping[str, Any]) -> None:
    pooled = metrics[(metrics["readout"] == "primary_5pct") & (metrics["scope"] == "pooled")].iloc[0]
    lines = [
        "# Repair-head v3 final single-shot outer findings",
        "",
        f"**Pre-registered primary verdict: {'PASS' if decision['pass'] else 'NO-GO'}.**",
        "",
        (
            f"Pooled realized primary deltas: Acc {pooled.delta_acc:+.3f}, "
            f"Edit {pooled.delta_edit:+.3f}, F1@25 {pooled['delta_f1@25']:+.3f}."
        ),
        "",
        "| Fold | Delta Acc | Delta Edit | Delta F1@10 | Delta F1@25 | Delta F1@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary = metrics[metrics["readout"] == "primary_5pct"]
    for _, row in primary.iterrows():
        lines.append(
            f"| {row['fold']} | {row['delta_acc']:+.3f} | {row['delta_edit']:+.3f} | "
            f"{row['delta_f1@10']:+.3f} | {row['delta_f1@25']:+.3f} | "
            f"{row['delta_f1@50']:+.3f} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for check, passed in decision["checks"].items():
        lines.append(f"- `{check}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "The optional OOF-selected larger-budget envelope is secondary and does not alter the primary verdict.",
            "If any primary gate fails, the repair line closes permanently; no further threshold or rule attempt is authorized.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def execute(study_dir: Path, verify_workers: int) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    frozen = load_json(study_dir / "frozen_outer_config.json")
    manifest = load_json(study_dir / "outer_input_manifest.json")
    approval = load_json(study_dir / "outer_review_approval.json")
    verify_self_digest(frozen, "frozen_config_digest")
    if frozen["outer_evaluation_blocked"]:
        raise RuntimeError("Frozen OOF decision blocks outer evaluation")
    if not frozen["mechanism_gate"]["pass"]:
        raise RuntimeError("Agreement mechanism gate did not pass")
    if not frozen["governance"]["final_outer_attempt_for_repair_line"]:
        raise RuntimeError("Final-attempt governance flag is absent")
    if not approval.get("authorizes_exactly_one_outer_evaluation"):
        raise RuntimeError("Outer approval does not authorize the run")
    for field, value in (
        ("frozen_config_digest", frozen["frozen_config_digest"]),
        ("source_digest", metadata["source_provenance"]["source_digest"]),
        ("outer_input_manifest_digest", manifest["manifest_digest"]),
    ):
        if approval.get(field) != value:
            raise RuntimeError(f"Outer approval mismatch: {field}")
    verify_source_provenance(metadata["source_provenance"])

    results_dir = study_dir / "outer_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "run_complete.json").exists() or (results_dir / "run_failed.json").exists():
        raise FileExistsError("Single outer evaluation already terminated")
    status_path = results_dir / "run_status.json"
    _status(status_path, "verifying_approved_outer_inputs")
    verification = verify_manifest_entries(manifest, workers=verify_workers)
    atomic_write_json(results_dir / "outer_input_verification.json", verification)
    atomic_write_json(
        results_dir / "outer_access_audit.json",
        {
            "outer_content_first_opened_utc": datetime.now(timezone.utc).isoformat(),
            "approval_sha256": file_sha256(study_dir / "outer_review_approval.json"),
            "frozen_config_digest": frozen["frozen_config_digest"],
            "single_shot": True,
        },
    )

    paths = fixed_paths(manifest)
    corpus = pd.read_csv(paths["selector_analysis/repair_training_corpus_segments.csv"])
    scores = pd.read_csv(paths["selector_analysis/segment_scores.csv"])
    outer = scores[(scores["scope"] == "outer_test") & (scores["mode"] == "official")].copy()
    validate_segment_table(corpus, oof_only=True)
    validate_segment_table(outer, oof_only=False)
    corpus = corpus.sort_values(
        ["outer_fold", "inner_fold", "case_id", "segment_index"], kind="mergesort"
    ).reset_index(drop=True)
    outer = outer.sort_values(
        ["outer_fold", "case_id", "segment_index"], kind="mergesort"
    ).reset_index(drop=True)
    id_to_name, name_to_id = read_mapping(paths["label_mapping"])
    ground_truth = load_ground_truth(manifest, name_to_id)
    cases = build_cases(outer, ground_truth, include_inner=False)
    checks = _reconcile_baseline(
        cases, pd.read_csv(paths["selector_analysis/baseline_metrics.csv"])
    )

    train = corpus.copy()
    train["record_scope"] = "oof_train"
    test = outer.copy()
    test["record_scope"] = "outer_test"
    records = pd.concat([train, test], ignore_index=True, sort=False)
    records.index = np.arange(len(records))
    entries = {str(row["case_id"]): row for row in manifest["features"]}
    _status(status_path, "pooling_approved_outer_visual_features")
    features = pool_all_features(records, entries)

    probabilities_by_fold: Dict[int, np.ndarray] = {}
    model_audit = []
    model_dir = study_dir / "outer_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    _status(status_path, "fitting_frozen_fold_models")
    for fold in OUTER_FOLDS:
        train_positions = np.flatnonzero(
            (records["record_scope"] == "oof_train").to_numpy()
            & (records["outer_fold"].astype(int).to_numpy() == fold)
        )
        eval_positions = np.flatnonzero(
            (records["record_scope"] == "outer_test").to_numpy()
            & (records["outer_fold"].astype(int).to_numpy() == fold)
        )
        fold_config = frozen["folds"][str(fold)]
        probabilities, artifacts = fit_agreement_ensemble_probabilities(
            records,
            features,
            train_positions,
            eval_positions,
        )
        probabilities_by_fold[fold] = probabilities
        model_path = model_dir / f"outer_fold_{fold}.joblib"
        joblib.dump(artifacts, model_path)
        model_audit.append(
            {
                "outer_fold": fold,
                "model_variant": "seven_member_agreement_ensemble",
                "agreement_k": int(fold_config["agreement_k"]),
                "train_segments": len(train_positions),
                "train_cases": records.iloc[train_positions]["case_id"].nunique(),
                "test_segments": len(eval_positions),
                "test_cases": records.iloc[eval_positions]["case_id"].nunique(),
                "model_sha256": file_sha256(model_path),
                "all_convergence_assertions_passed": True,
                "selection_was_oof_only": True,
            }
        )
    pd.DataFrame(model_audit).to_csv(results_dir / "model_audit.csv", index=False)

    selections: Dict[tuple[int, float], pd.DataFrame] = {}
    for fold in OUTER_FOLDS:
        fold_segments = outer[outer["outer_fold"].astype(int) == fold]
        total_frames = sum(len(case.gt) for case in cases if case.outer_fold == fold)
        budgets = {PRIMARY_BUDGET}
        secondary = frozen["folds"][str(fold)]["secondary_budget"]
        if secondary is not None:
            budgets.add(float(secondary))
        for budget in budgets:
            selections[(fold, budget)] = select_budget_rows(
                fold_segments, total_frames, budget
            )
    checks.extend(
        _reconcile_primary_masks(
            cases,
            selections,
            pd.read_csv(paths["selector_analysis/selector_budget_metrics.csv"]),
        )
    )

    primary_predictions: Dict[Any, np.ndarray] = {}
    secondary_predictions: Dict[Any, np.ndarray] = {}
    primary_ledgers = []
    primary_videos = []
    metric_rows = []
    sensitivity_rows = []
    for fold in OUTER_FOLDS:
        fold_segments = outer[outer["outer_fold"].astype(int) == fold].reset_index(drop=True)
        fold_cases = [case for case in cases if case.outer_fold == fold]
        fold_config = frozen["folds"][str(fold)]
        predictions, ledger, videos = apply_configuration(
            fold_cases,
            fold_segments,
            probabilities_by_fold[fold],
            selections[(fold, PRIMARY_BUDGET)],
            threshold=float(fold_config["threshold"]),
            rule_name=fold_config["rule_name"],
            rule_parameter=fold_config["rule_parameter"],
            agreement_k=int(fold_config["agreement_k"]),
        )
        primary_predictions.update(predictions)
        primary_ledgers.append(ledger)
        primary_videos.append(videos)
        metric_rows.append(
            _metric_row(
                "outer_fold",
                f"outer_{fold}",
                fold_cases,
                predictions,
                readout="primary_5pct",
            )
        )

        secondary_budget = fold_config["secondary_budget"]
        if secondary_budget is None:
            secondary_predictions.update(predictions)
        else:
            secondary_fold_predictions, _, _ = apply_configuration(
                fold_cases,
                fold_segments,
                probabilities_by_fold[fold],
                selections[(fold, float(secondary_budget))],
                threshold=float(fold_config["threshold"]),
                rule_name=fold_config["rule_name"],
                rule_parameter=fold_config["rule_parameter"],
                agreement_k=int(fold_config["agreement_k"]),
            )
            secondary_predictions.update(secondary_fold_predictions)
            metric_rows.append(
                _metric_row(
                    "outer_fold",
                    f"outer_{fold}",
                    fold_cases,
                    secondary_fold_predictions,
                    readout=f"secondary_{float(secondary_budget):.0%}",
                )
            )

        for threshold in SENSITIVITY_THRESHOLDS:
            sensitivity_predictions, _, _ = apply_configuration(
                fold_cases,
                fold_segments,
                probabilities_by_fold[fold],
                selections[(fold, PRIMARY_BUDGET)],
                threshold=threshold,
                rule_name=fold_config["rule_name"],
                rule_parameter=fold_config["rule_parameter"],
                agreement_k=int(fold_config["agreement_k"]),
            )
            row = _metric_row(
                "outer_fold",
                f"outer_{fold}",
                fold_cases,
                sensitivity_predictions,
                readout="threshold_sensitivity",
            )
            row["threshold"] = threshold
            sensitivity_rows.append(row)

    metric_rows.append(
        _metric_row("pooled", "pooled", cases, primary_predictions, readout="primary_5pct")
    )
    if any(frozen["folds"][str(fold)]["secondary_budget"] is not None for fold in OUTER_FOLDS):
        metric_rows.append(
            _metric_row(
                "pooled",
                "pooled",
                cases,
                secondary_predictions,
                readout="secondary_envelope_mixed_by_fold",
            )
        )
    final_metrics = pd.DataFrame(metric_rows)
    final_metrics.to_csv(results_dir / "final_metrics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(results_dir / "threshold_sensitivity.csv", index=False)

    ledger = pd.concat(primary_ledgers, ignore_index=True)
    anonymized_ledger = _anonymize_ledger(ledger)
    anonymized_ledger.to_csv(results_dir / "helped_hurt_lateral_ledger.csv", index=False)
    videos = pd.concat(primary_videos, ignore_index=True)
    videos.to_csv(results_dir / "per_video_distribution.csv", index=False)
    _class_contributions(ledger, id_to_name).to_csv(
        results_dir / "class_contributions.csv", index=False
    )
    pd.DataFrame(checks).to_csv(results_dir / "validation_checks.csv", index=False)

    pooled = final_metrics[
        (final_metrics["readout"] == "primary_5pct") & (final_metrics["scope"] == "pooled")
    ].iloc[0]
    folds = final_metrics[
        (final_metrics["readout"] == "primary_5pct")
        & (final_metrics["scope"] == "outer_fold")
    ]
    pooled_delta = {metric: float(pooled[f"delta_{metric}"]) for metric in METRICS}
    decision = primary_gate_decision(
        pooled_delta,
        folds["delta_acc"].to_numpy(dtype=float),
        videos,
    )
    decision.update(
        {
            "criteria_were_pre_registered": True,
            "single_shot_outer_evaluation": True,
            "frozen_config_digest": frozen["frozen_config_digest"],
            "primary_budget": PRIMARY_BUDGET,
            "pooled_delta": pooled_delta,
            "secondary_cannot_rescue_primary": True,
            "final_outer_attempt_for_repair_line": True,
            "failure_closes_repair_line_permanently": True,
        }
    )
    atomic_write_json(results_dir / "decision.json", decision)
    _write_findings(results_dir / "findings.md", final_metrics, decision)

    outputs = [
        path
        for path in sorted(results_dir.iterdir())
        if path.is_file()
        and path.name not in {"run_status.json", "run_complete.json", "run_failed.json"}
    ]
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "single_shot": True,
        "decision_pass": bool(decision["pass"]),
        "frozen_config_digest": frozen["frozen_config_digest"],
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    atomic_write_json(results_dir / "run_complete.json", complete)
    status_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    lock_path = study_dir / ".outer_run.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Outer evaluation is already running") from error
        try:
            execute(study_dir, args.verify_workers)
        except Exception as error:
            results = study_dir / "outer_results"
            results.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                results / "run_failed.json",
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
