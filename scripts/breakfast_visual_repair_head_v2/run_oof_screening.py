#!/usr/bin/env python3
"""Run OOF-only harm mitigation and envelope screening without opening outer data."""

from __future__ import annotations

import argparse
import fcntl
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from common import (
    BUDGET_CANDIDATES,
    CAP_PERCENT_CANDIDATES,
    MARGIN_CANDIDATES,
    MODEL_PROMOTION_GAIN_PP,
    OOF_HARM_FLOOR_PP,
    OUTER_FOLDS,
    PRIMARY_BUDGET,
    PROTOCOL_VERSION,
    SECONDARY_BUDGET_GAIN_PP,
    SENSITIVITY_THRESHOLDS,
    SPAN_SIZE_PERCENT_CANDIDATES,
    TAU_CANDIDATES,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    verify_self_digest,
    verify_source_provenance,
)
from core import (
    METRICS,
    aggregate_metrics,
    baseline_metrics,
    build_cases,
    crossfit_model_probabilities,
    evaluate_configuration,
    fixed_paths,
    load_ground_truth,
    pool_all_features,
    read_mapping,
    select_budget_rows,
    selection_sort_key,
    validate_segment_table,
    verify_manifest_entries,
)


RULE_ORDER = {"video_cap": 0, "incumbent_margin": 1, "large_span_guard": 2}
MODEL_ORDER = {"plain_logistic": 0, "isotonic_logistic": 1, "logistic_mlp_average": 2}


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


def _result_with_context(
    result: Mapping[str, Any], *, fold: int, stage: str, model_variant: str, budget: float
) -> Dict[str, Any]:
    return {
        "outer_fold": fold,
        "stage": stage,
        "model_variant": model_variant,
        "budget": budget,
        **dict(result),
    }


def _choose_best(rows: list[Dict[str, Any]], order: Mapping[str, int]) -> Dict[str, Any] | None:
    safe = [row for row in rows if bool(row["harm_constraint_pass"])]
    if not safe:
        return None
    return max(safe, key=lambda row: selection_sort_key(row, order))


def _harm_summary(ledger: pd.DataFrame, videos: pd.DataFrame) -> Dict[str, Any]:
    accepted = ledger[ledger["accepted"]].copy()
    hurt = accepted[accepted["outcome"] == "hurt"]
    deltas = videos["delta_acc"].to_numpy(dtype=float)
    return {
        "videos": len(videos),
        "videos_harmed": int((deltas < 0).sum()),
        "videos_below_minus_5pp": int((deltas < OOF_HARM_FLOOR_PP).sum()),
        "delta_acc_quantiles": {
            str(q): float(np.quantile(deltas, q)) for q in (0.0, 0.01, 0.05, 0.5, 0.95, 1.0)
        },
        "accepted_spans": len(accepted),
        "hurt_spans": len(hurt),
        "hurt_span_broken_frames": int(hurt["broken_frames"].sum()),
        "hurt_span_confidence_quantiles": (
            {
                str(q): float(np.quantile(hurt["proposal_probability"], q))
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            }
            if len(hurt)
            else {}
        ),
        "failure_mode_confirmed": bool(
            (deltas < OOF_HARM_FLOOR_PP).any() and len(hurt) > 0
        ),
    }


def _write_findings(
    path: Path,
    frozen: Mapping[str, Any],
    forensics: Mapping[str, Any],
    pooled_metrics: Mapping[str, float] | None,
) -> None:
    lines = [
        "# Repair-head v2 OOF screening findings",
        "",
        "Outer-test inputs remained sealed throughout this phase.",
        "No raw video identity is present in an OOF output.",
        "",
        "## OOF harm forensics",
        "",
        f"- OOF videos harmed at tau=0.5: {forensics['videos_harmed']}/{forensics['videos']}",
        f"- OOF videos below -5pp: {forensics['videos_below_minus_5pp']}",
        f"- Accepted/hurt spans: {forensics['accepted_spans']}/{forensics['hurt_spans']}",
        f"- Expected small confident harm tail confirmed: {forensics['failure_mode_confirmed']}",
        "",
        "## Frozen fold-specific configurations",
        "",
    ]
    if frozen["outer_evaluation_blocked"]:
        lines.append("**Outer evaluation is blocked:** at least one fold had no harm-safe rule.")
    else:
        lines.extend(
            [
                "| Fold | Rule | Parameter | Tau | Model | Secondary budget |",
                "|---:|---|---:|---:|---|---:|",
            ]
        )
        for fold, config in frozen["folds"].items():
            lines.append(
                f"| {fold} | {config['rule_name']} | {config['rule_parameter']} | "
                f"{config['threshold']} | {config['model_variant']} | "
                f"{config['secondary_budget']} |"
            )
        if pooled_metrics is not None:
            lines.extend(
                [
                    "",
                    (
                        "Cross-fitted OOF primary deltas: "
                        f"Acc {pooled_metrics['delta_acc']:+.3f}, "
                        f"Edit {pooled_metrics['delta_edit']:+.3f}, "
                        f"F1@25 {pooled_metrics['delta_f1@25']:+.3f}."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "Model-screen and larger-budget results are OOF-only. Outer evaluation remains single-shot and requires separate exact-digest approval.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def execute(study_dir: Path, verify_workers: int) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "screening_config.json")
    oof_manifest = load_json(study_dir / "oof_input_manifest.json")
    if metadata["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("Protocol version mismatch")
    verify_self_digest(config, "config_digest")
    verify_source_provenance(metadata["source_provenance"])

    results_dir = study_dir / "oof_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "oof_screen_complete.json").exists() or (
        results_dir / "run_failed.json"
    ).exists():
        raise FileExistsError("OOF study already terminated; use a fresh immutable directory")
    status_path = results_dir / "run_status.json"
    _status(status_path, "verifying_oof_inputs_outer_sealed")
    verification = verify_manifest_entries(oof_manifest, workers=verify_workers)
    atomic_write_json(results_dir / "oof_input_verification.json", verification)

    paths = fixed_paths(oof_manifest)
    corpus = pd.read_csv(paths["selector_analysis/repair_training_corpus_segments.csv"])
    validate_segment_table(corpus, oof_only=True)
    corpus = corpus.sort_values(
        ["outer_fold", "inner_fold", "case_id", "segment_index"], kind="mergesort"
    ).reset_index(drop=True)
    _, name_to_id = read_mapping(paths["label_mapping"])
    ground_truth = load_ground_truth(oof_manifest, name_to_id)
    cases = build_cases(corpus, ground_truth, include_inner=True)
    feature_entries = {str(row["case_id"]): row for row in oof_manifest["features"]}

    _status(status_path, "pooling_oof_visual_features", cases=len(cases), segments=len(corpus))
    features = pool_all_features(corpus, feature_entries)
    _status(status_path, "crossfitting_three_model_variants")
    probabilities, model_audit = crossfit_model_probabilities(corpus, features)
    pd.DataFrame(model_audit).to_csv(results_dir / "crossfit_model_audit.csv", index=False)
    np.savez_compressed(
        results_dir / "crossfit_probabilities.npz",
        segment_id=corpus["segment_id"].to_numpy(dtype=np.int64),
        **probabilities,
    )

    selections: Dict[tuple[int, float], pd.DataFrame] = {}
    for fold in OUTER_FOLDS:
        fold_segments = corpus[corpus["outer_fold"].astype(int) == fold]
        total_frames = sum(len(case.gt) for case in cases if case.outer_fold == fold)
        for budget in BUDGET_CANDIDATES:
            selections[(fold, budget)] = select_budget_rows(
                fold_segments, total_frames, budget
            )

    screen_rows: list[Dict[str, Any]] = []
    forensics_videos: list[pd.DataFrame] = []
    forensics_summaries: list[Dict[str, Any]] = []
    sensitivity_rows: list[Dict[str, Any]] = []
    frozen_folds: Dict[str, Any] = {}
    final_predictions: Dict[Any, np.ndarray] = {}
    final_videos: list[pd.DataFrame] = []
    outer_blocked = False

    _status(status_path, "screening_oof_harm_rules")
    for fold in OUTER_FOLDS:
        fold_mask = corpus["outer_fold"].astype(int).to_numpy() == fold
        fold_segments = corpus.loc[fold_mask].reset_index(drop=True)
        fold_cases = [case for case in cases if case.outer_fold == fold]
        fold_probabilities = {
            name: values[fold_mask] for name, values in probabilities.items()
        }

        forensic_result, _, forensic_ledger, forensic_video = evaluate_configuration(
            fold_cases,
            fold_segments,
            fold_probabilities["plain_logistic"],
            selections[(fold, PRIMARY_BUDGET)],
            threshold=0.5,
            rule_name="none",
            rule_parameter=None,
        )
        screen_rows.append(
            _result_with_context(
                forensic_result,
                fold=fold,
                stage="forensics_no_mitigation",
                model_variant="plain_logistic",
                budget=PRIMARY_BUDGET,
            )
        )
        forensic_video.insert(0, "forensics_outer_fold", fold)
        forensics_videos.append(forensic_video)
        summary = {"outer_fold": fold, **_harm_summary(forensic_ledger, forensic_video)}
        forensics_summaries.append(summary)

        harm_candidates: list[Dict[str, Any]] = []
        specs = [
            *(('video_cap', value) for value in CAP_PERCENT_CANDIDATES),
            *(('incumbent_margin', value) for value in MARGIN_CANDIDATES),
            *(('large_span_guard', value) for value in SPAN_SIZE_PERCENT_CANDIDATES),
        ]
        for rule_name, parameter in specs:
            result, _, _, _ = evaluate_configuration(
                fold_cases,
                fold_segments,
                fold_probabilities["plain_logistic"],
                selections[(fold, PRIMARY_BUDGET)],
                threshold=0.5,
                rule_name=rule_name,
                rule_parameter=parameter,
            )
            row = _result_with_context(
                result,
                fold=fold,
                stage="harm_rule_screen",
                model_variant="plain_logistic",
                budget=PRIMARY_BUDGET,
            )
            harm_candidates.append(row)
            screen_rows.append(row)
        chosen_rule = _choose_best(harm_candidates, RULE_ORDER)
        if chosen_rule is None:
            outer_blocked = True
            frozen_folds[str(fold)] = {"blocked_reason": "no_harm_safe_rule"}
            continue

        tau_candidates: list[Dict[str, Any]] = []
        for threshold in TAU_CANDIDATES:
            result, _, _, _ = evaluate_configuration(
                fold_cases,
                fold_segments,
                fold_probabilities["plain_logistic"],
                selections[(fold, PRIMARY_BUDGET)],
                threshold=threshold,
                rule_name=chosen_rule["rule_name"],
                rule_parameter=chosen_rule["rule_parameter"],
            )
            row = _result_with_context(
                result,
                fold=fold,
                stage="tau_selection",
                model_variant="plain_logistic",
                budget=PRIMARY_BUDGET,
            )
            tau_candidates.append(row)
            screen_rows.append(row)
        chosen_tau = _choose_best(tau_candidates, {chosen_rule["rule_name"]: 0})
        if chosen_tau is None:
            outer_blocked = True
            frozen_folds[str(fold)] = {"blocked_reason": "no_harm_safe_tau"}
            continue

        model_candidates: list[Dict[str, Any]] = []
        for model_variant in (
            "plain_logistic",
            "isotonic_logistic",
            "logistic_mlp_average",
        ):
            result, _, _, _ = evaluate_configuration(
                fold_cases,
                fold_segments,
                fold_probabilities[model_variant],
                selections[(fold, PRIMARY_BUDGET)],
                threshold=chosen_tau["threshold"],
                rule_name=chosen_rule["rule_name"],
                rule_parameter=chosen_rule["rule_parameter"],
            )
            row = _result_with_context(
                result,
                fold=fold,
                stage="model_screen",
                model_variant=model_variant,
                budget=PRIMARY_BUDGET,
            )
            model_candidates.append(row)
            screen_rows.append(row)
        plain = next(row for row in model_candidates if row["model_variant"] == "plain_logistic")
        promotable = [
            row
            for row in model_candidates
            if row["model_variant"] != "plain_logistic"
            and row["harm_constraint_pass"]
            and row["delta_acc"] - plain["delta_acc"] >= MODEL_PROMOTION_GAIN_PP
        ]
        chosen_model = (
            max(
                promotable,
                key=lambda row: (
                    float(row["delta_acc"]),
                    float(row["delta_f1@25"]),
                    float(row["delta_edit"]),
                    float(row["worst_video_delta_acc"]),
                    -MODEL_ORDER[row["model_variant"]],
                ),
            )
            if promotable
            else plain
        )

        budget_candidates: list[Dict[str, Any]] = []
        for budget in BUDGET_CANDIDATES:
            result, predictions, ledger, video = evaluate_configuration(
                fold_cases,
                fold_segments,
                fold_probabilities[chosen_model["model_variant"]],
                selections[(fold, budget)],
                threshold=chosen_tau["threshold"],
                rule_name=chosen_rule["rule_name"],
                rule_parameter=chosen_rule["rule_parameter"],
            )
            row = _result_with_context(
                result,
                fold=fold,
                stage="budget_envelope",
                model_variant=chosen_model["model_variant"],
                budget=budget,
            )
            row["_predictions"] = predictions
            row["_ledger"] = ledger
            row["_videos"] = video
            budget_candidates.append(row)
            screen_rows.append({key: value for key, value in row.items() if not key.startswith("_")})
        primary = next(row for row in budget_candidates if row["budget"] == PRIMARY_BUDGET)
        secondary_eligible = [
            row
            for row in budget_candidates
            if row["budget"] > PRIMARY_BUDGET
            and row["harm_constraint_pass"]
            and row["delta_acc"] - primary["delta_acc"] >= SECONDARY_BUDGET_GAIN_PP
        ]
        secondary = (
            max(secondary_eligible, key=lambda row: (row["delta_acc"], -row["budget"]))
            if secondary_eligible
            else None
        )

        for threshold in SENSITIVITY_THRESHOLDS:
            result, _, _, _ = evaluate_configuration(
                fold_cases,
                fold_segments,
                fold_probabilities[chosen_model["model_variant"]],
                selections[(fold, PRIMARY_BUDGET)],
                threshold=threshold,
                rule_name=chosen_rule["rule_name"],
                rule_parameter=chosen_rule["rule_parameter"],
            )
            sensitivity_rows.append(
                _result_with_context(
                    result,
                    fold=fold,
                    stage="threshold_sensitivity",
                    model_variant=chosen_model["model_variant"],
                    budget=PRIMARY_BUDGET,
                )
            )

        frozen_folds[str(fold)] = {
            "rule_name": chosen_rule["rule_name"],
            "rule_parameter": float(chosen_rule["rule_parameter"]),
            "threshold": float(chosen_tau["threshold"]),
            "model_variant": chosen_model["model_variant"],
            "primary_budget": PRIMARY_BUDGET,
            "secondary_budget": None if secondary is None else float(secondary["budget"]),
            "oof_primary_evidence": {
                metric: float(primary[metric])
                for metric in (
                    "delta_acc",
                    "delta_edit",
                    "delta_f1@25",
                    "worst_video_delta_acc",
                )
            },
            "model_promotion_gain_over_plain_pp": float(
                chosen_model["delta_acc"] - plain["delta_acc"]
            ),
            "secondary_gain_over_5pct_pp": (
                None if secondary is None else float(secondary["delta_acc"] - primary["delta_acc"])
            ),
        }
        final_predictions.update(primary["_predictions"])
        final_videos.append(primary["_videos"])

    pd.DataFrame(screen_rows).to_csv(results_dir / "oof_screening_candidates.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(results_dir / "oof_threshold_sensitivity.csv", index=False)
    anonymized_forensics = pd.concat(forensics_videos, ignore_index=True)
    anonymized_forensics.to_csv(results_dir / "oof_forensics_video_distribution.csv", index=False)
    aggregate_forensics = {
        "per_fold": forensics_summaries,
        "videos": int(sum(row["videos"] for row in forensics_summaries)),
        "videos_harmed": int(sum(row["videos_harmed"] for row in forensics_summaries)),
        "videos_below_minus_5pp": int(
            sum(row["videos_below_minus_5pp"] for row in forensics_summaries)
        ),
        "accepted_spans": int(sum(row["accepted_spans"] for row in forensics_summaries)),
        "hurt_spans": int(sum(row["hurt_spans"] for row in forensics_summaries)),
        "failure_mode_confirmed": bool(
            any(row["failure_mode_confirmed"] for row in forensics_summaries)
        ),
        "raw_case_ids_present": False,
    }
    atomic_write_json(results_dir / "oof_forensics_summary.json", aggregate_forensics)

    pooled_evidence = None
    if not outer_blocked:
        base = baseline_metrics(cases)
        repaired = aggregate_metrics(cases, final_predictions)
        pooled_evidence = {
            **{f"baseline_{metric}": float(base[metric]) for metric in METRICS},
            **{f"repair_{metric}": float(repaired[metric]) for metric in METRICS},
            **{f"delta_{metric}": float(repaired[metric] - base[metric]) for metric in METRICS},
            "worst_video_delta_acc": float(
                pd.concat(final_videos, ignore_index=True)["delta_acc"].min()
            ),
        }
    frozen = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_was_oof_only": True,
        "selection_scope": "fold_specific_matching_outer_train_oof_only",
        "outer_test_opened": False,
        "outer_evaluation_blocked": outer_blocked,
        "folds": frozen_folds,
        "pooled_crossfit_oof_evidence": pooled_evidence,
        "screening_config_digest": config["config_digest"],
        "oof_input_manifest_digest": oof_manifest["manifest_digest"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "contains_raw_video_ids": False,
    }
    frozen["frozen_config_digest"] = canonical_digest(frozen)
    atomic_write_json(results_dir / "frozen_outer_config.json", frozen)
    _write_findings(
        results_dir / "findings.md", frozen, aggregate_forensics, pooled_evidence
    )

    outputs = [
        path
        for path in sorted(results_dir.iterdir())
        if path.is_file()
        and path.name not in {"run_status.json", "oof_screen_complete.json", "run_failed.json"}
    ]
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "outer_inputs_opened": False,
        "outer_evaluation_blocked": outer_blocked,
        "frozen_config_digest": frozen["frozen_config_digest"],
        "output_sha256": {
            path.name: file_sha256(path) for path in outputs
        },
    }
    atomic_write_json(results_dir / "oof_screen_complete.json", complete)
    status_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    lock_path = study_dir / ".oof_run.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("OOF screening is already running") from error
        try:
            execute(study_dir, args.verify_workers)
        except Exception as error:
            results = study_dir / "oof_results"
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
