#!/usr/bin/env python3
"""Run the sealed OOF agreement-mechanism screen for repair-head v3."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from common import (
    AGREEMENT_K_CANDIDATES,
    AGREEMENT_MEMBER_NAMES,
    BUDGET_CANDIDATES,
    CAP_PERCENT_CANDIDATES,
    EXPECTED_V2_REFERENCE_HURT_SPANS,
    MARGIN_CANDIDATES,
    MECHANISM_MINIMUM_VETO_RATIO,
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
    crossfit_agreement_probabilities,
    evaluate_configuration,
    fixed_paths,
    load_ground_truth,
    pool_all_features,
    read_mapping,
    reference_action_support,
    select_budget_rows,
    validate_segment_table,
    verify_manifest_entries,
)


RULE_ORDER = {"none": 0, "video_cap": 1, "incumbent_margin": 2, "large_span_guard": 3}


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


def _with_context(
    result: Mapping[str, Any], *, fold: int, stage: str, budget: float
) -> Dict[str, Any]:
    return {
        "outer_fold": fold,
        "stage": stage,
        "budget": budget,
        **dict(result),
    }


def _candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["delta_acc"]),
        float(row["delta_f1@25"]),
        float(row["delta_edit"]),
        float(row["worst_video_delta_acc"]),
        int(str(row["rule_name"]) == "none"),
        -int(row["agreement_k"]),
        -float(row["threshold"]),
        -RULE_ORDER[str(row["rule_name"])],
        -float(row.get("rule_parameter") or 0.0),
    )


def _mechanism_summary(
    support: pd.DataFrame,
    *,
    tail_video_hashes: set[str],
    agreement_k: int,
    threshold: float,
) -> Dict[str, Any]:
    helpful = support[support["reference_outcome"] == "helped"]
    harmful = support[support["reference_outcome"] == "hurt"]
    if len(helpful) == 0 or len(harmful) == 0:
        raise RuntimeError("Reference actions require both helpful and harmful outcomes")
    helpful_rate = float(helpful["reference_action_vetoed"].mean())
    harmful_rate = float(harmful["reference_action_vetoed"].mean())
    infinite = helpful_rate == 0.0 and harmful_rate > 0.0
    undefined = helpful_rate == 0.0 and harmful_rate == 0.0
    ratio = None if helpful_rate == 0.0 else harmful_rate / helpful_rate
    passes = bool(
        infinite
        or (
            not undefined
            and ratio is not None
            and ratio >= MECHANISM_MINIMUM_VETO_RATIO
        )
    )
    tail_harmful = harmful[harmful["video_key_sha256"].isin(tail_video_hashes)]
    return {
        "agreement_k": int(agreement_k),
        "threshold": float(threshold),
        "helpful_reference_actions": len(helpful),
        "harmful_reference_actions": len(harmful),
        "helpful_actions_vetoed": int(helpful["reference_action_vetoed"].sum()),
        "harmful_actions_vetoed": int(harmful["reference_action_vetoed"].sum()),
        "helpful_veto_rate": helpful_rate,
        "harmful_veto_rate": harmful_rate,
        "harmful_to_helpful_veto_ratio": ratio,
        "veto_ratio_is_infinite": infinite,
        "veto_ratio_is_undefined": undefined,
        "tail_harmful_reference_actions": len(tail_harmful),
        "tail_harmful_actions_vetoed": int(
            tail_harmful["reference_action_vetoed"].sum()
        ),
        "tail_harmful_veto_rate": (
            None
            if len(tail_harmful) == 0
            else float(tail_harmful["reference_action_vetoed"].mean())
        ),
        "minimum_ratio": MECHANISM_MINIMUM_VETO_RATIO,
        "mechanism_gate_pass": passes,
    }


def _best_mechanism_row(rows: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    passing = [row for row in rows if row["mechanism_gate_pass"]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda row: (
            math.inf
            if row["veto_ratio_is_infinite"]
            else float(row["harmful_to_helpful_veto_ratio"]),
            float(row["harmful_veto_rate"]),
            -float(row["helpful_veto_rate"]),
            -int(row["agreement_k"]),
            -float(row["threshold"]),
        ),
    )


def _write_findings(
    path: Path,
    mechanism: Mapping[str, Any],
    frozen: Mapping[str, Any],
    pooled: Mapping[str, float] | None,
) -> None:
    lines = [
        "# Repair-head v3 OOF agreement findings",
        "",
        "This is the final outer attempt for the repair line and tests ensemble agreement, not another confidence threshold.",
        "Outer-test inputs remained sealed; no raw video identity is present in any OOF output.",
        "",
        "## Mechanism gate",
        "",
        f"- Reproduced harmful v2 reference actions: {mechanism['reference_hurt_spans']}",
        f"- Gate verdict: **{'PASS' if mechanism['pass'] else 'NO-GO'}**",
    ]
    best = mechanism.get("best_passing_candidate")
    if best is not None:
        ratio = (
            "infinite"
            if best["veto_ratio_is_infinite"]
            else f"{best['harmful_to_helpful_veto_ratio']:.3f}x"
        )
        lines.extend(
            [
                f"- Best passing readout: k={best['agreement_k']}, tau={best['threshold']}",
                f"- Harmful veto: {best['harmful_veto_rate']:.1%}; helpful veto: {best['helpful_veto_rate']:.1%}; ratio: {ratio}",
                f"- Tail harmful veto: {best['tail_harmful_veto_rate']}",
            ]
        )
    lines.extend(["", "## Frozen fold-specific configurations", ""])
    if frozen["outer_evaluation_blocked"]:
        lines.append(
            f"**Outer evaluation is blocked:** {frozen['blocked_reason']}. The repair line closes without another outer run."
        )
    else:
        lines.extend(
            [
                "| Fold | k | Tau | Overlay | Parameter | Secondary budget |",
                "|---:|---:|---:|---|---:|---:|",
            ]
        )
        for fold, config in frozen["folds"].items():
            lines.append(
                f"| {fold} | {config['agreement_k']} | {config['threshold']} | "
                f"{config['rule_name']} | {config['rule_parameter']} | "
                f"{config['secondary_budget']} |"
            )
        if pooled is not None:
            lines.extend(
                [
                    "",
                    (
                        "Cross-fitted OOF primary deltas: "
                        f"Acc {pooled['delta_acc']:+.3f}, Edit {pooled['delta_edit']:+.3f}, "
                        f"F1@25 {pooled['delta_f1@25']:+.3f}."
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "Outer evaluation remains single-shot and requires Fable's separate exact-digest approval.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def _finalize(
    results_dir: Path,
    status_path: Path,
    *,
    frozen: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    pooled: Mapping[str, float] | None,
) -> None:
    atomic_write_json(results_dir / "frozen_outer_config.json", frozen)
    _write_findings(results_dir / "findings.md", mechanism, frozen, pooled)
    outputs = [
        path
        for path in sorted(results_dir.iterdir())
        if path.is_file()
        and path.name
        not in {"run_status.json", "oof_screen_complete.json", "run_failed.json"}
    ]
    atomic_write_json(
        results_dir / "oof_screen_complete.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "outer_inputs_opened": False,
            "mechanism_gate_pass": bool(mechanism["pass"]),
            "outer_evaluation_blocked": bool(frozen["outer_evaluation_blocked"]),
            "frozen_config_digest": frozen["frozen_config_digest"],
            "output_sha256": {path.name: file_sha256(path) for path in outputs},
        },
    )
    status_path.unlink(missing_ok=True)


def execute(study_dir: Path, verify_workers: int) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "screening_config.json")
    manifest = load_json(study_dir / "oof_input_manifest.json")
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
    verification = verify_manifest_entries(manifest, workers=verify_workers)
    atomic_write_json(results_dir / "oof_input_verification.json", verification)

    paths = fixed_paths(manifest)
    corpus = pd.read_csv(paths["selector_analysis/repair_training_corpus_segments.csv"])
    validate_segment_table(corpus, oof_only=True)
    corpus = corpus.sort_values(
        ["outer_fold", "inner_fold", "case_id", "segment_index"], kind="mergesort"
    ).reset_index(drop=True)
    _, name_to_id = read_mapping(paths["label_mapping"])
    ground_truth = load_ground_truth(manifest, name_to_id)
    cases = build_cases(corpus, ground_truth, include_inner=True)
    feature_entries = {str(row["case_id"]): row for row in manifest["features"]}

    _status(status_path, "pooling_oof_visual_features", cases=len(cases), segments=len(corpus))
    features = pool_all_features(corpus, feature_entries)
    _status(status_path, "crossfitting_seven_member_agreement_ensemble")
    probabilities, model_audit = crossfit_agreement_probabilities(corpus, features)
    pd.DataFrame(model_audit).to_csv(results_dir / "crossfit_model_audit.csv", index=False)
    np.savez_compressed(
        results_dir / "crossfit_agreement_probabilities.npz",
        segment_id=corpus["segment_id"].to_numpy(dtype=np.int64),
        member_names=np.asarray(AGREEMENT_MEMBER_NAMES, dtype="U40"),
        probabilities=probabilities,
    )

    selections: Dict[tuple[int, float], pd.DataFrame] = {}
    for fold in OUTER_FOLDS:
        fold_segments = corpus[corpus["outer_fold"].astype(int) == fold]
        total_frames = sum(len(case.gt) for case in cases if case.outer_fold == fold)
        for budget in BUDGET_CANDIDATES:
            selections[(fold, budget)] = select_budget_rows(
                fold_segments, total_frames, budget
            )

    _status(status_path, "reproducing_v2_reference_actions")
    reference_ledgers: list[pd.DataFrame] = []
    reference_videos: list[pd.DataFrame] = []
    reference_rows: list[Dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        mask = corpus["outer_fold"].astype(int).to_numpy() == fold
        fold_segments = corpus.loc[mask].reset_index(drop=True)
        fold_cases = [case for case in cases if case.outer_fold == fold]
        result, _, ledger, videos = evaluate_configuration(
            fold_cases,
            fold_segments,
            probabilities[mask, 0, :],
            selections[(fold, PRIMARY_BUDGET)],
            threshold=0.5,
            rule_name="none",
            rule_parameter=None,
        )
        reference_rows.append(
            _with_context(result, fold=fold, stage="v2_reference_actions", budget=PRIMARY_BUDGET)
        )
        reference_ledgers.append(ledger)
        reference_videos.append(videos)
    reference_ledger = pd.concat(reference_ledgers, ignore_index=True)
    reference_video = pd.concat(reference_videos, ignore_index=True)
    reference_hurt = reference_ledger[
        reference_ledger["accepted"] & (reference_ledger["outcome"] == "hurt")
    ]
    if len(reference_hurt) != EXPECTED_V2_REFERENCE_HURT_SPANS:
        raise RuntimeError(
            "v2 reference-action drift: expected "
            f"{EXPECTED_V2_REFERENCE_HURT_SPANS} hurt spans, got {len(reference_hurt)}"
        )
    tail_hashes = set(
        reference_video.loc[
            reference_video["delta_acc"] < OOF_HARM_FLOOR_PP, "video_key_sha256"
        ].astype(str)
    )

    _status(status_path, "testing_agreement_veto_mechanism")
    mechanism_rows: list[Dict[str, Any]] = []
    support_outputs: list[pd.DataFrame] = []
    for agreement_k in AGREEMENT_K_CANDIDATES:
        for threshold in TAU_CANDIDATES:
            fold_support: list[pd.DataFrame] = []
            for fold in OUTER_FOLDS:
                mask = corpus["outer_fold"].astype(int).to_numpy() == fold
                fold_segments = corpus.loc[mask].reset_index(drop=True)
                fold_reference = reference_ledgers[fold - 1]
                support = reference_action_support(
                    fold_segments,
                    probabilities[mask],
                    fold_reference,
                    agreement_k=agreement_k,
                    threshold=threshold,
                )
                fold_support.append(support)
            pooled_support = pd.concat(fold_support, ignore_index=True)
            summary = _mechanism_summary(
                pooled_support,
                tail_video_hashes=tail_hashes,
                agreement_k=agreement_k,
                threshold=threshold,
            )
            mechanism_rows.append(summary)
            support_outputs.append(pooled_support)
    pd.DataFrame(reference_rows).to_csv(
        results_dir / "v2_reference_action_metrics.csv", index=False
    )
    mechanism_frame = pd.DataFrame(mechanism_rows)
    mechanism_frame.to_csv(results_dir / "mechanism_gate_candidates.csv", index=False)
    support_frame = pd.concat(support_outputs, ignore_index=True)
    support_frame["segment_key_sha256"] = [
        hashlib.sha256(f"outer={fold}|segment={segment}".encode()).hexdigest()
        for fold, segment in zip(support_frame["outer_fold"], support_frame["segment_id"])
    ]
    support_frame.drop(columns=["segment_id"], inplace=True)
    support_frame.to_csv(results_dir / "mechanism_action_support.csv", index=False)

    best_mechanism = _best_mechanism_row(mechanism_rows)
    mechanism = {
        "pass": best_mechanism is not None,
        "minimum_harmful_to_helpful_veto_ratio": MECHANISM_MINIMUM_VETO_RATIO,
        "reference_hurt_spans": len(reference_hurt),
        "reference_helped_spans": int(
            (reference_ledger["accepted"] & (reference_ledger["outcome"] == "helped")).sum()
        ),
        "reference_tail_videos_below_minus_5pp": len(tail_hashes),
        "best_passing_candidate": best_mechanism,
    }
    atomic_write_json(results_dir / "mechanism_gate_decision.json", mechanism)

    frozen_base: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "governance": config["governance"],
        "selection_was_oof_only": True,
        "selection_scope": "fold_specific_matching_outer_train_oof_only",
        "outer_test_opened": False,
        "ensemble_member_order": list(AGREEMENT_MEMBER_NAMES),
        "mechanism_gate": mechanism,
        "screening_config_digest": config["config_digest"],
        "oof_input_manifest_digest": manifest["manifest_digest"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "contains_raw_video_ids": False,
    }
    if best_mechanism is None:
        frozen = {
            **frozen_base,
            "outer_evaluation_blocked": True,
            "blocked_reason": "agreement_veto_selectivity_below_2x",
            "folds": {},
            "pooled_crossfit_oof_evidence": None,
        }
        frozen["frozen_config_digest"] = canonical_digest(frozen)
        _finalize(
            results_dir,
            status_path,
            frozen=frozen,
            mechanism=mechanism,
            pooled=None,
        )
        return

    _status(status_path, "selecting_fold_specific_agreement_configurations")
    overlay_specs: list[tuple[str, float | None]] = [
        ("none", None),
        *(("video_cap", value) for value in CAP_PERCENT_CANDIDATES),
        *(("incumbent_margin", value) for value in MARGIN_CANDIDATES),
        *(("large_span_guard", value) for value in SPAN_SIZE_PERCENT_CANDIDATES),
    ]
    candidate_rows: list[Dict[str, Any]] = []
    sensitivity_rows: list[Dict[str, Any]] = []
    frozen_folds: Dict[str, Any] = {}
    final_predictions: Dict[Any, np.ndarray] = {}
    final_videos: list[pd.DataFrame] = []
    outer_blocked = False
    for fold in OUTER_FOLDS:
        mask = corpus["outer_fold"].astype(int).to_numpy() == fold
        fold_segments = corpus.loc[mask].reset_index(drop=True)
        fold_cases = [case for case in cases if case.outer_fold == fold]
        fold_probabilities = probabilities[mask]
        fold_candidates: list[Dict[str, Any]] = []
        for agreement_k in AGREEMENT_K_CANDIDATES:
            for threshold in TAU_CANDIDATES:
                for rule_name, rule_parameter in overlay_specs:
                    result, _, _, _ = evaluate_configuration(
                        fold_cases,
                        fold_segments,
                        fold_probabilities,
                        selections[(fold, PRIMARY_BUDGET)],
                        threshold=threshold,
                        rule_name=rule_name,
                        rule_parameter=rule_parameter,
                        agreement_k=agreement_k,
                    )
                    row = _with_context(
                        result,
                        fold=fold,
                        stage="agreement_configuration_selection",
                        budget=PRIMARY_BUDGET,
                    )
                    fold_candidates.append(row)
                    candidate_rows.append(row)
        safe = [row for row in fold_candidates if row["harm_constraint_pass"]]
        if not safe:
            outer_blocked = True
            frozen_folds[str(fold)] = {"blocked_reason": "no_harm_safe_agreement_configuration"}
            continue
        chosen = max(safe, key=_candidate_key)

        budget_rows: list[Dict[str, Any]] = []
        for budget in BUDGET_CANDIDATES:
            result, predictions, ledger, videos = evaluate_configuration(
                fold_cases,
                fold_segments,
                fold_probabilities,
                selections[(fold, budget)],
                threshold=float(chosen["threshold"]),
                rule_name=str(chosen["rule_name"]),
                rule_parameter=chosen["rule_parameter"],
                agreement_k=int(chosen["agreement_k"]),
            )
            row = _with_context(
                result, fold=fold, stage="budget_envelope", budget=budget
            )
            row["_predictions"] = predictions
            row["_ledger"] = ledger
            row["_videos"] = videos
            budget_rows.append(row)
            candidate_rows.append(
                {key: value for key, value in row.items() if not key.startswith("_")}
            )
        primary = next(row for row in budget_rows if row["budget"] == PRIMARY_BUDGET)
        secondary_candidates = [
            row
            for row in budget_rows
            if row["budget"] > PRIMARY_BUDGET
            and row["harm_constraint_pass"]
            and row["delta_acc"] - primary["delta_acc"] >= SECONDARY_BUDGET_GAIN_PP
        ]
        secondary = (
            max(secondary_candidates, key=lambda row: (row["delta_acc"], -row["budget"]))
            if secondary_candidates
            else None
        )
        for threshold in SENSITIVITY_THRESHOLDS:
            result, _, _, _ = evaluate_configuration(
                fold_cases,
                fold_segments,
                fold_probabilities,
                selections[(fold, PRIMARY_BUDGET)],
                threshold=threshold,
                rule_name=str(chosen["rule_name"]),
                rule_parameter=chosen["rule_parameter"],
                agreement_k=int(chosen["agreement_k"]),
            )
            sensitivity_rows.append(
                _with_context(
                    result,
                    fold=fold,
                    stage="threshold_sensitivity_descriptive_only",
                    budget=PRIMARY_BUDGET,
                )
            )
        frozen_folds[str(fold)] = {
            "agreement_k": int(chosen["agreement_k"]),
            "threshold": float(chosen["threshold"]),
            "rule_name": str(chosen["rule_name"]),
            "rule_parameter": (
                None if chosen["rule_parameter"] is None else float(chosen["rule_parameter"])
            ),
            "primary_budget": PRIMARY_BUDGET,
            "secondary_budget": None if secondary is None else float(secondary["budget"]),
            "oof_primary_evidence": {
                metric: float(primary[metric])
                for metric in ("delta_acc", "delta_edit", "delta_f1@25", "worst_video_delta_acc")
            },
            "secondary_gain_over_5pct_pp": (
                None
                if secondary is None
                else float(secondary["delta_acc"] - primary["delta_acc"])
            ),
        }
        final_predictions.update(primary["_predictions"])
        final_videos.append(primary["_videos"])

    pd.DataFrame(candidate_rows).to_csv(
        results_dir / "oof_agreement_selection_candidates.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        results_dir / "oof_threshold_sensitivity.csv", index=False
    )
    pooled = None
    blocked_reason = None
    if outer_blocked:
        blocked_reason = "at_least_one_fold_has_no_harm_safe_configuration"
    else:
        base = baseline_metrics(cases)
        repaired = aggregate_metrics(cases, final_predictions)
        pooled = {
            **{f"baseline_{metric}": float(base[metric]) for metric in METRICS},
            **{f"repair_{metric}": float(repaired[metric]) for metric in METRICS},
            **{f"delta_{metric}": float(repaired[metric] - base[metric]) for metric in METRICS},
            "worst_video_delta_acc": float(
                pd.concat(final_videos, ignore_index=True)["delta_acc"].min()
            ),
        }
    frozen = {
        **frozen_base,
        "outer_evaluation_blocked": outer_blocked,
        "blocked_reason": blocked_reason,
        "folds": frozen_folds,
        "pooled_crossfit_oof_evidence": pooled,
    }
    frozen["frozen_config_digest"] = canonical_digest(frozen)
    _finalize(
        results_dir,
        status_path,
        frozen=frozen,
        mechanism=mechanism,
        pooled=pooled,
    )


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
