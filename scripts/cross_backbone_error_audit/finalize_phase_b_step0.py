#!/usr/bin/env python3
"""Choose the pre-declared Phase-B branch using carved validation only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase_b_selection_common import (
    EPOCH_GRID,
    atomic_write_json,
    canonical_digest,
    choose_step0_branch,
    file_sha256,
    informative_breakfast_fold,
    load_json,
    select_best_checkpoint,
    verify_manifest,
    verify_source,
)
from phase_b_training_common import DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def verified_validation_rows(study: Path, dataset: str, fold: int) -> list[dict]:
    status = load_json(study / "status" / f"carved_validation_{dataset}_fold{fold}.json")
    if status.get("status") != "complete" or status.get("scope") != "carved_validation":
        raise RuntimeError(f"Incomplete carved validation: {dataset}/fold{fold}")
    if status.get("test_data_used_for_selection") is not False:
        raise RuntimeError("Validation provenance says test data entered selection")
    output = study / "validation_metrics" / dataset / f"fold{fold}.csv"
    if Path(status["output_path"]).resolve() != output.resolve():
        raise RuntimeError("Validation output namespace drift")
    if file_sha256(output) != status["output_sha256"]:
        raise RuntimeError("Validation metric hash drift")
    frame = pd.read_csv(output)
    if set(frame["scope"]) != {"carved_validation"}:
        raise RuntimeError("Non-validation row entered checkpoint selection")
    if set(frame["epoch"].astype(int)) != set(EPOCH_GRID) or len(frame) != len(EPOCH_GRID):
        raise RuntimeError("Validation checkpoint grid drift")
    return frame.to_dict("records")


def execute(study: Path) -> None:
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    paths = verify_manifest(manifest, full_hash=False)
    if not config.get("fable_approval_digest") or not config["step0_validation_inference_allowed"]:
        raise RuntimeError("Step-0 finalization is not approved")
    if config["selection_may_read_test_trajectory"] or config["phase_c_allowed"]:
        raise RuntimeError("Selection firewall drift")
    existing = study / "selection" / "step0_decision.json"
    if existing.is_file():
        decision = load_json(existing)
        if decision["decision_digest"] != canonical_digest({key: value for key, value in decision.items() if key != "decision_digest"}):
            raise RuntimeError("Existing Step-0 decision digest drift")
        print(f"already complete: {decision['selected_branch']}")
        return

    all_rows: dict[tuple[str, int], list[dict]] = {}
    breakfast_evidence: dict[int, dict] = {}
    for dataset, dataset_config in DATASETS.items():
        for fold in range(1, int(dataset_config["folds"]) + 1):
            rows = verified_validation_rows(study, dataset, fold)
            all_rows[(dataset, fold)] = rows
            if dataset == "breakfast":
                breakfast_evidence[fold] = informative_breakfast_fold(rows)
    branch = choose_step0_branch(breakfast_evidence)
    decision_payload = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selected_branch": branch,
        "branch_rule": config["step0_branch_rule"],
        "breakfast_validation_evidence": {str(key): value for key, value in sorted(breakfast_evidence.items())},
        "selection_evidence_namespace": "validation_metrics",
        "test_trajectory_read": False,
        "test_data_used_for_selection": False,
        "phase_c_allowed": False,
    }
    decision_payload["decision_digest"] = canonical_digest(decision_payload)
    atomic_write_json(existing, decision_payload)

    if branch == "A_REUSE_EXISTING_CHECKPOINTS":
        records: list[dict] = []
        for (dataset, fold), rows in sorted(all_rows.items()):
            best = select_best_checkpoint(rows)
            checkpoint = paths[f"checkpoint/{dataset}/fold{fold}/epoch{int(best['epoch'])}"]
            records.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "selected_epoch": int(best["epoch"]),
                    "validation_composite": float(best["composite"]),
                    "validation_acc": float(best["acc"]),
                    "validation_edit": float(best["edit"]),
                    "validation_f1@10": float(best["f1@10"]),
                    "validation_f1@25": float(best["f1@25"]),
                    "validation_f1@50": float(best["f1@50"]),
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "selection_source": "carved_training_fold_validation_only",
                    "carve_sha256": file_sha256(study / "carves" / dataset / f"fold{fold}" / "carved_validation.bundle"),
                    "training_regime": "existing_full_train_checkpoint",
                    "seen_video_validation_caveat": True,
                    "test_data_used_for_selection": False,
                }
            )
        frame = pd.DataFrame(records).sort_values(["dataset", "fold"])
        csv_path = study / "selection" / "selection_records.csv"
        frame.to_csv(csv_path, index=False)
        payload = {
            "branch": branch,
            "record_count": len(records),
            "records": records,
            "csv_sha256": file_sha256(csv_path),
            "seen_video_validation_caveat": True,
            "test_data_used_for_selection": False,
        }
        payload["selection_digest"] = canonical_digest(payload)
        atomic_write_json(study / "selection" / "selection_records.json", payload)
    else:
        plan = {
            "branch": branch,
            "cell_count": 13,
            "train_all_cells_from_scratch": True,
            "train_bundle_pattern": "carves/{dataset}/fold{fold}/train_remainder.bundle",
            "selection_bundle_pattern": "carves/{dataset}/fold{fold}/carved_validation.bundle",
            "official_config_otherwise_unchanged": True,
            "test_data_used_for_training_or_selection": False,
            "conditional_training_authorized": bool(config["conditional_branch_b_training_allowed"]),
        }
        plan["plan_digest"] = canonical_digest(plan)
        atomic_write_json(study / "selection" / "branch_b_retrain_plan.json", plan)
    print(branch)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
