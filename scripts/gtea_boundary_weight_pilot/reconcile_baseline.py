#!/usr/bin/env python3
"""Gate treatment training on reproduction of the locked D100/E10000 baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from common import (
    FINAL_EPOCH,
    RECONCILIATION_FRAME_DISAGREEMENT_TOLERANCE,
    RECONCILIATION_METRIC_TOLERANCE,
    atomic_write_json,
    export_dir,
    load_json,
    load_task,
    read_bundle,
    variant_id,
    verify_export,
    verify_source_digest,
)
from metrics import evaluate_export
from run_variant import validate_completed_task


METRIC_KEYS = ("acc", "edit", "f1@10", "f1@25", "f1@50")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()

    study_dir = args.study_dir.resolve()
    metadata = load_json(study_dir / "study_metadata.json")
    diffact_root = Path(metadata["diffact_root"])
    verify_source_digest(metadata, diffact_root)
    baseline_task = load_task(study_dir, variant_id(0.1))
    if validate_completed_task(baseline_task, metadata) is None:
        raise RuntimeError("In-harness 0.1 baseline is incomplete; cannot reconcile")
    case_ids = read_bundle(Path(baseline_task["test_manifest"]))
    new_export = export_dir(baseline_task, FINAL_EPOCH, 0)
    locked_export = Path(metadata["locked_reference_export"])
    verify_export(new_export, case_ids)
    verify_export(locked_export, case_ids)

    new_summaries, _, _, new_streams, _ = evaluate_export(
        data_root=Path(metadata["data_root"]),
        case_ids=case_ids,
        output_dir=new_export,
    )
    locked_summaries, _, _, locked_streams, _ = evaluate_export(
        data_root=Path(metadata["data_root"]),
        case_ids=case_ids,
        output_dir=locked_export,
    )
    stream_results: Dict[str, Any] = {}
    passed = True
    for stream in ("pre_purge", "post_purge"):
        metric_differences = {
            key: float(new_summaries[stream][key] - locked_summaries[stream][key])
            for key in METRIC_KEYS
        }
        metric_pass = all(
            abs(value) <= RECONCILIATION_METRIC_TOLERANCE
            for value in metric_differences.values()
        )
        disagreements = 0
        frames = 0
        per_case_disagreement: Dict[str, float] = {}
        for case_id in case_ids:
            new_prediction = np.asarray(new_streams[stream][case_id])
            locked_prediction = np.asarray(locked_streams[stream][case_id])
            if new_prediction.shape != locked_prediction.shape:
                raise ValueError(f"{case_id}: locked/new prediction shapes differ")
            case_disagreement = int((new_prediction != locked_prediction).sum())
            disagreements += case_disagreement
            frames += len(new_prediction)
            per_case_disagreement[case_id] = case_disagreement / len(new_prediction)
        disagreement_fraction = disagreements / frames if frames else 0.0
        disagreement_pass = (
            disagreement_fraction <= RECONCILIATION_FRAME_DISAGREEMENT_TOLERANCE
        )
        stream_pass = metric_pass and disagreement_pass
        passed = passed and stream_pass
        stream_results[stream] = {
            "passed": stream_pass,
            "new_metrics": {key: new_summaries[stream][key] for key in METRIC_KEYS},
            "locked_metrics": {
                key: locked_summaries[stream][key] for key in METRIC_KEYS
            },
            "metric_differences_points": metric_differences,
            "metric_tolerance_points": RECONCILIATION_METRIC_TOLERANCE,
            "metric_passed": metric_pass,
            "frame_count": frames,
            "disagreement_frames": disagreements,
            "frame_disagreement_fraction": disagreement_fraction,
            "frame_disagreement_tolerance_fraction": (
                RECONCILIATION_FRAME_DISAGREEMENT_TOLERANCE
            ),
            "frame_disagreement_passed": disagreement_pass,
            "per_case_disagreement_fraction": per_case_disagreement,
        }

    payload = {
        "protocol_version": metadata["protocol_version"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "in_harness_task": baseline_task["task_id"],
        "in_harness_export": str(new_export),
        "locked_export": str(locked_export),
        "checkpoint_epoch": FINAL_EPOCH,
        "inference_seed": 0,
        "streams": stream_results,
        "failure_action": "stop_before_treatment_training",
        "source_digest": metadata["source_provenance"]["source_digest"],
    }
    output_path = study_dir / "baseline_reconciliation.json"
    atomic_write_json(output_path, payload)
    atomic_write_json(
        study_dir / "state" / "baseline_reconciliation.json",
        {
            "task_id": "baseline_reconciliation",
            "task_type": "hard_dependency_gate",
            "status": "complete" if passed else "failed",
            "completed_utc": payload["completed_utc"],
            "passed": passed,
            "output_path": str(output_path),
        },
    )
    if not passed:
        raise RuntimeError(
            "Fresh 0.1 baseline did not reconcile with locked D100/E10000 export; "
            "treatment queue intentionally stopped. See baseline_reconciliation.json."
        )
    print(f"PASS baseline reconciliation: {output_path}", flush=True)


if __name__ == "__main__":
    main()

