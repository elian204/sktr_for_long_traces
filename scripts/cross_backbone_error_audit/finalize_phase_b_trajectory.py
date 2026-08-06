#!/usr/bin/env python3
"""Consolidate the firewalled descriptive per-epoch test trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase_b_selection_common import EPOCH_GRID, atomic_write_json, file_sha256, load_json, verify_manifest, verify_source
from phase_b_training_common import DATASETS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    study = parser.parse_args().study_dir.resolve()
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    verify_source(metadata["source_provenance"])
    verify_manifest(load_json(study / "input_manifest.json"), full_hash=False)
    if not config.get("fable_approval_digest") or not config["test_trajectory_inference_allowed"]:
        raise RuntimeError("Test trajectory finalization is not approved")
    rows: list[pd.DataFrame] = []
    for dataset, dataset_config in DATASETS.items():
        for fold in range(1, int(dataset_config["folds"]) + 1):
            status = load_json(study / "status" / f"test_trajectory_{dataset}_fold{fold}.json")
            path = study / "test_trajectory" / dataset / f"fold{fold}.csv"
            if status.get("status") != "complete" or status.get("test_data_used_for_selection") is not False:
                raise RuntimeError(f"Invalid trajectory completion: {dataset}/fold{fold}")
            if file_sha256(path) != status["output_sha256"]:
                raise RuntimeError("Trajectory metric hash drift")
            frame = pd.read_csv(path)
            if set(frame["epoch"].astype(int)) != set(EPOCH_GRID) or set(frame["scope"]) != {"test_trajectory"}:
                raise RuntimeError("Trajectory grid/scope drift")
            rows.append(frame)
    output = study / "trajectory" / "per_epoch_test_metrics.csv"
    pd.concat(rows, ignore_index=True).sort_values(["dataset", "fold", "epoch"]).to_csv(output, index=False)
    completion = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cell_count": 13,
        "epoch_grid": list(EPOCH_GRID),
        "descriptive_appendix_only": True,
        "used_for_selection": False,
        "output_path": str(output),
        "output_sha256": file_sha256(output),
    }
    atomic_write_json(study / "trajectory" / "trajectory_complete.json", completion)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
