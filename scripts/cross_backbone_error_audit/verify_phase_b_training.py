#!/usr/bin/env python3
"""Fail-closed preflight for the MS-TCN++ Phase-B production study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from phase_b_training_common import (
    OFFICIAL_SOURCE_FILES,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    verify_manifest,
    verify_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--full-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    paths = verify_manifest(manifest, full_hash=args.full_hash)
    if not config["gpu_training_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("Phase-B training is not approved")
    if config["phase_c_allowed"] or config["sealed_studies_opened"]:
        raise RuntimeError("Phase-B study unexpectedly opens Phase C or sealed studies")
    decision = load_json(paths["option0/results/phase_b_option0_decision.json"])
    if decision["decision_digest"] != config["option0_decision_digest"]:
        raise RuntimeError("Option-0 decision digest drift")
    for task in config["tasks"]:
        runtime = study / "cells" / task["dataset"] / f"fold{task['fold']}" / "runtime"
        if runtime.resolve() == Path(config["official_source"]).resolve():
            raise RuntimeError("Training runtime must not be the external official worktree")
        if not (runtime / "data").is_symlink():
            raise RuntimeError(f"Missing immutable data link: {runtime}")
        for name in OFFICIAL_SOURCE_FILES:
            expected = paths[f"official_source/{name}"]
            if file_sha256(runtime / name) != file_sha256(expected):
                raise RuntimeError(f"Runtime source drift: {task['dataset']}/fold{task['fold']}/{name}")
    result = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "full_hash": bool(args.full_hash),
        "input_manifest_digest": manifest["manifest_digest"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "task_count": len(config["tasks"]),
    }
    result["preflight_digest"] = canonical_digest(result)
    atomic_write_json(study / "status" / "preflight_complete.json", result)
    print(f"Phase-B preflight passed for {len(config['tasks'])} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
