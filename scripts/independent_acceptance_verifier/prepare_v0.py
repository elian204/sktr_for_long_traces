#!/usr/bin/env python3
"""Prepare the immutable OOF-only V0 candidate-corpus review study."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    CORPUS_RELATIVE,
    DEFAULT_SELECTOR_STUDY,
    DEFAULT_STUDY_DIR,
    DEFAULT_VISUAL_OOF_STUDY,
    INNER_FOLDS,
    OUTER_FOLDS,
    PROTOCOL_VERSION,
    SELECTOR_ANALYSIS_RELATIVE,
    VISUAL_PROBABILITY_RELATIVE,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    source_provenance,
    write_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--selector-study", type=Path, default=DEFAULT_SELECTOR_STUDY)
    parser.add_argument("--visual-oof-study", type=Path, default=DEFAULT_VISUAL_OOF_STUDY)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {role}: {path}")
    rows.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    )


def build_manifest(selector: Path, visual: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    add_input(rows, "selector/oof_segment_corpus", selector / CORPUS_RELATIVE)
    add_input(rows, "selector/repair_corpus_schema", selector / "repair_corpus_schema.json")
    add_input(rows, "selector/study_metadata", selector / "study_metadata.json")
    add_input(rows, "selector/analysis_complete", selector / SELECTOR_ANALYSIS_RELATIVE / "analysis_complete.json")
    add_input(rows, "selector/scale_decision", selector / SELECTOR_ANALYSIS_RELATIVE / "scale_decision.json")
    add_input(rows, "visual_oof/probabilities", visual / VISUAL_PROBABILITY_RELATIVE)
    add_input(rows, "visual_oof/screen_complete", visual / "oof_results" / "oof_screen_complete.json")
    add_input(rows, "visual_oof/input_verification", visual / "oof_results" / "oof_input_verification.json")
    add_input(rows, "visual_oof/study_metadata", visual / "study_metadata.json")
    for outer in OUTER_FOLDS:
        for inner in INNER_FOLDS:
            prefix = selector / "align" / f"outer_fold_{outer}" / f"inner_fold_{inner}"
            add_input(rows, f"ground_truth/outer{outer}/inner{inner}/rows", prefix / "ground_truth.csv")
            add_input(rows, f"ground_truth/outer{outer}/inner{inner}/video_index", prefix / "video_index_map.txt")
    rows.sort(key=lambda row: row["role"])
    compact = [{"role": row["role"], "sha256": row["sha256"]} for row in rows]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "manifest_digest": canonical_digest(compact),
        "outer_test_roles": [],
        "sealed_outer_opened": False,
    }


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    if study_dir.exists():
        if not args.replace:
            raise FileExistsError(study_dir)
        shutil.rmtree(study_dir)
    study_dir.mkdir(parents=True)
    (study_dir / "results").mkdir()
    (study_dir / "logs").mkdir()
    selector = args.selector_study.resolve()
    visual = args.visual_oof_study.resolve()
    manifest = build_manifest(selector, visual)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "V0_candidate_corpus_only",
        "selector_study": str(selector),
        "visual_oof_study": str(visual),
        "selector_budget": 0.05,
        "candidate_sources": {
            "incumbent": 1,
            "visual_head_plain_logistic": 3,
            "diffact_segment_mean_softmax": 5,
            "inpainting_v2_reserved_slots": [],
        },
        "outer_test_open_allowed": False,
        "v1_training_allowed": False,
        "v2_sampling_allowed": False,
        "v3_outer_evaluation_allowed": False,
        "declared_outer_attempt_budget": 1,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "config": config,
        "source_provenance": provenance,
        "input_manifest_digest": manifest["manifest_digest"],
        "review_state": "V0_READY_ONLY",
        "gpu_launched": False,
        "outer_test_opened": False,
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study_dir / "study_config.json", config)
    atomic_write_json(study_dir / "input_manifest.json", manifest)
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable review study: independent-acceptance verifier V0.",
            "OOF candidate-corpus construction only; outer test is forbidden.",
            "V1/V2/V3 remain blocked pending Fable review at each boundary.",
            "Regenerate a new version after any source, input, or protocol change.",
        ],
    )
    runner = study_dir / "run_v0.sh"
    runner.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {Path(__file__).resolve().parent / 'run_v0.py'} "
        f"--study-dir {study_dir} 2>&1 | tee {study_dir / 'logs' / 'v0.log'}\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    print(study_dir)
    print(f"inputs={manifest['file_count']} bytes={manifest['total_bytes']}")
    print(f"spec_sha256={metadata['spec_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
