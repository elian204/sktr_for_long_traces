#!/usr/bin/env python3
"""Prepare a review-only V2 corrected-sampler B0 validity study."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2_common import (
    B1_KILL_BARS,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_SELECTOR_DIR,
    DEFAULT_STUDY_DIR,
    DEFAULT_V0_DIR,
    EARLY_STOP_MIN_SAMPLES,
    EARLY_STOP_PATIENCE,
    HALO_WIDTHS,
    INNER_FOLDS,
    MAX_SAMPLES,
    OUTER_FOLDS,
    PROTOCOL_VERSION,
    RESTART_TIMES,
    atomic_write_json,
    canonical_digest,
    checkpoint_and_config,
    file_sha256,
    load_json,
    source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--v0-dir", type=Path, default=DEFAULT_V0_DIR)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--authorize-b0", action="store_true")
    parser.add_argument("--fable-approval-digest")
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V2 B0 input {role}: {path}")
    rows.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    )


def main() -> int:
    args = parse_args()
    if args.authorize_b0 and not args.fable_approval_digest:
        raise ValueError("Authorized B0 generation requires --fable-approval-digest")
    if not args.authorize_b0 and args.fable_approval_digest:
        raise ValueError("Review-only B0 must not claim an approval digest")
    study_dir = args.study_dir.resolve()
    if study_dir.exists():
        if not args.replace:
            raise FileExistsError(study_dir)
        shutil.rmtree(study_dir)
    for name in ("logs", "results", "status"):
        (study_dir / name).mkdir(parents=True, exist_ok=True)

    v0 = args.v0_dir.resolve()
    selector = args.selector_dir.resolve()
    diffact = args.diffact_root.resolve()
    v0_input = load_json(v0 / "input_manifest.json")
    v0_paths = {str(row["role"]): Path(row["path"]) for row in v0_input["files"]}
    rows: list[dict[str, Any]] = []
    for name in ("candidate_corpus.csv", "flagged_oof_spans.csv", "v0_complete.json"):
        add_input(rows, f"v0/results/{name}", v0 / "results" / name)
    add_input(rows, "v0/oof_segment_corpus", v0_paths["selector/oof_segment_corpus"])
    add_input(rows, "features/nested_manifest", args.feature_manifest.resolve())
    add_input(rows, "breakfast/mapping", Path("/home/dsi/eli-bogdanov/data/data/breakfast/mapping.txt"))
    for name in ("model.py", "main.py", "dataset.py", "utils.py"):
        add_input(rows, f"diffact/source/{name}", diffact / name)
    task_rows: list[dict[str, Any]] = []
    for outer in OUTER_FOLDS:
        for inner in INNER_FOLDS:
            checkpoint, config_path = checkpoint_and_config(selector, outer, inner)
            add_input(rows, f"task/outer{outer}/inner{inner}/checkpoint", checkpoint)
            add_input(rows, f"task/outer{outer}/inner{inner}/config", config_path)
            task_rows.append(
                {
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "config": str(config_path.resolve()),
                    "config_sha256": file_sha256(config_path),
                }
            )
    rows.sort(key=lambda row: row["role"])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_digest": canonical_digest(
            [{"role": row["role"], "sha256": row["sha256"]} for row in rows]
        ),
    }
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "OOF_only_B0_validity",
        "diffact_root": str(diffact),
        "tasks": task_rows,
        "sampler": {
            "three_regions": ["free_core", "halo_buffer", "clamped_exterior"],
            "halo_widths": list(HALO_WIDTHS),
            "restart_times": list(RESTART_TIMES),
            "pure_noise_start": 999,
            "sampling_timesteps": 25,
            "context_noise": "one_fixed_tensor_shared_across_reverse_steps",
            "postprocess": {"type": "median", "value": 15},
            "non_core_restore_after_postprocess": True,
            "max_samples_per_setting": MAX_SAMPLES,
            "early_stop_min_samples": EARLY_STOP_MIN_SAMPLES,
            "early_stop_patience_no_new_collapsed_trace": EARLY_STOP_PATIENCE,
            "candidate": "cluster_medoid_normalized_segmental_edit_threshold_0.25",
            "per_frame_voting_allowed": False,
        },
        "b0_checks": ["postprocess_exterior_invariance", "empty_mask_identity", "seeded_replay"],
        "b1_kill_bars": B1_KILL_BARS,
        "b0_sampling_allowed": bool(args.authorize_b0),
        "b1_oracle_allowed": False,
        "v1_candidate_join_allowed": False,
        "outer_test_open_allowed": False,
        "v3_outer_evaluation_allowed": False,
        "fable_approval_digest": args.fable_approval_digest,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "review_state": "V2_B0_REVIEW_ONLY" if not args.authorize_b0 else "V2_B0_APPROVED_READY",
        "input_manifest_digest": manifest["manifest_digest"],
        "source_provenance": provenance,
        "outer_test_opened": False,
        "gpu_launched": False,
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study_dir / "study_config.json", config)
    atomic_write_json(study_dir / "input_manifest.json", manifest)
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    (study_dir / "DO_NOT_EDIT.txt").write_text(
        "Immutable V2 B0 corrected-sampler validity study.\n"
        "Review-only unless a Fable approval digest is embedded.\n"
        "B1, candidate joining, and all outer-test access are disabled.\n"
    )
    for outer in OUTER_FOLDS:
        for inner in INNER_FOLDS:
            device = (outer + inner - 2) % 4
            script = study_dir / f"run_outer{outer}_inner{inner}.sh"
            script.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f"/usr/bin/python {Path(__file__).resolve().parent / 'run_v2_b0_validity.py'} "
                f"--study-dir {study_dir} --outer-fold {outer} --inner-fold {inner} "
                f"--device {device} 2>&1 | tee {study_dir / 'logs' / f'outer{outer}_inner{inner}.log'}\n"
            )
            script.chmod(0o755)
    print(study_dir)
    print(f"spec_sha256={metadata['spec_sha256']}")
    print(f"source_digest={provenance['source_digest']}")
    print(f"input_manifest_digest={manifest['manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
