#!/usr/bin/env python3
"""Prepare a review-only nested-OOF V1 verifier study."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v1_common import (
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_STUDY_DIR,
    DEFAULT_V0_DIR,
    INNER_FOLDS,
    MODEL_CONFIG,
    OUTER_FOLDS,
    PROTOCOL_VERSION,
    THRESHOLDS,
    TRAIN_CONFIG,
    V1_GATES,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--v0-dir", type=Path, default=DEFAULT_V0_DIR)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--authorize-run", action="store_true")
    parser.add_argument("--fable-approval-digest")
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V1 input {role}: {path}")
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
    if args.authorize_run and not args.fable_approval_digest:
        raise ValueError("Authorized V1 generation requires --fable-approval-digest")
    if not args.authorize_run and args.fable_approval_digest:
        raise ValueError("Review-only V1 must not claim an approval digest")
    study_dir = args.study_dir.resolve()
    if study_dir.exists():
        if not args.replace:
            raise FileExistsError(study_dir)
        shutil.rmtree(study_dir)
    for name in ("cache", "logs", "results", "rotations", "status"):
        (study_dir / name).mkdir(parents=True, exist_ok=True)

    v0 = args.v0_dir.resolve()
    rows: list[dict[str, Any]] = []
    for name in (
        "candidate_corpus.csv",
        "flagged_oof_spans.csv",
        "candidate_corpus_schema.json",
        "v0_complete.json",
    ):
        add_input(rows, f"v0/results/{name}", v0 / "results" / name)
    add_input(rows, "v0/input_manifest", v0 / "input_manifest.json")
    add_input(rows, "v0/study_metadata", v0 / "study_metadata.json")
    add_input(rows, "features/nested_manifest", args.feature_manifest.resolve())
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
        "scope": "OOF_only_no_outer_test",
        "v0_dir": str(v0),
        "model": MODEL_CONFIG,
        "training": TRAIN_CONFIG,
        "nested_design": {
            "outer_folds": list(OUTER_FOLDS),
            "inner_folds": list(INNER_FOLDS),
            "rotations": 12,
            "per_rotation": "train final model on two inner folds; evaluate untouched third",
            "threshold_selection": "two-way crossfit inside the two tuning folds",
            "same_case_across_outer_folds_never_crosses_a_model_boundary": True,
            "gate_aggregation": "combine four outer-fold evaluations within each held-inner index",
        },
        "thresholds": list(THRESHOLDS),
        "gates": V1_GATES,
        "feature_build_allowed": bool(args.authorize_run),
        "gpu_training_allowed": bool(args.authorize_run),
        "outer_test_open_allowed": False,
        "v2_sampling_allowed": False,
        "v3_outer_evaluation_allowed": False,
        "fable_approval_digest": args.fable_approval_digest,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "review_state": "V1_REVIEW_ONLY" if not args.authorize_run else "V1_APPROVED_READY",
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
        "Immutable independent-verifier V1 study.\n"
        "This instance is review-only unless study_config explicitly carries Fable approval.\n"
        "Outer-test and V3 access are forbidden.\n"
    )
    build_script = study_dir / "build_features.sh"
    build_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {Path(__file__).resolve().parent / 'build_v1_features.py'} "
        f"--study-dir {study_dir} 2>&1 | tee {study_dir / 'logs' / 'build_features.log'}\n"
    )
    build_script.chmod(0o755)
    for outer in OUTER_FOLDS:
        for held in INNER_FOLDS:
            script = study_dir / f"run_outer{outer}_held{held}.sh"
            device = (outer + held - 2) % 4
            script.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f"/usr/bin/python {Path(__file__).resolve().parent / 'train_v1_rotation.py'} "
                f"--study-dir {study_dir} --outer-fold {outer} --held-inner {held} "
                f"--device {device} 2>&1 | tee {study_dir / 'logs' / f'outer{outer}_held{held}.log'}\n"
            )
            script.chmod(0o755)
    aggregate = study_dir / "aggregate.sh"
    aggregate.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {Path(__file__).resolve().parent / 'aggregate_v1.py'} "
        f"--study-dir {study_dir} 2>&1 | tee {study_dir / 'logs' / 'aggregate.log'}\n"
    )
    aggregate.chmod(0o755)
    queue = study_dir / "run_all_rotations_on_device.sh"
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "if [[ $# -ne 1 ]]; then echo 'usage: run_all_rotations_on_device.sh DEVICE'; exit 2; fi",
        "device=$1",
        "mkdir -p logs status",
        "printf '%s\\n' \"$device\" > status/chosen_gpu.txt",
    ]
    for outer in OUTER_FOLDS:
        for held in INNER_FOLDS:
            commands.append(
                f"/usr/bin/python {Path(__file__).resolve().parent / 'train_v1_rotation.py'} "
                f"--study-dir {study_dir} --outer-fold {outer} --held-inner {held} "
                f"--device \"$device\" 2>&1 | tee -a {study_dir / 'logs' / 'v1_queue.log'}"
            )
    commands.append(f"{study_dir / 'aggregate.sh'}")
    queue.write_text("\n".join(commands) + "\n")
    queue.chmod(0o755)
    waiter = study_dir / "wait_first_gpu.sh"
    waiter.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs status\n"
        "if [[ ! -f cache/feature_cache_complete.json ]]; then ./build_features.sh; fi\n"
        "while true; do\n"
        "  for device in 0 1 2 3; do\n"
        "    first=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
        "    if [[ \"$first\" -eq 0 ]]; then\n"
        "      sleep 30\n"
        "      second=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
        "      if [[ \"$second\" -eq 0 ]]; then exec ./run_all_rotations_on_device.sh \"$device\"; fi\n"
        "    fi\n"
        "  done\n"
        "  sleep 30\n"
        "done\n"
    )
    waiter.chmod(0o755)
    launcher = study_dir / "launch_tmux.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs status\n"
        "if tmux has-session -t iav_v1_waiter 2>/dev/null; then echo 'iav_v1_waiter already exists'; exit 1; fi\n"
        f"tmux new-session -d -s iav_v1_waiter 'cd {study_dir} && ./wait_first_gpu.sh'\n"
    )
    launcher.chmod(0o755)
    print(study_dir)
    print(f"spec_sha256={metadata['spec_sha256']}")
    print(f"source_digest={provenance['source_digest']}")
    print(f"input_manifest_digest={manifest['manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
