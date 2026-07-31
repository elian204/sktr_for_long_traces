#!/usr/bin/env python3
"""Prepare the separately reviewed, single-shot outer evaluation for v2."""

from __future__ import annotations

import argparse
import json
import platform
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from common import (
    DEFAULT_OOF_STUDY,
    DEFAULT_OUTER_REVIEW_STUDY,
    PROTOCOL_VERSION,
    WORKSPACE_ROOT,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    source_provenance,
    verify_self_digest,
    write_lines,
)
from core import verify_manifest_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-study", type=Path, default=DEFAULT_OOF_STUDY)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_OUTER_REVIEW_STUDY)
    parser.add_argument("--verify-workers", type=int, default=4)
    return parser.parse_args()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def prepare(args: argparse.Namespace) -> Path:
    oof_study = args.oof_study.resolve()
    study_dir = args.study_dir.resolve()
    if study_dir.exists() and any(study_dir.iterdir()):
        raise FileExistsError(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    complete = load_json(oof_study / "oof_results" / "oof_screen_complete.json")
    frozen = load_json(oof_study / "oof_results" / "frozen_outer_config.json")
    if complete["outer_inputs_opened"] is not False:
        raise RuntimeError("OOF phase did not preserve the outer seal")
    if complete["outer_evaluation_blocked"] or frozen["outer_evaluation_blocked"]:
        raise RuntimeError("OOF screening blocked outer evaluation")
    verify_self_digest(frozen, "frozen_config_digest")
    if complete["frozen_config_digest"] != frozen["frozen_config_digest"]:
        raise RuntimeError("Frozen-config digest disagreement")
    frozen_path = oof_study / "oof_results" / "frozen_outer_config.json"
    if complete["output_sha256"].get("frozen_outer_config.json") != file_sha256(frozen_path):
        raise RuntimeError("Frozen configuration is not the completed OOF artifact")

    sealed = load_json(oof_study / "sealed_outer_input_manifest.json")
    verify_self_digest(sealed, "manifest_digest")
    input_verification = verify_manifest_entries(sealed, workers=args.verify_workers)
    source = source_provenance()

    atomic_write_json(study_dir / "frozen_outer_config.json", frozen)
    atomic_write_json(study_dir / "outer_input_manifest.json", sealed)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "oof_study": str(oof_study),
        "oof_complete_sha256": file_sha256(
            oof_study / "oof_results" / "oof_screen_complete.json"
        ),
        "oof_findings_sha256": file_sha256(oof_study / "oof_results" / "findings.md"),
        "frozen_config_digest": frozen["frozen_config_digest"],
        "outer_input_manifest_digest": sealed["manifest_digest"],
        "outer_input_verification_at_staging": input_verification,
        "source_provenance": source,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "cpu_only": True,
        },
        "state": "awaiting_fable_exact_digest_approval",
        "outer_test_opened": False,
    }
    atomic_write_json(study_dir / "study_metadata.json", metadata)

    python = Path(sys.executable).resolve()
    runner = WORKSPACE_ROOT / "scripts" / "breakfast_visual_repair_head_v2" / "run_outer_evaluation.py"
    _write_executable(
        study_dir / "run_outer.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"STUDY_DIR={json.dumps(str(study_dir))}\n"
        "if [[ ! -f \"$STUDY_DIR/outer_review_approval.json\" ]]; then "
        "echo 'Outer review approval is absent; refusing.' >&2; exit 1; fi\n"
        "mkdir -p \"$STUDY_DIR/logs\"\n"
        "export CUDA_VISIBLE_DEVICES=\"\"\n"
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}\n"
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}\n"
        f"{json.dumps(str(python))} {json.dumps(str(runner))} --study-dir \"$STUDY_DIR\" "
        "2>&1 | tee \"$STUDY_DIR/logs/outer_evaluation.log\"\n",
    )
    _write_executable(
        study_dir / "status.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"STUDY_DIR={json.dumps(str(study_dir))}\n"
        "if [[ -f \"$STUDY_DIR/outer_results/run_complete.json\" ]]; then echo outer_complete; "
        "elif [[ -f \"$STUDY_DIR/outer_results/run_failed.json\" ]]; then echo outer_failed; "
        "elif [[ -f \"$STUDY_DIR/outer_results/run_status.json\" ]]; then cat \"$STUDY_DIR/outer_results/run_status.json\"; "
        "elif [[ -f \"$STUDY_DIR/outer_review_approval.json\" ]]; then echo approved_not_launched; "
        "else echo awaiting_fable_exact_digest_approval; fi\n",
    )
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Single-shot outer evaluation. Do not inspect outer inputs before approval.",
            "Only record_outer_approval.py may add outer_review_approval.json.",
            "Only run_outer.sh may add declared outer outputs after approval.",
        ],
    )
    print(f"Prepared outer review study: {study_dir}")
    print(f"Frozen config digest: {frozen['frozen_config_digest']}")
    print(f"Outer input digest: {sealed['manifest_digest']}")
    print("Outer content was hash-verified but not parsed; approval is absent.")
    return study_dir


def main() -> int:
    prepare(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
