#!/usr/bin/env python3
"""Prepare the immutable OOF-only screening phase for repair-head v2."""

from __future__ import annotations

import argparse
import json
import platform
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import sklearn

from common import (
    BASELINE_FORENSICS_THRESHOLD,
    BUDGET_CANDIDATES,
    CAP_PERCENT_CANDIDATES,
    DEFAULT_OOF_STUDY,
    DEFAULT_V1_REVIEW_STUDY,
    LARGE_SPAN_THRESHOLD_INCREMENT,
    MARGIN_CANDIDATES,
    MAXIMUM_HIGHER_CONFIDENCE,
    MODEL_PROMOTION_GAIN_PP,
    OOF_HARM_FLOOR_PP,
    OUTER_FOLDS,
    PRIMARY_BUDGET,
    PROTOCOL_VERSION,
    SECONDARY_BUDGET_GAIN_PP,
    SENSITIVITY_THRESHOLDS,
    SPAN_SIZE_PERCENT_CANDIDATES,
    TAU_CANDIDATES,
    WORKSPACE_ROOT,
    atomic_write_json,
    canonical_digest,
    load_json,
    source_provenance,
    verify_self_digest,
    write_lines,
)
from core import verify_manifest_entries


OOF_ROLES = (
    "selector_analysis/repair_training_corpus_segments.csv",
    "selector_root/selector_config.json",
    "selector_root/repair_corpus_schema.json",
    "label_mapping",
)
SEALED_OUTER_ROLES = (
    "selector_analysis/segment_scores.csv",
    "selector_analysis/baseline_metrics.csv",
    "selector_analysis/selector_budget_metrics.csv",
    "selector_analysis/repair_training_corpus_segments.csv",
    "selector_root/selector_config.json",
    "selector_root/repair_corpus_schema.json",
    "label_mapping",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_OOF_STUDY)
    parser.add_argument("--v1-review-study", type=Path, default=DEFAULT_V1_REVIEW_STUDY)
    parser.add_argument("--verify-workers", type=int, default=4)
    return parser.parse_args()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def _derived_manifest(
    upstream: Mapping[str, Any], roles: tuple[str, ...], *, phase: str
) -> Dict[str, Any]:
    role_map = {str(row["role"]): row for row in upstream["fixed_artifacts"]}
    missing = sorted(set(roles) - set(role_map))
    if missing:
        raise ValueError(f"Upstream input manifest misses roles: {missing}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "upstream_v1_input_manifest_digest": upstream["manifest_digest"],
        "fixed_artifacts": [role_map[role] for role in roles],
        "ground_truth": list(upstream["ground_truth"]),
        "features": list(upstream["features"]),
    }
    payload["manifest_digest"] = canonical_digest(payload)
    return payload


def prepare(args: argparse.Namespace) -> Path:
    study_dir = args.study_dir.resolve()
    if study_dir.exists() and any(study_dir.iterdir()):
        raise FileExistsError(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    upstream_dir = args.v1_review_study.resolve()
    upstream_manifest = load_json(upstream_dir / "input_manifest.json")
    upstream_metadata = load_json(upstream_dir / "study_metadata.json")
    verify_self_digest(upstream_manifest, "manifest_digest")
    if upstream_metadata["input_manifest_digest"] != upstream_manifest["manifest_digest"]:
        raise RuntimeError("v1 metadata/input digest disagreement")
    upstream_verification = verify_manifest_entries(
        upstream_manifest, workers=args.verify_workers
    )

    oof_manifest = _derived_manifest(upstream_manifest, OOF_ROLES, phase="oof_screening")
    sealed_manifest = _derived_manifest(
        upstream_manifest, SEALED_OUTER_ROLES, phase="sealed_outer_evaluation"
    )
    atomic_write_json(study_dir / "oof_input_manifest.json", oof_manifest)
    atomic_write_json(study_dir / "sealed_outer_input_manifest.json", sealed_manifest)

    config = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_scope": "independently_within_each_outer_folds_oof_corpus",
        "contamination_contract": {
            "oof_runner_allowed_manifest": "oof_input_manifest.json",
            "oof_runner_forbidden_manifest": "sealed_outer_input_manifest.json",
            "v1_outer_results_forbidden": True,
            "raw_case_ids_forbidden_in_oof_outputs": True,
            "outer_evaluation_requires_separate_review_and_exact_digest_approval": True,
        },
        "forensics": {
            "model": "plain_logistic",
            "budget": PRIMARY_BUDGET,
            "threshold": BASELINE_FORENSICS_THRESHOLD,
            "rule": "none",
        },
        "harm_rule_screen": {
            "fixed_model": "plain_logistic",
            "fixed_budget": PRIMARY_BUDGET,
            "fixed_threshold": BASELINE_FORENSICS_THRESHOLD,
            "video_cap_percent": list(CAP_PERCENT_CANDIDATES),
            "incumbent_margin": list(MARGIN_CANDIDATES),
            "large_span_percent": list(SPAN_SIZE_PERCENT_CANDIDATES),
            "large_span_threshold_increment": LARGE_SPAN_THRESHOLD_INCREMENT,
            "higher_threshold_cap": MAXIMUM_HIGHER_CONFIDENCE,
            "harm_floor_pp": OOF_HARM_FLOOR_PP,
            "selection_metrics": [
                "delta_acc",
                "delta_f1@25",
                "delta_edit",
                "worst_video_delta_acc",
                "rule_order",
                "lower_numeric_parameter",
            ],
        },
        "tau_selection": {
            "candidates": list(TAU_CANDIDATES),
            "budget": PRIMARY_BUDGET,
            "requires_harm_constraint": True,
        },
        "model_screen": {
            "candidates": [
                "plain_logistic",
                "isotonic_logistic",
                "logistic_mlp_average",
            ],
            "promotion_gain_pp": MODEL_PROMOTION_GAIN_PP,
            "requires_harm_constraint": True,
            "mlp": {
                "hidden_layer_sizes": [128],
                "solver": "adam",
                "max_iter": 300,
                "early_stopping": True,
                "n_iter_no_change": 20,
                "random_state": 0,
                "sample_weighted": False,
            },
        },
        "budget_envelope": {
            "primary": PRIMARY_BUDGET,
            "candidates": list(BUDGET_CANDIDATES),
            "secondary_minimum_gain_over_primary_pp": SECONDARY_BUDGET_GAIN_PP,
            "requires_harm_constraint": True,
        },
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
    }
    config["config_digest"] = canonical_digest(config)
    atomic_write_json(study_dir / "screening_config.json", config)

    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "upstream_v1_review_study": str(upstream_dir),
        "upstream_v1_input_manifest_digest": upstream_manifest["manifest_digest"],
        "upstream_v1_runtime_input_verification": upstream_verification,
        "oof_input_manifest_digest": oof_manifest["manifest_digest"],
        "sealed_outer_input_manifest_digest": sealed_manifest["manifest_digest"],
        "screening_config_digest": config["config_digest"],
        "source_provenance": provenance,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "cpu_only": True,
        },
        "state": "oof_staged_outer_sealed",
    }
    atomic_write_json(study_dir / "study_metadata.json", metadata)

    runner = WORKSPACE_ROOT / "scripts" / "breakfast_visual_repair_head_v2" / "run_oof_screening.py"
    python = Path(sys.executable).resolve()
    _write_executable(
        study_dir / "run_oof.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"STUDY_DIR={json.dumps(str(study_dir))}\n"
        "mkdir -p \"$STUDY_DIR/logs\"\n"
        "export CUDA_VISIBLE_DEVICES=\"\"\n"
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}\n"
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}\n"
        f"{json.dumps(str(python))} {json.dumps(str(runner))} --study-dir \"$STUDY_DIR\" "
        "2>&1 | tee \"$STUDY_DIR/logs/oof_screening.log\"\n",
    )
    _write_executable(
        study_dir / "status.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"STUDY_DIR={json.dumps(str(study_dir))}\n"
        "if [[ -f \"$STUDY_DIR/oof_results/oof_screen_complete.json\" ]]; then echo oof_complete_outer_unopened; "
        "elif [[ -f \"$STUDY_DIR/oof_results/run_failed.json\" ]]; then echo oof_failed; "
        "elif [[ -f \"$STUDY_DIR/oof_results/run_status.json\" ]]; then cat \"$STUDY_DIR/oof_results/run_status.json\"; "
        "else echo oof_staged_outer_sealed; fi\n",
    )
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable OOF-screening study; outer inputs are sealed.",
            "The OOF runner must never open sealed_outer_input_manifest.json.",
            "Only declared OOF outputs may be added by run_oof.sh.",
        ],
    )
    print(f"Prepared OOF screening study: {study_dir}")
    print(f"Upstream unchanged input digest: {upstream_manifest['manifest_digest']}")
    print(f"OOF manifest digest: {oof_manifest['manifest_digest']}")
    print(f"Sealed outer manifest digest: {sealed_manifest['manifest_digest']}")
    print("Outer-test content was not opened; nothing was launched.")
    return study_dir


def main() -> int:
    prepare(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
