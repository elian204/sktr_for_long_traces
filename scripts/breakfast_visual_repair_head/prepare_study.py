#!/usr/bin/env python3
"""Prepare, but never launch, an immutable Breakfast visual repair-head study."""

from __future__ import annotations

import argparse
import json
import os
import platform
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import sklearn

from common import (
    CLASSIFIER_MAX_ITER,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_SELECTOR_STUDY,
    DEFAULT_STUDY_DIR,
    FEATURE_DIMENSION,
    OUTER_FOLDS,
    PRIMARY_BUDGET,
    PRIMARY_GATES,
    PRIMARY_THRESHOLD,
    PROTOCOL_VERSION,
    SENSITIVITY_THRESHOLDS,
    TOP_K,
    WORKSPACE_ROOT,
    atomic_write_json,
    canonical_digest,
    feature_orientation,
    file_sha256,
    load_json,
    source_provenance,
    write_lines,
)


REQUIRED_ANALYSIS_ARTIFACTS = (
    "segment_scores.csv",
    "repair_training_corpus_segments.csv",
    "baseline_metrics.csv",
    "selector_budget_metrics.csv",
    "scale_decision.json",
    "run_metadata.json",
    "validation_checks.csv",
)
REQUIRED_SELECTOR_ROOT_ARTIFACTS = (
    "study_metadata.json",
    "selector_config.json",
    "repair_corpus_schema.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--selector-study", type=Path, default=DEFAULT_SELECTOR_STUDY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--hash-workers", type=int, default=4)
    return parser.parse_args()


def _artifact(path: Path, *, role: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _hash_feature(item: tuple[str, Path, int]) -> Dict[str, Any]:
    case_id, path, expected_frames = item
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r")
    orientation, frames = feature_orientation(array.shape)
    if frames != expected_frames:
        raise ValueError(
            f"{case_id}: I3D time length={frames}, expected={expected_frames}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{case_id}: expected floating I3D data, got {array.dtype}")
    return {
        "case_id": case_id,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "orientation": orientation,
        "time_frames": frames,
        "feature_dimension": FEATURE_DIMENSION,
    }


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def _validate_selector_inputs(
    selector_study: Path, analysis_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    decision = load_json(analysis_dir / "scale_decision.json")
    if decision.get("green_light_repair_head") is not True:
        raise RuntimeError("Selector study did not pass its frozen repair-head gate")
    if decision.get("primary_variant") != "base" or decision.get("primary_mode") != "official":
        raise ValueError("Selector decision is not the frozen official/base protocol")
    if abs(float(decision["decision_budget"]) - PRIMARY_BUDGET) > 1e-12:
        raise ValueError("Selector decision budget is not exactly 5%")

    config = load_json(selector_study / "selector_config.json")
    if config.get("primary_variant") != "base":
        raise ValueError("selector_config primary variant drifted")
    base_features = config["variants"]["base"]["feature_columns"]
    if any(
        str(column).startswith(("task__", "camera__")) for column in base_features
    ):
        raise ValueError("Frozen base selector unexpectedly contains metadata")

    scores = pd.read_csv(analysis_dir / "segment_scores.csv")
    corpus = pd.read_csv(analysis_dir / "repair_training_corpus_segments.csv")
    required = {
        "segment_id",
        "outer_fold",
        "scope",
        "mode",
        "case_id",
        "segment_index",
        "start",
        "end",
        "length",
        "predicted_label",
        "correct_label",
        "base_score",
    }
    missing_scores = sorted(required - set(scores.columns))
    missing_corpus = sorted(required - set(corpus.columns))
    if missing_scores or missing_corpus:
        raise ValueError(
            f"Selector CSV contract missing columns: scores={missing_scores}, "
            f"corpus={missing_corpus}"
        )
    candidate_columns = {
        f"candidate_rank_{rank}_class_id" for rank in range(1, TOP_K + 1)
    }
    missing_candidates = sorted(candidate_columns - set(scores.columns))
    if missing_candidates:
        raise ValueError(f"Missing frozen top-five columns: {missing_candidates}")

    corpus_expected = scores[
        (scores["scope"] == "oof_validation") & (scores["mode"] == "official")
    ]
    if len(corpus) != len(corpus_expected):
        raise ValueError(
            f"Repair corpus row count={len(corpus)}, expected={len(corpus_expected)}"
        )
    identity_columns = [
        "segment_id",
        "outer_fold",
        "case_id",
        "segment_index",
        "start",
        "end",
        "length",
        "predicted_label",
        "correct_label",
        "base_score",
    ]
    actual_identity = corpus.sort_values("segment_id")[identity_columns].reset_index(drop=True)
    expected_identity = corpus_expected.sort_values("segment_id")[identity_columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual_identity, expected_identity, check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise ValueError("Repair corpus is not the exact frozen official OOF subset") from error
    for frame, name in ((scores, "scores"), (corpus, "corpus")):
        if set(frame["outer_fold"].astype(int)) != set(OUTER_FOLDS):
            raise ValueError(f"{name}: missing outer folds")
        if frame["base_score"].isna().any():
            raise ValueError(f"{name}: missing base selector scores")
        if not (frame["end"].astype(int) - frame["start"].astype(int)).equals(
            frame["length"].astype(int)
        ):
            raise ValueError(f"{name}: inconsistent segment lengths")
    if set(corpus["scope"]) != {"oof_validation"} or set(corpus["mode"]) != {
        "official"
    }:
        raise ValueError("Repair corpus must contain official OOF segments only")
    return scores, corpus, decision


def _case_frame_contract(
    scores: pd.DataFrame, corpus: pd.DataFrame
) -> Dict[str, int]:
    rows = pd.concat(
        [
            corpus[["outer_fold", "case_id", "start", "end"]],
            scores[
                (scores["scope"] == "outer_test") & (scores["mode"] == "official")
            ][["outer_fold", "case_id", "start", "end"]],
        ],
        ignore_index=True,
    )
    frame_counts: Dict[str, int] = {}
    for outer_fold in OUTER_FOLDS:
        train_cases = set(
            corpus[corpus["outer_fold"].astype(int) == outer_fold]["case_id"].astype(str)
        )
        test_cases = set(
            scores[
                (scores["outer_fold"].astype(int) == outer_fold)
                & (scores["scope"] == "outer_test")
                & (scores["mode"] == "official")
            ]["case_id"].astype(str)
        )
        if train_cases & test_cases:
            raise ValueError(f"OOF repair training overlaps outer test in fold {outer_fold}")
        if len(train_cases | test_cases) != 1712:
            raise ValueError(f"Fold {outer_fold} does not partition all 1,712 cases")
    for (outer_fold, case_id), group in rows.groupby(["outer_fold", "case_id"]):
        ordered = group.sort_values(["start", "end"], kind="mergesort")
        starts = ordered["start"].to_numpy(dtype=int)
        ends = ordered["end"].to_numpy(dtype=int)
        if starts[0] != 0 or np.any(starts[1:] != ends[:-1]):
            raise ValueError(
                f"Non-contiguous segment partition: outer={outer_fold}, case={case_id}"
            )
        frames = int(ends[-1])
        previous = frame_counts.setdefault(str(case_id), frames)
        if previous != frames:
            raise ValueError(f"Cross-fold frame-count mismatch for {case_id}")
    if len(frame_counts) != 1712:
        raise ValueError(f"Expected all 1,712 Breakfast videos, got {len(frame_counts)}")
    return frame_counts


def prepare(args: argparse.Namespace) -> Path:
    study_dir = args.study_dir.resolve()
    if study_dir.exists() and any(study_dir.iterdir()):
        raise FileExistsError(f"Study directory is not empty: {study_dir}")
    study_dir.mkdir(parents=True, exist_ok=True)

    selector_study = args.selector_study.resolve()
    data_root = args.data_root.resolve()
    feature_root = args.feature_root.resolve()
    analysis_dir = selector_study / "analysis" / "video_selector_allfolds_v2"
    for name in REQUIRED_ANALYSIS_ARTIFACTS:
        if not (analysis_dir / name).is_file():
            raise FileNotFoundError(analysis_dir / name)
    for name in REQUIRED_SELECTOR_ROOT_ARTIFACTS:
        if not (selector_study / name).is_file():
            raise FileNotFoundError(selector_study / name)

    scores, corpus, decision = _validate_selector_inputs(selector_study, analysis_dir)
    frame_counts = _case_frame_contract(scores, corpus)

    mapping_path = data_root / "mapping.txt"
    mapping = _artifact(mapping_path, role="label_mapping")
    if len(mapping_path.read_text().splitlines()) != 48:
        raise ValueError("Breakfast mapping must contain 48 labels")

    fixed_artifacts = []
    for name in REQUIRED_ANALYSIS_ARTIFACTS:
        fixed_artifacts.append(_artifact(analysis_dir / name, role=f"selector_analysis/{name}"))
    for name in REQUIRED_SELECTOR_ROOT_ARTIFACTS:
        fixed_artifacts.append(_artifact(selector_study / name, role=f"selector_root/{name}"))
    fixed_artifacts.append(mapping)

    ground_truth = []
    for case_id, expected_frames in sorted(frame_counts.items()):
        path = data_root / "groundTruth" / f"{case_id}.txt"
        artifact = _artifact(path, role="ground_truth")
        lines = path.read_text().splitlines()
        if len(lines) != expected_frames:
            raise ValueError(
                f"{case_id}: GT frames={len(lines)}, expected={expected_frames}"
            )
        artifact.update({"case_id": case_id, "time_frames": expected_frames})
        ground_truth.append(artifact)

    feature_items = [
        (case_id, feature_root / f"{case_id}.npy", expected_frames)
        for case_id, expected_frames in sorted(frame_counts.items())
    ]
    workers = max(1, int(args.hash_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        features = list(executor.map(_hash_feature, feature_items))
    features.sort(key=lambda row: row["case_id"])

    manifest_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fixed_artifacts": fixed_artifacts,
        "ground_truth": ground_truth,
        "features": features,
        "counts": {
            "fixed_artifacts": len(fixed_artifacts),
            "ground_truth_files": len(ground_truth),
            "feature_files": len(features),
            "feature_bytes": sum(int(row["bytes"]) for row in features),
        },
    }
    manifest_payload["manifest_digest"] = canonical_digest(manifest_payload)
    atomic_write_json(study_dir / "input_manifest.json", manifest_payload)

    config = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "breakfast",
        "outer_folds": list(OUTER_FOLDS),
        "primary": {
            "selector_variant": "base",
            "selector_mode": "official",
            "frame_budget": PRIMARY_BUDGET,
            "budget_scope": "independently_per_outer_fold",
            "partial_segment_rule": "one_deterministic_centered_cutoff_at_most",
            "pooling": "per_segment_i3d_channel_mean_plus_population_std",
            "feature_dimension": FEATURE_DIMENSION,
            "scaler": {
                "class": "sklearn.preprocessing.StandardScaler",
                "parameters": "sklearn_defaults",
                "sample_weighted": False,
            },
            "classifier": {
                "class": "sklearn.linear_model.LogisticRegression",
                "parameters": {
                    "C": 1.0,
                    "max_iter": CLASSIFIER_MAX_ITER,
                    "all_others": "sklearn_defaults",
                },
                "loss_weight": "segment_length",
                "target": "ground_truth_majority_label",
                "convergence_warning_policy": "abort_study_hard_failure",
            },
            "proposal": "unrestricted_classifier_top1",
            "threshold": PRIMARY_THRESHOLD,
            "threshold_probability": "unmodified_multinomial_probability",
            "abstain_if_same_as_current_label": True,
            "relabel_scope": "selected_frames_only",
            "uses_metadata": False,
            "uses_process_features": False,
            "uses_shape_features": False,
        },
        "gates": PRIMARY_GATES,
        "analysis_only": {
            "thresholds": list(SENSITIVITY_THRESHOLDS),
            "proposal_variants": ["free_choice", "top5_restricted"],
            "top5_probability_renormalized": False,
        },
        "extensions": {
            "launch_only_if_primary_passes": True,
            "gpu_studies": [
                "diffact_encoder_feature_pooling",
                "multi_sample_diffusion_proposals",
            ],
        },
    }
    config["config_digest"] = canonical_digest(config)
    atomic_write_json(study_dir / "study_config.json", config)

    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "selector_study": str(selector_study),
        "selector_analysis_dir": str(analysis_dir),
        "data_root": str(data_root),
        "feature_root": str(feature_root),
        "input_manifest": str(study_dir / "input_manifest.json"),
        "input_manifest_digest": manifest_payload["manifest_digest"],
        "config_digest": config["config_digest"],
        "selector_gate_passed": True,
        "selector_gate_snapshot": decision,
        "source_provenance": provenance,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "cpu_only": True,
        },
        "execution_state": "staged_not_launched",
    }
    atomic_write_json(study_dir / "study_metadata.json", metadata)

    runner = WORKSPACE_ROOT / "scripts" / "breakfast_visual_repair_head" / "run_repair_head.py"
    python = Path(sys.executable).resolve()
    _write_executable(
        study_dir / "run.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"STUDY_DIR={json.dumps(str(study_dir))}\n"
        "mkdir -p \"$STUDY_DIR/logs\"\n"
        "export CUDA_VISIBLE_DEVICES=\"\"\n"
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}\n"
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}\n"
        f"{json.dumps(str(python))} {json.dumps(str(runner))} "
        "--study-dir \"$STUDY_DIR\" 2>&1 | tee \"$STUDY_DIR/logs/run.log\"\n",
    )
    _write_executable(
        study_dir / "status.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"STUDY_DIR={json.dumps(str(study_dir))}\n"
        "if [[ -f \"$STUDY_DIR/results/run_complete.json\" ]]; then\n"
        "  echo complete\n"
        "elif [[ -f \"$STUDY_DIR/results/run_failed.json\" ]]; then\n"
        "  echo failed\n"
        "elif [[ -f \"$STUDY_DIR/results/run_status.json\" ]]; then\n"
        "  cat \"$STUDY_DIR/results/run_status.json\"\n"
        "else\n"
        "  echo staged_not_launched\n"
        "fi\n",
    )
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable pre-registered study directory.",
            "Do not edit protocol, metadata, configuration, or input manifests in place.",
            "Only run.sh may add declared logs, models, predictions, and results.",
        ],
    )
    print(f"Prepared review study: {study_dir}")
    print(f"Input manifest digest: {manifest_payload['manifest_digest']}")
    print(f"Feature files hash-locked: {len(features)} ({manifest_payload['counts']['feature_bytes']} bytes)")
    print("No model was fit and no study was launched.")
    return study_dir


def main() -> int:
    prepare(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
