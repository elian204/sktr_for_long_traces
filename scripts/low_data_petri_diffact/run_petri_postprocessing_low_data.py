#!/usr/bin/env python3
"""Run factorized low-data Petri postprocessing on exported DiffAct streams."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from low_data_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPERIMENT_DIR,
    FRACTIONS,
    SEEDS,
    WORKSPACE_ROOT,
    collapse_runs,
    load_ground_truth_indices,
    load_mapping,
    one_hot_softmax,
    read_case_manifest,
)

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.calibration import calibrate_probabilities, softmax_numpy  # noqa: E402
from src.evaluation import compute_tas_metrics_asformer  # noqa: E402
from src.incremental_softmax_recovery import incremental_softmax_recovery, setup_logging  # noqa: E402
from src.utils import inverse_softmax, linear_prob_combiner  # noqa: E402


HP_STRATEGIES = {
    "trigram_heavy": [0.1, 0.15, 0.75],
    "unigram_super_heavy": [0.75, 0.15, 0.1],
}

DATASET_HP_DEFAULTS = {
    ("50salads", "diffact"): {"alpha": 0.3, "strategy": "unigram_super_heavy"},
    ("gtea", "diffact"): {"alpha": 0.95, "strategy": "trigram_heavy"},
    ("breakfast", "diffact"): {"alpha": 0.7, "strategy": "trigram_heavy"},
}

PAPER_DIFFACT_SKTR_METHOD = "petri_conformance"
PAPER_DIFFACT_SKTR_CHUNK_SIZE = 11
PAPER_DIFFACT_SKTR_WORKERS = 7
PAPER_DIFFACT_SKTR_PROGRESS_CHUNKS = 20
PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION = "frame_events"
COUPLED_CONDITION = "honest_low_data_petri"
DIFFACT_LOW_PETRI_FULL_CONDITION = "structural_prior_full_train_petri"
DIFFACT_FULL_PETRI_LOW_CONDITION = "full_diffact_low_data_petri"
ORACLE_TEST_FOLD_CONDITION = "oracle_test_fold_petri"
CONDITIONS = (
    COUPLED_CONDITION,
    DIFFACT_LOW_PETRI_FULL_CONDITION,
    DIFFACT_FULL_PETRI_LOW_CONDITION,
    ORACLE_TEST_FOLD_CONDITION,
)
PETRI_DISCOVERY_NESTED = "nested_train_fraction"
PETRI_DISCOVERY_FULL_TRAIN = "full_train"
PETRI_DISCOVERY_OFFICIAL_TEST = "official_test_fold"
PETRI_DISCOVERY_SOURCES = (
    PETRI_DISCOVERY_NESTED,
    PETRI_DISCOVERY_FULL_TRAIN,
    PETRI_DISCOVERY_OFFICIAL_TEST,
)
PRIMARY_PROTOCOL = "primary_no_validation"
LEGACY_PROTOCOL = "legacy_pilot"
PROTOCOL_CHOICES = ("auto", PRIMARY_PROTOCOL, LEGACY_PROTOCOL)


def resolve_protocol(metadata: Mapping[str, Any], requested: str = "auto") -> str:
    """Resolve new no-validation studies while treating unversioned metadata as legacy."""
    if requested != "auto":
        return requested
    if metadata.get("primary_no_validation") is True or metadata.get("validation_enabled") is False:
        return PRIMARY_PROTOCOL
    raw_version = metadata.get("protocol_version", metadata.get("protocol", ""))
    normalized = str(raw_version).strip().lower().replace("-", "_")
    try:
        if float(normalized.lstrip("v")) >= 2:
            return PRIMARY_PROTOCOL
    except ValueError:
        pass
    if (
        normalized.endswith("_v2")
        or normalized.startswith("v2_")
        or any(
            token in normalized
            for token in ("no_validation", "approved", "primary_curve")
        )
    ):
        return PRIMARY_PROTOCOL
    return LEGACY_PROTOCOL


def condition_for_fractions(diffact_fraction: int, petri_fraction: int) -> str:
    if diffact_fraction == petri_fraction:
        return COUPLED_CONDITION
    if petri_fraction == 100:
        return DIFFACT_LOW_PETRI_FULL_CONDITION
    if diffact_fraction == 100:
        return DIFFACT_FULL_PETRI_LOW_CONDITION
    raise ValueError(
        "Only the coupled curve, full-train bounds, and 25% process-scarcity "
        f"control are approved; got DiffAct={diffact_fraction}, Petri={petri_fraction}"
    )


def default_petri_discovery_source(condition: str) -> str:
    if condition == COUPLED_CONDITION:
        return PETRI_DISCOVERY_NESTED
    if condition == DIFFACT_LOW_PETRI_FULL_CONDITION:
        return PETRI_DISCOVERY_FULL_TRAIN
    if condition == ORACLE_TEST_FOLD_CONDITION:
        return PETRI_DISCOVERY_OFFICIAL_TEST
    if condition == DIFFACT_FULL_PETRI_LOW_CONDITION:
        return PETRI_DISCOVERY_NESTED
    raise ValueError(f"Unknown condition for Petri discovery source: {condition!r}")


def resolve_petri_train_cases(
    experiment_dir: Path,
    *,
    seed: int,
    petri_fraction: Optional[int],
    discovery_source: str,
) -> Tuple[List[str], Path]:
    """Resolve Petri discovery case IDs and the manifest path that produced them."""
    if discovery_source == PETRI_DISCOVERY_NESTED:
        if petri_fraction is None:
            raise ValueError(
                "nested_train_fraction discovery requires an explicit petri_fraction"
            )
        path = (
            experiment_dir
            / "manifests"
            / f"seed_{seed}"
            / f"train_cases_frac_{petri_fraction}.txt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing Petri discovery manifest: {path}")
        return read_case_manifest(path), path
    if discovery_source == PETRI_DISCOVERY_FULL_TRAIN:
        # Always use the seed-independent official training pool.
        path = experiment_dir / "manifests" / "train_pool_cases.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing full-train Petri discovery manifest: {path}")
        return read_case_manifest(path), path
    if discovery_source == PETRI_DISCOVERY_OFFICIAL_TEST:
        path = experiment_dir / "manifests" / "test_cases.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing oracle Petri discovery manifest: {path}")
        return read_case_manifest(path), path
    raise ValueError(
        f"petri_discovery_source must be one of {list(PETRI_DISCOVERY_SOURCES)}; "
        f"got {discovery_source!r}"
    )


def validate_fraction_condition(
    *,
    diffact_fraction: int,
    petri_fraction: Optional[int],
    condition: str,
    petri_discovery_source: Optional[str] = None,
) -> None:
    if diffact_fraction not in FRACTIONS:
        raise ValueError(
            f"DiffAct fraction must be selected from {list(FRACTIONS)}; got "
            f"DiffAct={diffact_fraction}"
        )
    discovery_source = petri_discovery_source or default_petri_discovery_source(condition)
    expected_discovery_source = default_petri_discovery_source(condition)
    if discovery_source != expected_discovery_source:
        raise ValueError(
            f"{condition} requires "
            f"petri_discovery_source={expected_discovery_source!r}; "
            f"got {discovery_source!r}"
        )
    if condition == ORACLE_TEST_FOLD_CONDITION:
        if petri_fraction is not None:
            raise ValueError(
                "oracle_test_fold_petri requires petri_fraction=null "
                f"(got {petri_fraction})"
            )
        return
    if petri_fraction is None or petri_fraction not in FRACTIONS:
        raise ValueError(
            f"Petri fraction must be selected from {list(FRACTIONS)}; got "
            f"Petri={petri_fraction}"
        )
    expected = condition_for_fractions(diffact_fraction, petri_fraction)
    if condition != expected:
        raise ValueError(
            f"Condition {condition!r} does not match DiffAct={diffact_fraction}, "
            f"Petri={petri_fraction}; expected {expected!r}"
        )
    if condition == DIFFACT_LOW_PETRI_FULL_CONDITION:
        if petri_fraction != 100 or diffact_fraction >= 100:
            raise ValueError(
                "structural_prior_full_train_petri allows D_f+P_100 only for f<100; "
                f"got DiffAct={diffact_fraction}, Petri={petri_fraction}"
            )
        return
    approved_cross = (
        diffact_fraction != petri_fraction
        and {diffact_fraction, petri_fraction} == {25, 100}
    )
    if diffact_fraction != petri_fraction and not approved_cross:
        raise ValueError(
            "Crossed process-scarcity control is approved only as D100+P25; "
            "full-train structural bounds use structural_prior_full_train_petri"
        )


def resolve_fraction_pairs(
    *,
    legacy_fractions: Optional[Sequence[int]],
    diffact_fractions: Optional[Sequence[int]],
    petri_fractions: Optional[Sequence[int]],
    condition: Optional[str],
    petri_discovery_source: Optional[str] = None,
) -> List[Tuple[int, Optional[int], str]]:
    if condition == ORACLE_TEST_FOLD_CONDITION:
        if petri_fractions is not None:
            raise ValueError(
                "--petri-fractions cannot be combined with --condition "
                f"{ORACLE_TEST_FOLD_CONDITION}"
            )
        if legacy_fractions is not None and diffact_fractions is not None:
            raise ValueError(
                "--fractions cannot be combined with --diffact-fractions for oracle"
            )
        diffs = list(
            FRACTIONS
            if diffact_fractions is None and legacy_fractions is None
            else (diffact_fractions if diffact_fractions is not None else legacy_fractions)
        )
        raw_pairs: List[Tuple[int, Optional[int]]] = [
            (int(diffact_fraction), None) for diffact_fraction in diffs
        ]
    else:
        explicit = diffact_fractions is not None or petri_fractions is not None
        if explicit:
            if legacy_fractions is not None:
                raise ValueError(
                    "--fractions cannot be combined with --diffact-fractions/--petri-fractions"
                )
            if diffact_fractions is None or petri_fractions is None:
                raise ValueError(
                    "--diffact-fractions and --petri-fractions must be supplied together "
                    f"(except for {ORACLE_TEST_FOLD_CONDITION})"
                )
            if len(diffact_fractions) != len(petri_fractions):
                raise ValueError(
                    "--diffact-fractions and --petri-fractions must have equal lengths; "
                    "pairs are zipped in the given order"
                )
            raw_pairs = [
                (int(diffact_fraction), int(petri_fraction))
                for diffact_fraction, petri_fraction in zip(diffact_fractions, petri_fractions)
            ]
        else:
            fractions = list(FRACTIONS if legacy_fractions is None else legacy_fractions)
            legacy_condition = condition or COUPLED_CONDITION
            if legacy_condition == COUPLED_CONDITION:
                raw_pairs = [(fraction, fraction) for fraction in fractions]
            elif legacy_condition == DIFFACT_LOW_PETRI_FULL_CONDITION:
                raw_pairs = [(fraction, 100) for fraction in fractions]
            elif legacy_condition == ORACLE_TEST_FOLD_CONDITION:
                raw_pairs = [(fraction, None) for fraction in fractions]
            else:
                raw_pairs = [(100, fraction) for fraction in fractions]

    resolved: List[Tuple[int, Optional[int], str]] = []
    seen: set[Tuple[int, Optional[int], str]] = set()
    for diffact_fraction, petri_fraction in raw_pairs:
        if (
            condition == DIFFACT_LOW_PETRI_FULL_CONDITION
            and int(diffact_fraction) == 100
            and petri_fraction == 100
        ):
            continue
        if condition == ORACLE_TEST_FOLD_CONDITION:
            resolved_condition = ORACLE_TEST_FOLD_CONDITION
        else:
            resolved_condition = condition or condition_for_fractions(
                int(diffact_fraction), int(petri_fraction)  # type: ignore[arg-type]
            )
        validate_fraction_condition(
            diffact_fraction=int(diffact_fraction),
            petri_fraction=None if petri_fraction is None else int(petri_fraction),
            condition=resolved_condition,
            petri_discovery_source=petri_discovery_source,
        )
        item = (
            int(diffact_fraction),
            None if petri_fraction is None else int(petri_fraction),
            resolved_condition,
        )
        if item not in seen:
            resolved.append(item)
            seen.add(item)
    if not resolved:
        raise ValueError(
            "D100+P100 is the shared coupled reference and is not emitted under a "
            "crossed-control condition"
        )
    return resolved


def load_softmax_case_map(softmax_dir: Path) -> Dict[str, str]:
    case_map_path = softmax_dir / "case_id_map.csv"
    if case_map_path.is_file():
        case_map = pd.read_csv(case_map_path)
        return {
            str(row["original_case_id"]): str(row["sequential_case_id"])
            for _, row in case_map.iterrows()
        }

    video_index_path = softmax_dir / "video_index_map.txt"
    if video_index_path.is_file():
        mapping: Dict[str, str] = {}
        with video_index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"Malformed video index row in {video_index_path}: {line!r}")
                sequential_case_id, original_case_id = parts
                mapping[str(original_case_id)] = str(sequential_case_id)
        return mapping

    raise FileNotFoundError(
        f"Missing softmax case map: expected {case_map_path} or {video_index_path}"
    )


def apply_temperature_to_softmax_list(
    softmax_list: Sequence[np.ndarray],
    temperature: float,
) -> List[np.ndarray]:
    calibrated: List[np.ndarray] = []
    for mat in softmax_list:
        logits = inverse_softmax(np.asarray(mat, dtype=np.float64))
        calibrated.append(softmax_numpy(logits / float(temperature), axis=0).astype(np.float32))
    return calibrated


def build_eval_df_and_softmax(
    *,
    data_root: Path,
    dataset: str,
    eval_cases: Sequence[str],
    eval_softmax_dir: Path,
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    label_to_idx, labels = load_mapping(data_root / dataset / "mapping.txt")
    n_classes = len(labels)
    eval_case_to_source = load_softmax_case_map(eval_softmax_dir)

    rows: List[Dict[str, str]] = []
    softmax_list: List[np.ndarray] = []
    for seq_id, case_id in enumerate(eval_cases):
        source_case = eval_case_to_source.get(str(case_id))
        if source_case is None:
            raise ValueError(f"{case_id}: missing from softmax case map in {eval_softmax_dir}")
        labels_idx = load_ground_truth_indices(data_root, dataset, case_id, label_to_idx)
        mat_path = eval_softmax_dir / f"{source_case}.npy"
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing eval softmax matrix: {mat_path}")
        mat = np.load(mat_path)
        if mat.shape[0] != n_classes:
            raise ValueError(f"{mat_path}: classes={mat.shape[0]} expected {n_classes}")
        if mat.shape[1] != len(labels_idx):
            raise ValueError(f"{case_id}: softmax T={mat.shape[1]} ground truth T={len(labels_idx)}")
        seq_s = str(seq_id)
        for lab in labels_idx:
            rows.append({"case:concept:name": seq_s, "concept:name": str(lab)})
        softmax_list.append(np.asarray(mat, dtype=np.float32))
    return pd.DataFrame(rows), softmax_list


def learn_temperature_from_validation(
    *,
    data_root: Path,
    dataset: str,
    experiment_dir: Path,
    seed: int,
    fraction: int,
    temp_bounds: Tuple[float, float],
) -> float:
    val_cases = read_case_manifest(experiment_dir / "manifests" / "val_cases.txt")
    val_softmax_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}" / "softmax_val"
    if not val_softmax_dir.is_dir():
        raise FileNotFoundError(
            f"Validation softmax is required for calibration but was not found: {val_softmax_dir}"
        )
    val_df, val_softmax = build_eval_df_and_softmax(
        data_root=data_root,
        dataset=dataset,
        eval_cases=val_cases,
        eval_softmax_dir=val_softmax_dir,
    )
    temperature = calibrate_probabilities(
        softmax_list=val_softmax,
        df=val_df,
        temp_bounds=temp_bounds,
        only_return_temperature=True,
    )
    return float(temperature)


def iter_requested(
    seeds: Iterable[int],
    fraction_pairs: Iterable[Tuple[int, Optional[int], str]],
    max_runs: Optional[int],
):
    count = 0
    for seed in seeds:
        for diffact_fraction, petri_fraction, condition in fraction_pairs:
            if max_runs is not None and count >= max_runs:
                return
            count += 1
            yield seed, diffact_fraction, petri_fraction, condition


def levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ai in enumerate(a, start=1):
        cur[0] = i
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[-1]


def build_combined_inputs(
    *,
    data_root: Path,
    dataset: str,
    train_cases: Sequence[str],
    eval_cases: Sequence[str],
    eval_softmax_dir: Path,
    out_dir: Path,
    temperature: Optional[float] = None,
    petri_discovery_representation: str = "run_collapsed_segments",
) -> Tuple[pd.DataFrame, List[np.ndarray], List[str], List[str], Path]:
    if petri_discovery_representation not in {"run_collapsed_segments", "frame_events"}:
        raise ValueError(
            "petri_discovery_representation must be one of "
            "{'run_collapsed_segments', 'frame_events'}"
        )
    label_to_idx, labels = load_mapping(data_root / dataset / "mapping.txt")
    n_classes = len(labels)
    rows: List[Dict[str, str]] = []
    softmax_list: List[np.ndarray] = []
    train_seq_ids: List[str] = []
    eval_seq_ids: List[str] = []
    mapping_rows: List[Dict[str, Any]] = []
    train_input_events = 0
    train_original_frames = 0

    for seq_id, case_id in enumerate(train_cases):
        full_labels_idx = load_ground_truth_indices(data_root, dataset, case_id, label_to_idx)
        train_original_frames += len(full_labels_idx)
        if petri_discovery_representation == "run_collapsed_segments":
            labels_idx = list(collapse_runs(full_labels_idx))
        else:
            labels_idx = full_labels_idx
        train_input_events += len(labels_idx)
        seq_s = str(seq_id)
        train_seq_ids.append(seq_s)
        for lab in labels_idx:
            rows.append({"case:concept:name": seq_s, "concept:name": str(lab)})
        softmax_list.append(one_hot_softmax(labels_idx, n_classes))
        mapping_rows.append(
            {
                "role": "train",
                "original_case_id": case_id,
                "sequential_case_id": seq_s,
                "softmax_list_index": seq_id,
                "source_softmax_case_id": "",
                "original_frame_events": len(full_labels_idx),
                "petri_discovery_events": len(labels_idx),
            }
        )

    offset = len(train_cases)
    eval_case_to_source = load_softmax_case_map(eval_softmax_dir)
    eval_frame_events = 0

    for j, case_id in enumerate(eval_cases):
        seq_id = offset + j
        seq_s = str(seq_id)
        source_case = eval_case_to_source.get(str(case_id))
        if source_case is None:
            raise ValueError(f"{case_id}: missing from softmax case map in {eval_softmax_dir}")
        labels_idx = load_ground_truth_indices(data_root, dataset, case_id, label_to_idx)
        mat_path = eval_softmax_dir / f"{source_case}.npy"
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing eval softmax matrix: {mat_path}")
        mat = np.load(mat_path)
        if mat.shape[0] != n_classes:
            raise ValueError(f"{mat_path}: classes={mat.shape[0]} expected {n_classes}")
        if mat.shape[1] != len(labels_idx):
            raise ValueError(f"{case_id}: softmax T={mat.shape[1]} ground truth T={len(labels_idx)}")
        if temperature is not None:
            mat = apply_temperature_to_softmax_list([mat], temperature)[0]
        eval_seq_ids.append(seq_s)
        eval_frame_events += len(labels_idx)
        for lab in labels_idx:
            rows.append({"case:concept:name": seq_s, "concept:name": str(lab)})
        softmax_list.append(np.asarray(mat, dtype=np.float32))
        mapping_rows.append(
            {
                "role": "eval",
                "original_case_id": case_id,
                "sequential_case_id": seq_s,
                "softmax_list_index": seq_id,
                "source_softmax_case_id": source_case,
                "original_frame_events": len(labels_idx),
                "petri_discovery_events": "",
            }
        )

    df = pd.DataFrame(rows)
    map_path = out_dir / "combined_case_id_map.csv"
    pd.DataFrame(mapping_rows).to_csv(map_path, index=False)

    checks = {
        "n_train_cases": len(train_cases),
        "n_eval_cases": len(eval_cases),
        "n_combined_cases": len(train_cases) + len(eval_cases),
        "n_softmax_matrices": len(softmax_list),
        "all_softmax_lengths_match_ground_truth": True,
        "class_count": n_classes,
        "class_mapping_path": str(data_root / dataset / "mapping.txt"),
        "softmax_source_dir": str(eval_softmax_dir),
        "temperature_applied": temperature,
        "petri_discovery_representation": petri_discovery_representation,
        "train_original_frame_events": train_original_frames,
        "train_petri_discovery_events": train_input_events,
        "eval_frame_events": eval_frame_events,
        "eval_postprocessing_representation": "frame_events",
    }
    (out_dir / "alignment_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return df, softmax_list, train_seq_ids, eval_seq_ids, map_path


def resolve_case_artifact(
    softmax_dir: Path,
    *,
    original_case_id: str,
    source_case_id: str,
    suffix: str,
) -> Optional[Path]:
    candidates = [
        softmax_dir / f"{source_case_id}{suffix}",
        softmax_dir / f"{original_case_id}{suffix}",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def csv_identifier(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def prediction_labels_from_array(
    array: np.ndarray,
    *,
    n_frames: int,
    artifact_path: Path,
    probabilities: bool,
) -> List[str]:
    values = np.asarray(array)
    if probabilities:
        values = np.squeeze(values)
        if values.ndim == 1:
            labels = values
        elif values.ndim == 2 and values.shape[1] == n_frames:
            labels = np.argmax(values, axis=0)
        elif values.ndim == 2 and values.shape[0] == n_frames:
            labels = np.argmax(values, axis=1)
        else:
            raise ValueError(
                f"{artifact_path}: raw probability shape {values.shape} is not aligned "
                f"to {n_frames} frames"
            )
    else:
        values = np.squeeze(values)
        if values.ndim != 1:
            raise ValueError(
                f"{artifact_path}: official prediction must be one-dimensional after squeeze; "
                f"got {values.shape}"
            )
        labels = values
    if len(labels) != n_frames:
        raise ValueError(
            f"{artifact_path}: prediction length={len(labels)} expected {n_frames}"
        )
    return [str(int(x)) for x in labels]


def attach_diffact_comparators(
    *,
    results_df: pd.DataFrame,
    combined_map_path: Path,
    softmax_dir: Path,
    require_comparators: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attach official DiffAct predictions and pre-postprocessing raw argmax when exported."""
    map_df = pd.read_csv(combined_map_path)
    eval_rows = map_df[map_df["role"] == "eval"]
    specs = {
        "official_diffact": ("_pred.npy", "official_diffact_activity", False),
        "raw_diffact": ("_raw.npy", "raw_diffact_activity", True),
    }
    provenance: Dict[str, Any] = {
        "softmax_source_dir": str(softmax_dir),
        "required": bool(require_comparators),
        "streams": {},
    }
    result = results_df.copy()

    for stream, (suffix, column, probabilities) in specs.items():
        resolved: List[Tuple[pd.Series, Path]] = []
        missing: List[str] = []
        for _, row in eval_rows.iterrows():
            original_case = csv_identifier(row["original_case_id"])
            source_case = csv_identifier(row["source_softmax_case_id"])
            path = resolve_case_artifact(
                softmax_dir,
                original_case_id=original_case,
                source_case_id=source_case,
                suffix=suffix,
            )
            if path is None:
                missing.append(original_case)
            else:
                resolved.append((row, path))

        if missing and resolved:
            raise FileNotFoundError(
                f"Incomplete {stream} comparator stream in {softmax_dir}; "
                f"missing cases={missing}"
            )
        if missing:
            provenance["streams"][stream] = {
                "available": False,
                "filename_suffix": suffix,
                "missing_cases": missing,
            }
            if require_comparators:
                raise FileNotFoundError(
                    f"Required {stream} comparator stream is absent from {softmax_dir}; "
                    f"expected {{case_id}}{suffix}"
                )
            continue

        result[column] = ""
        artifact_paths: Dict[str, str] = {}
        for row, path in resolved:
            seq_case = csv_identifier(row["sequential_case_id"])
            mask = result["case:concept:name"].astype(str) == seq_case
            n_frames = int(mask.sum())
            labels = prediction_labels_from_array(
                np.load(path),
                n_frames=n_frames,
                artifact_path=path,
                probabilities=probabilities,
            )
            result.loc[mask, column] = labels
            artifact_paths[csv_identifier(row["original_case_id"])] = str(path)
        provenance["streams"][stream] = {
            "available": True,
            "filename_suffix": suffix,
            "artifact_paths": artifact_paths,
        }
    return result, provenance


def preflight_required_diffact_comparators(
    *,
    combined_map_path: Path,
    softmax_dir: Path,
) -> None:
    """Fail before expensive decoding when required comparator artifacts are incomplete."""
    map_df = pd.read_csv(combined_map_path)
    eval_rows = map_df[map_df["role"] == "eval"]
    missing_by_suffix: Dict[str, List[str]] = {}
    for suffix in ("_pred.npy", "_raw.npy"):
        missing: List[str] = []
        for _, row in eval_rows.iterrows():
            original_case = csv_identifier(row["original_case_id"])
            source_case = csv_identifier(row["source_softmax_case_id"])
            if (
                resolve_case_artifact(
                    softmax_dir,
                    original_case_id=original_case,
                    source_case_id=source_case,
                    suffix=suffix,
                )
                is None
            ):
                missing.append(original_case)
        if missing:
            missing_by_suffix[suffix] = missing
    if missing_by_suffix:
        raise FileNotFoundError(
            "Required official DiffAct comparator artifacts are incomplete in "
            f"{softmax_dir}: {missing_by_suffix}"
        )


def allowed_transition_set(df: pd.DataFrame, train_cases: Sequence[str]) -> set[tuple[str, str]]:
    train_set = set(str(c) for c in train_cases)
    allowed: set[tuple[str, str]] = set()
    for case, g in df[df["case:concept:name"].isin(train_set)].groupby("case:concept:name", sort=False):
        seq = [str(x) for x in g["concept:name"].tolist()]
        collapsed = collapse_runs(seq)
        allowed.update((str(a), str(b)) for a, b in zip(collapsed, collapsed[1:]))
    return allowed


def illegal_transition_pct(
    results_df: pd.DataFrame,
    pred_col: str,
    allowed: set[tuple[str, str]],
) -> float:
    illegal = 0
    total = 0
    for _, g in results_df.groupby("case:concept:name", sort=False):
        collapsed = collapse_runs([str(x) for x in g[pred_col].tolist()])
        for a, b in zip(collapsed, collapsed[1:]):
            total += 1
            if (str(a), str(b)) not in allowed:
                illegal += 1
    return 100.0 * illegal / total if total else 0.0


def transition_viterbi_decode(
    softmax_matrix: np.ndarray,
    allowed_transitions: set[tuple[str, str]],
    *,
    illegal_penalty: float,
    eps: float,
) -> Tuple[List[str], List[float]]:
    """Decode all frames with a soft penalty for run-to-run transitions absent from training."""
    probs = np.asarray(softmax_matrix, dtype=np.float64)
    n_classes, n_steps = probs.shape
    log_probs = np.log(np.maximum(probs, eps))

    transition_score = np.full((n_classes, n_classes), -float(illegal_penalty), dtype=np.float64)
    for i in range(n_classes):
        transition_score[i, i] = 0.0
    for src, dst in allowed_transitions:
        try:
            transition_score[int(src), int(dst)] = 0.0
        except (TypeError, ValueError):
            continue

    scores = np.empty((n_classes, n_steps), dtype=np.float64)
    backptr = np.zeros((n_classes, n_steps), dtype=np.int16)
    scores[:, 0] = log_probs[:, 0]
    for t in range(1, n_steps):
        candidates = scores[:, t - 1][:, None] + transition_score
        backptr[:, t] = np.argmax(candidates, axis=0).astype(np.int16)
        scores[:, t] = candidates[backptr[:, t], np.arange(n_classes)] + log_probs[:, t]

    path = np.empty(n_steps, dtype=np.int16)
    path[-1] = int(np.argmax(scores[:, -1]))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backptr[path[t], t]

    preds = [str(int(x)) for x in path]
    move_costs = [0.0]
    for prev, cur in zip(preds, preds[1:]):
        if prev == cur or (prev, cur) in allowed_transitions:
            move_costs.append(0.0)
        else:
            move_costs.append(float(illegal_penalty))
    return preds, move_costs


def build_recovery_records(
    *,
    case_id: str,
    sktr_preds: Sequence[str],
    argmax_preds: Sequence[str],
    ground_truth_sequence: Sequence[str],
    softmax_matrix: np.ndarray,
    sktr_move_costs: Sequence[float],
    prob_threshold: float,
    postprocess_seconds: float,
) -> Tuple[pd.DataFrame, float, float]:
    seq_len = len(ground_truth_sequence)
    if len(sktr_preds) != seq_len or len(argmax_preds) != seq_len:
        raise ValueError(
            f"{case_id}: prediction length mismatch "
            f"sktr={len(sktr_preds)} argmax={len(argmax_preds)} gt={seq_len}"
        )
    sktr_array = np.asarray(sktr_preds, dtype=str)
    argmax_array = np.asarray(argmax_preds, dtype=str)
    gt_array = np.asarray(ground_truth_sequence, dtype=str)
    sktr_matches = sktr_array == gt_array
    argmax_matches = argmax_array == gt_array

    all_probs: List[List[float]] = []
    all_activities: List[List[str]] = []
    for t in range(seq_len):
        probs = softmax_matrix[:, t]
        keep = np.where(probs >= prob_threshold)[0]
        all_probs.append([round(float(probs[i]), 2) for i in keep])
        all_activities.append([str(int(i)) for i in keep])

    records = pd.DataFrame(
        {
            "case:concept:name": [case_id] * seq_len,
            "step": list(range(seq_len)),
            "sktr_activity": list(sktr_preds),
            "argmax_activity": list(argmax_preds),
            "ground_truth": list(ground_truth_sequence),
            "all_probs": all_probs,
            "all_activities": all_activities,
            "is_correct": sktr_matches,
            "cumulative_accuracy": np.cumsum(sktr_matches) / np.arange(1, seq_len + 1),
            "sktr_move_cost": [round(float(x), 4) for x in sktr_move_costs],
            "postprocess_seconds": [float(postprocess_seconds)] * seq_len,
        }
    )
    return records, float(np.mean(sktr_matches)), float(np.mean(argmax_matches))


def write_case_output_atomic(records_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    records_df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def run_transition_viterbi_recovery(
    *,
    df: pd.DataFrame,
    softmax_list: Sequence[np.ndarray],
    eval_seq_ids: Sequence[str],
    allowed_transitions: set[tuple[str, str]],
    prob_threshold: float,
    illegal_penalty: float,
    case_output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    all_records: List[pd.DataFrame] = []
    sktr_accs: List[float] = []
    argmax_accs: List[float] = []
    case_output_dir.mkdir(parents=True, exist_ok=True)
    for seq_case in eval_seq_ids:
        seq_s = str(seq_case)
        case_start = time.perf_counter()
        case_df = df[df["case:concept:name"].astype(str) == seq_s]
        if case_df.empty:
            raise ValueError(f"{seq_s}: no evaluation rows found")
        mat = np.asarray(softmax_list[int(seq_s)], dtype=np.float32)
        gt = [str(x) for x in case_df["concept:name"].tolist()]
        argmax_preds = [str(int(x)) for x in np.argmax(mat, axis=0)]
        sktr_preds, move_costs = transition_viterbi_decode(
            mat,
            allowed_transitions,
            illegal_penalty=illegal_penalty,
            eps=prob_threshold,
        )
        records, sktr_acc, argmax_acc = build_recovery_records(
            case_id=seq_s,
            sktr_preds=sktr_preds,
            argmax_preds=argmax_preds,
            ground_truth_sequence=gt,
            softmax_matrix=mat,
            sktr_move_costs=move_costs,
            prob_threshold=prob_threshold,
            postprocess_seconds=time.perf_counter() - case_start,
        )
        write_case_output_atomic(records, case_output_dir / f"{seq_s}.csv")
        all_records.append(records)
        sktr_accs.append(sktr_acc)
        argmax_accs.append(argmax_acc)

    results_df = pd.concat(all_records, ignore_index=True)
    accuracy_dict = {"sktr_accuracy": sktr_accs, "argmax_accuracy": argmax_accs}
    return results_df, accuracy_dict


def mean_run_collapsed_distance(results_df: pd.DataFrame, pred_col: str) -> float:
    distances: List[int] = []
    for _, g in results_df.groupby("case:concept:name", sort=False):
        gt = collapse_runs([str(x) for x in g["ground_truth"].tolist()])
        pred = collapse_runs([str(x) for x in g[pred_col].tolist()])
        distances.append(levenshtein_distance(pred, gt))
    return float(np.mean(distances)) if distances else 0.0


def run_collapsed_distance_for_case(case_df: pd.DataFrame, pred_col: str) -> int:
    gt = collapse_runs([str(x) for x in case_df["ground_truth"].tolist()])
    pred = collapse_runs([str(x) for x in case_df[pred_col].tolist()])
    return levenshtein_distance(pred, gt)


def comparison_method_map(results_df: pd.DataFrame, post_method: str) -> Dict[str, str]:
    methods: Dict[str, str] = {}
    if "raw_diffact_activity" in results_df.columns:
        methods["raw_argmax"] = "raw_diffact_activity"
        methods["canonical_softmax_argmax"] = "argmax_activity"
    else:
        methods["raw_argmax"] = "argmax_activity"
    if "official_diffact_activity" in results_df.columns:
        methods["official_diffact"] = "official_diffact_activity"
    methods[post_method] = "sktr_activity"
    return methods


def build_per_case_metric_rows(
    *,
    dataset: str,
    seed: int,
    diffact_fraction: int,
    petri_fraction: Optional[int],
    condition: str,
    split: str,
    post_method: str,
    results_df: pd.DataFrame,
    combined_map_path: Path,
    mapping_path: Path,
    allowed_transitions: set[tuple[str, str]],
    calibrated: bool,
) -> pd.DataFrame:
    map_df = pd.read_csv(combined_map_path)
    seq_to_original = {
        str(row["sequential_case_id"]): str(row["original_case_id"])
        for _, row in map_df[map_df["role"] == "eval"].iterrows()
    }
    rows: List[Dict[str, Any]] = []
    method_map = comparison_method_map(results_df, post_method)
    for seq_case, case_df in results_df.groupby("case:concept:name", sort=False):
        seq_case_s = str(seq_case)
        original_case = seq_to_original.get(seq_case_s, seq_case_s)
        for method, pred_col in method_map.items():
            metrics = compute_tas_metrics_asformer(
                case_df,
                pred_col=pred_col,
                gt_col="ground_truth",
                case_col="case:concept:name",
                dataset_name=dataset,
                mapping_path=mapping_path,
            )
            metric_values = {
                "frame_accuracy": metrics["acc"],
                "edit": metrics["edit"],
                "f1@10": metrics["f1@10"],
                "f1@25": metrics["f1@25"],
                "f1@50": metrics["f1@50"],
                "run_collapsed_levenshtein": float(run_collapsed_distance_for_case(case_df, pred_col)),
                "illegal_transition_pct": illegal_transition_pct(case_df, pred_col, allowed_transitions),
            }
            if method == post_method and post_method == "petri_conformance":
                metric_values["alignment_cost"] = float(case_df["sktr_move_cost"].sum())
            elif method == post_method and "sktr_move_cost" in case_df.columns:
                metric_values["transition_penalty_cost"] = float(case_df["sktr_move_cost"].sum())
            for metric_name, metric_value in metric_values.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "fraction": diffact_fraction,
                        "diffact_fraction": diffact_fraction,
                        "petri_fraction": petri_fraction,
                        "condition": condition,
                        "split": split,
                        "method": method,
                        "calibrated": bool(calibrated),
                        "original_case_id": original_case,
                        "sequential_case_id": seq_case_s,
                        "n_frames": int(len(case_df)),
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                    }
                )
    return pd.DataFrame(rows)


def build_metric_rows(
    *,
    dataset: str,
    seed: int,
    diffact_fraction: int,
    petri_fraction: Optional[int],
    condition: str,
    split: str,
    post_method: str,
    results_df: pd.DataFrame,
    mapping_path: Path,
    allowed_transitions: set[tuple[str, str]],
    total_seconds: float,
    calibrated: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    method_map = comparison_method_map(results_df, post_method)
    for method, pred_col in method_map.items():
        metrics = compute_tas_metrics_asformer(
            results_df,
            pred_col=pred_col,
            gt_col="ground_truth",
            case_col="case:concept:name",
            dataset_name=dataset,
            mapping_path=mapping_path,
        )
        metric_values = {
            "frame_accuracy": metrics["acc"],
            "edit": metrics["edit"],
            "f1@10": metrics["f1@10"],
            "f1@25": metrics["f1@25"],
            "f1@50": metrics["f1@50"],
            "run_collapsed_levenshtein": mean_run_collapsed_distance(results_df, pred_col),
            "illegal_transition_pct": illegal_transition_pct(results_df, pred_col, allowed_transitions),
        }
        if method == post_method and post_method == "petri_conformance":
            metric_values["mean_alignment_cost"] = float(results_df["sktr_move_cost"].mean())
            metric_values["total_alignment_cost"] = float(results_df["sktr_move_cost"].sum())
        elif method == post_method and "sktr_move_cost" in results_df.columns:
            metric_values["mean_transition_penalty"] = float(results_df["sktr_move_cost"].mean())
            metric_values["total_transition_penalty"] = float(results_df["sktr_move_cost"].sum())
        for metric_name, metric_value in metric_values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fraction": diffact_fraction,
                    "diffact_fraction": diffact_fraction,
                    "petri_fraction": petri_fraction,
                    "condition": condition,
                    "split": split,
                    "method": method,
                    "calibrated": bool(calibrated),
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "n_cases": int(results_df["case:concept:name"].nunique()),
                    "total_seconds": float(total_seconds),
                }
            )
    return pd.DataFrame(rows)


def build_runtime_tables(
    *,
    seed: int,
    diffact_fraction: int,
    petri_fraction: Optional[int],
    condition: str,
    split: str,
    method: str,
    results_df: pd.DataFrame,
    combined_map_path: Path,
    total_seconds: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_cases = max(1, results_df["case:concept:name"].nunique())
    per_case_rows: List[Dict[str, Any]] = []
    if "postprocess_seconds" in results_df.columns:
        map_df = pd.read_csv(combined_map_path)
        seq_to_original = {
            str(row["sequential_case_id"]): str(row["original_case_id"])
            for _, row in map_df[map_df["role"] == "eval"].iterrows()
        }
        for seq_case, group in results_df.groupby("case:concept:name", sort=False):
            seconds = pd.to_numeric(group["postprocess_seconds"], errors="coerce").dropna()
            if seconds.empty:
                continue
            seq_case_s = str(seq_case)
            per_case_rows.append(
                {
                    "seed": seed,
                    "fraction": diffact_fraction,
                    "diffact_fraction": diffact_fraction,
                    "petri_fraction": petri_fraction,
                    "condition": condition,
                    "method": method,
                    "split": split,
                    "original_case_id": seq_to_original.get(seq_case_s, seq_case_s),
                    "sequential_case_id": seq_case_s,
                    "n_frames": int(len(group)),
                    "seconds": float(seconds.iloc[0]),
                }
            )

    per_case_runtime = pd.DataFrame(
        per_case_rows,
        columns=[
            "seed",
            "fraction",
            "diffact_fraction",
            "petri_fraction",
            "condition",
            "method",
            "split",
            "original_case_id",
            "sequential_case_id",
            "n_frames",
            "seconds",
        ],
    )
    if per_case_runtime.empty:
        mean_seconds = float(total_seconds / n_cases)
        median_seconds = float("nan")
    else:
        mean_seconds = float(per_case_runtime["seconds"].mean())
        median_seconds = float(per_case_runtime["seconds"].median())

    runtime_df = pd.DataFrame(
        [
            {
                "seed": seed,
                "fraction": diffact_fraction,
                "diffact_fraction": diffact_fraction,
                "petri_fraction": petri_fraction,
                "condition": condition,
                "method": method,
                "split": split,
                "total_seconds": float(total_seconds),
                "mean_seconds_per_case": mean_seconds,
                "median_seconds_per_case": median_seconds,
            }
        ]
    )
    return runtime_df, per_case_runtime


def validation_selection_score(metrics_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    petri = metrics_df[
        (
            metrics_df["method"].isin(
                ["petri_conformance", "petri_transition_viterbi"]
            )
        )
        & (metrics_df["metric_name"].isin(["edit", "f1@10", "f1@25", "f1@50"]))
    ]
    values = {
        str(row["metric_name"]): float(row["metric_value"])
        for _, row in petri.iterrows()
    }
    score_terms = [values[name] for name in ("edit", "f1@10", "f1@25", "f1@50") if name in values]
    score = float(np.mean(score_terms)) if score_terms else float("-inf")
    return score, values


def select_validation_hyperparameters(
    *,
    experiment_dir: Path,
    condition: str,
    seed: int,
    diffact_fraction: int,
    petri_fraction: Optional[int],
    method: str,
    protocol: str,
) -> Optional[Dict[str, Any]]:
    """Select Petri postprocessing parameters from validation metrics only."""
    if condition == ORACLE_TEST_FOLD_CONDITION or petri_fraction is None:
        factorized_root = (
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"diffact_frac_{diffact_fraction}"
            / "petri_source_test"
        )
    else:
        factorized_root = (
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"diffact_frac_{diffact_fraction}"
            / f"petri_frac_{petri_fraction}"
        )
    legacy_root = (
        experiment_dir / "petri" / condition / f"seed_{seed}" / f"frac_{diffact_fraction}"
    )
    roots = [factorized_root]
    if legacy_root != factorized_root:
        roots.append(legacy_root)
    if not any(root.exists() for root in roots):
        return None
    candidates: List[Dict[str, Any]] = []
    metric_paths: set[Path] = set()
    for run_root in roots:
        metric_paths.update(run_root.glob("val*/metrics.csv"))
        metric_paths.update(run_root.glob("*/val*/metrics.csv"))
    for metrics_path in sorted(metric_paths):
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            continue
        config_path = metrics_path.parent / "postprocess_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        if str(config.get("method")) != str(method):
            continue
        score, score_metrics = validation_selection_score(metrics_df)
        candidates.append(
            {
                "candidate_id": metrics_path.parent.name,
                "metrics_path": str(metrics_path),
                "config_path": str(config_path),
                "score": score,
                "score_metrics": score_metrics,
                "method": config.get("method"),
                "chunk_size": config.get("chunk_size"),
                "prob_threshold": config.get("prob_threshold"),
                "transition_illegal_penalty": config.get("transition_illegal_penalty"),
                "cost_function": config.get("cost_function"),
                "model_move_cost": config.get("model_move_cost"),
                "log_move_cost": config.get("log_move_cost"),
                "tau_move_cost": config.get("tau_move_cost"),
                "conformance_switch_penalty_weight": config.get(
                    "conformance_switch_penalty_weight"
                ),
                "use_calibration": config.get("use_calibration"),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    selected = dict(candidates[0])
    payload = {
        "dataset_selection_scope": "validation_only",
        "selection_uses_test_data": False,
        "selection_metric": "mean(edit,f1@10,f1@25,f1@50)",
        "condition": condition,
        "seed": seed,
        "fraction": diffact_fraction,
        "diffact_fraction": diffact_fraction,
        "petri_fraction": petri_fraction,
        "n_candidates": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }
    selection_root = (
        legacy_root
        if (
            protocol == LEGACY_PROTOCOL
            and condition in {COUPLED_CONDITION, DIFFACT_LOW_PETRI_FULL_CONDITION}
        )
        else factorized_root
    )
    selection_path = selection_root / method / "selected_hyperparameters.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["selection_path"] = str(selection_path)
    return payload


def legacy_output_matches_method(out_dir: Path, method: str, calibrated: bool) -> bool:
    config_path = out_dir / "postprocess_config.json"
    metrics_path = out_dir / "metrics.csv"
    if not config_path.is_file() or not metrics_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return False
    return str(config.get("method")) == str(method) and bool(config.get("calibrated", False)) == bool(calibrated)


def resolve_postprocess_output_dir(
    *,
    experiment_dir: Path,
    condition: str,
    seed: int,
    diffact_fraction: int,
    petri_fraction: Optional[int],
    method: str,
    split_dir: str,
    protocol: str,
    calibrated: bool,
) -> Path:
    if condition == ORACLE_TEST_FOLD_CONDITION or petri_fraction is None:
        return (
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"diffact_frac_{diffact_fraction}"
            / "petri_source_test"
            / method
            / split_dir
        )
    if (
        protocol == LEGACY_PROTOCOL
        and condition in {COUPLED_CONDITION, DIFFACT_LOW_PETRI_FULL_CONDITION}
    ):
        legacy_root = (
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"frac_{diffact_fraction}"
        )
        legacy = legacy_root / split_dir
        if legacy_output_matches_method(legacy, method, calibrated):
            return legacy
        return legacy_root / method / split_dir
    run_root = (
        experiment_dir
        / "petri"
        / condition
        / f"seed_{seed}"
        / f"diffact_frac_{diffact_fraction}"
        / f"petri_frac_{petri_fraction}"
    )
    return run_root / method / split_dir


def validate_primary_decoder_config(
    *,
    split: str,
    method: str,
    chunk_size: int,
    prob_threshold: float,
    model_move_cost: float,
    conformance_switch_penalty_weight: float,
    petri_discovery_representation: str,
    calibrated: bool,
) -> None:
    expected = {
        "split": "test",
        "method": PAPER_DIFFACT_SKTR_METHOD,
        "chunk_size": PAPER_DIFFACT_SKTR_CHUNK_SIZE,
        "prob_threshold": 1e-6,
        "model_move_cost": 1.0,
        "conformance_switch_penalty_weight": 1.0,
        "petri_discovery_representation": PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION,
        "calibrated": False,
    }
    actual = {
        "split": split,
        "method": method,
        "chunk_size": chunk_size,
        "prob_threshold": prob_threshold,
        "model_move_cost": model_move_cost,
        "conformance_switch_penalty_weight": conformance_switch_penalty_weight,
        "petri_discovery_representation": petri_discovery_representation,
        "calibrated": calibrated,
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            "Primary no-validation protocol requires the fixed canonical "
            f"petri_conformance decoder config; mismatches={mismatches}"
        )


def run_one(
    *,
    experiment_dir: Path,
    data_root: Path,
    dataset: str,
    seed: int,
    diffact_fraction: int,
    petri_fraction: Optional[int],
    split: str,
    condition: str,
    protocol: str,
    method: str,
    chunk_size: int,
    prob_threshold: float,
    model_move_cost: float,
    conformance_switch_penalty_weight: float,
    progress_log_interval_chunks: int,
    petri_discovery_representation: str,
    petri_discovery_source: str,
    transition_illegal_penalty: float,
    calibrated: bool,
    temp_bounds: Tuple[float, float],
    workers: int,
    inner_parallel: bool,
    runtime_pilot_case_limit: Optional[int],
    require_diffact_comparators: bool,
    force: bool,
) -> Path:
    validate_fraction_condition(
        diffact_fraction=diffact_fraction,
        petri_fraction=petri_fraction,
        condition=condition,
        petri_discovery_source=petri_discovery_source,
    )
    if runtime_pilot_case_limit is not None and runtime_pilot_case_limit < 1:
        raise ValueError("--runtime-pilot-case-limit must be positive")
    if protocol == PRIMARY_PROTOCOL:
        validate_primary_decoder_config(
            split=split,
            method=method,
            chunk_size=chunk_size,
            prob_threshold=prob_threshold,
            model_move_cost=model_move_cost,
            conformance_switch_penalty_weight=conformance_switch_penalty_weight,
            petri_discovery_representation=petri_discovery_representation,
            calibrated=calibrated,
        )
    if calibrated and split != "test":
        raise ValueError("Validation-learned calibration is only evaluated on --split test")
    split_dir = f"{split}_calibrated" if calibrated else split
    if runtime_pilot_case_limit is not None:
        split_dir = f"{split_dir}_runtime_pilot_n{runtime_pilot_case_limit}"
    out_dir = resolve_postprocess_output_dir(
        experiment_dir=experiment_dir,
        condition=condition,
        seed=seed,
        diffact_fraction=diffact_fraction,
        petri_fraction=petri_fraction,
        method=method,
        split_dir=split_dir,
        protocol=protocol,
        calibrated=calibrated,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    metrics_csv = out_dir / "metrics.csv"
    per_case_metrics_csv = out_dir / "per_case_metrics.csv"
    per_case_runtime_csv = out_dir / "per_case_runtime.csv"
    if (
        metrics_csv.is_file()
        and results_csv.is_file()
        and per_case_metrics_csv.is_file()
        and per_case_runtime_csv.is_file()
        and not force
    ):
        print(
            f"SKIP seed={seed} diffact_frac={diffact_fraction} "
            f"petri_frac={petri_fraction} split={split}: existing {metrics_csv}"
        )
        return metrics_csv

    diffact_manifest = (
        experiment_dir
        / "manifests"
        / f"seed_{seed}"
        / f"train_cases_frac_{diffact_fraction}.txt"
    )
    if not diffact_manifest.is_file():
        raise FileNotFoundError(f"Missing DiffAct training manifest: {diffact_manifest}")
    diffact_train_cases = read_case_manifest(diffact_manifest)
    train_cases, petri_manifest = resolve_petri_train_cases(
        experiment_dir,
        seed=seed,
        petri_fraction=petri_fraction,
        discovery_source=petri_discovery_source,
    )
    all_eval_cases = read_case_manifest(experiment_dir / "manifests" / f"{split}_cases.txt")
    eval_cases = (
        all_eval_cases[:runtime_pilot_case_limit]
        if runtime_pilot_case_limit is not None
        else all_eval_cases
    )
    softmax_dir = (
        experiment_dir
        / "diffact"
        / f"seed_{seed}"
        / f"frac_{diffact_fraction}"
        / f"softmax_{split}"
    )
    if not softmax_dir.is_dir():
        raise FileNotFoundError(
            f"Missing DiffAct softmax directory for seed={seed}, "
            f"diffact_fraction={diffact_fraction}, split={split}: {softmax_dir}. "
            "The producing train/infer task must complete before Petri postprocessing."
        )

    learned_temperature: Optional[float] = None
    if calibrated:
        learned_temperature = learn_temperature_from_validation(
            data_root=data_root,
            dataset=dataset,
            experiment_dir=experiment_dir,
            seed=seed,
            fraction=diffact_fraction,
            temp_bounds=temp_bounds,
        )
        (out_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fraction": diffact_fraction,
                    "diffact_fraction": diffact_fraction,
                    "petri_fraction": petri_fraction,
                    "condition": condition,
                    "temperature": learned_temperature,
                    "temp_bounds": list(temp_bounds),
                    "fit_split": "val",
                    "applied_split": split,
                    "uses_test_for_fitting": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    validation_selection: Optional[Dict[str, Any]] = None
    if split == "test" and protocol == LEGACY_PROTOCOL:
        validation_selection = select_validation_hyperparameters(
            experiment_dir=experiment_dir,
            condition=condition,
            seed=seed,
            diffact_fraction=diffact_fraction,
            petri_fraction=petri_fraction,
            method=method,
            protocol=protocol,
        )
        if validation_selection is None:
            raise FileNotFoundError(
                "Test postprocessing requires validation metrics for hyperparameter "
                f"selection with method={method}: "
                f"{experiment_dir / 'petri' / condition / f'seed_{seed}'}"
            )
        selected_hp = validation_selection["selected"]
        if selected_hp.get("method") is not None:
            method = str(selected_hp["method"])
        if selected_hp.get("chunk_size") is not None:
            chunk_size = int(selected_hp["chunk_size"])
        if selected_hp.get("prob_threshold") is not None:
            prob_threshold = float(selected_hp["prob_threshold"])
        if selected_hp.get("transition_illegal_penalty") is not None:
            transition_illegal_penalty = float(selected_hp["transition_illegal_penalty"])
        if selected_hp.get("model_move_cost") is not None:
            model_move_cost = float(selected_hp["model_move_cost"])
        if selected_hp.get("conformance_switch_penalty_weight") is not None:
            conformance_switch_penalty_weight = float(
                selected_hp["conformance_switch_penalty_weight"]
            )

    df, softmax_list, train_seq_ids, eval_seq_ids, map_path = build_combined_inputs(
        data_root=data_root,
        dataset=dataset,
        train_cases=train_cases,
        eval_cases=eval_cases,
        eval_softmax_dir=softmax_dir,
        out_dir=out_dir,
        temperature=learned_temperature,
        petri_discovery_representation=petri_discovery_representation,
    )
    if require_diffact_comparators:
        preflight_required_diffact_comparators(
            combined_map_path=map_path,
            softmax_dir=softmax_dir,
        )
    allowed = allowed_transition_set(df, train_seq_ids)
    hp = DATASET_HP_DEFAULTS[(dataset, "diffact")]
    weights = HP_STRATEGIES[hp["strategy"]]
    is_oracle = petri_discovery_source == PETRI_DISCOVERY_OFFICIAL_TEST
    cfg = {
        "method": method,
        "n_train_traces": None,
        "n_test_traces": None,
        "train_cases": train_seq_ids,
        "test_cases": eval_seq_ids,
        "ensure_train_variant_diversity": False,
        "ensure_test_variant_diversity": False,
        "allow_train_cases_in_test": is_oracle,
        "use_same_traces_for_train_test": False,
        "compute_marking_transition_map": False,
        "sequential_sampling": False,
        "n_indices": 10**9,
        "n_per_run": None,
        "independent_sampling": True,
        "prob_threshold": prob_threshold,
        "chunk_size": chunk_size,
        "conformance_switch_penalty_weight": conformance_switch_penalty_weight,
        "conditioning_alpha": hp["alpha"],
        "conditioning_interpolation_weights": weights,
        "conditioning_combine_fn": linear_prob_combiner,
        "max_hist_len": 3,
        "conditioning_n_prev_labels": 3,
        "conditioning_state_mode": "topm",
        "conditioning_top_m": 1,
        "candidate_top_k": 3,
        "candidate_top_p": 1.0,
        "candidate_min_k": 1,
        "candidate_source": "conditioned",
        "candidate_apply_to_sync": True,
        "restrict_log_moves": True,
        "restrict_model_moves_to_tau": True,
        "max_consecutive_tau_moves": 8,
        "progress_log_interval_chunks": progress_log_interval_chunks,
        "transition_illegal_penalty": transition_illegal_penalty,
        "enabled_cache_size": 100000,
        "cost_function": "linear",
        "model_move_cost": model_move_cost,
        "log_move_cost": 1.0,
        "tau_move_cost": 1e-6,
        "non_sync_penalty": 1.0,
        "use_calibration": False,
        "use_collapsed_runs": True,
        "round_precision": 2,
        "random_seed": seed,
        "save_model": False,
        "save_model_path": None,
        "parallel_processing": False,
        "dataset_parallelization": inner_parallel,
        "dataset_parallelization_context": None,
        "max_workers": workers,
        "merge_mismatched_boundaries": False,
        "case_output_dir": str(out_dir / "case_outputs"),
        "resume_case_outputs": True,
        "verbose": True,
        "log_level": logging.INFO,
    }
    serializable_cfg = {
        k: (str(v) if callable(v) else v)
        for k, v in cfg.items()
        if k != "conditioning_combine_fn"
    }
    serializable_cfg["conditioning_combine_fn"] = "linear_prob_combiner"
    discovery_label = {
        COUPLED_CONDITION: "same_fraction_training_subset",
        DIFFACT_LOW_PETRI_FULL_CONDITION: "full_training_pool_structural_prior",
        DIFFACT_FULL_PETRI_LOW_CONDITION: "low_data_process_discovery_with_full_diffact",
        ORACLE_TEST_FOLD_CONDITION: "official_test_fold_gt_oracle",
    }[condition]
    serializable_cfg.update(
        {
            "dataset": dataset,
            "seed": seed,
            "fraction": diffact_fraction,
            "diffact_fraction": diffact_fraction,
            "petri_fraction": petri_fraction,
            "split": split,
            "condition": condition,
            "protocol": protocol,
            "petri_discovery_source": petri_discovery_source,
            "oracle_upper_bound": is_oracle,
            "petri_discovery_uses_test_gt": is_oracle,
            "is_oracle_condition": is_oracle,
            "petri_discovery_manifest": str(petri_manifest),
            "train_original_cases": train_cases,
            "diffact_train_original_cases": diffact_train_cases,
            "petri_train_original_cases": train_cases,
            "n_diffact_train_cases": len(diffact_train_cases),
            "n_petri_train_cases": len(train_cases),
            "eval_original_cases": eval_cases,
            "n_eval_cases": len(eval_cases),
            "n_full_eval_cases": len(all_eval_cases),
            "combined_case_map": str(map_path),
            "softmax_dir": str(softmax_dir),
            "softmax_source": {
                "directory": str(softmax_dir),
                "diffact_fraction": diffact_fraction,
                "canonical_probability_pattern": "{case_id}.npy",
                "raw_probability_pattern": "{case_id}_raw.npy",
                "official_prediction_pattern": "{case_id}_pred.npy",
            },
            "calibrated": calibrated,
            "learned_temperature": learned_temperature,
            "temp_bounds": list(temp_bounds),
            "calibration_fit_split": "val" if calibrated else None,
            "calibration_uses_test_data": False,
            "validation_selection": validation_selection,
            "validation_selection_enabled": protocol == LEGACY_PROTOCOL,
            "test_set_tuning": False,
            "main_condition_petri_discovery": discovery_label,
            "run_scope": (
                "runtime_only_partial_pilot"
                if runtime_pilot_case_limit is not None
                else "complete_evaluation"
            ),
            "runtime_pilot_case_limit": runtime_pilot_case_limit,
            "complete_evaluation_set": runtime_pilot_case_limit is None,
            "deterministic_eval_case_selection": (
                "first_n_manifest_order" if runtime_pilot_case_limit is not None else None
            ),
            "no_event_level_subsampling": cfg["n_indices"] >= 10**9 and not cfg["sequential_sampling"],
            "petri_discovery_trace_representation": petri_discovery_representation,
            "eval_postprocessing_trace_representation": "frame_events",
            "fixed_decoder_config": {
                "policy": (
                    "canonical_petri_conformance_no_tuning"
                    if protocol == PRIMARY_PROTOCOL
                    else "legacy_validation_selected_pilot"
                ),
                "method": method,
                "chunk_size": chunk_size,
                "prob_threshold": prob_threshold,
                "model_move_cost": model_move_cost,
                "log_move_cost": cfg["log_move_cost"],
                "tau_move_cost": cfg["tau_move_cost"],
                "conformance_switch_penalty_weight": conformance_switch_penalty_weight,
                "conditioning_alpha": hp["alpha"],
                "conditioning_interpolation_weights": weights,
                "conditioning_state_mode": cfg["conditioning_state_mode"],
                "conditioning_top_m": cfg["conditioning_top_m"],
                "candidate_top_k": cfg["candidate_top_k"],
                "restrict_log_moves": cfg["restrict_log_moves"],
                "restrict_model_moves_to_tau": cfg["restrict_model_moves_to_tau"],
                "max_consecutive_tau_moves": cfg["max_consecutive_tau_moves"],
                "petri_discovery_representation": petri_discovery_representation,
                "temperature_calibration": calibrated,
            },
            "paper_diffact_sktr_config": (
                method == PAPER_DIFFACT_SKTR_METHOD
                and chunk_size == PAPER_DIFFACT_SKTR_CHUNK_SIZE
                and workers == PAPER_DIFFACT_SKTR_WORKERS
                and petri_discovery_representation == PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION
                and cfg["restrict_log_moves"]
                and cfg["restrict_model_moves_to_tau"]
                and cfg["max_consecutive_tau_moves"] == 8
                and cfg["conditioning_state_mode"] == "topm"
                and cfg["conditioning_top_m"] == 1
                and cfg["candidate_top_k"] == 3
            ),
        }
    )
    config_path = out_dir / "postprocess_config.json"
    config_path.write_text(
        json.dumps(serializable_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if validation_selection is not None:
        (out_dir / "selected_hyperparameters.json").write_text(
            json.dumps(validation_selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    start = time.perf_counter()
    if method == "petri_conformance":
        recovery_cfg = {
            k: v
            for k, v in cfg.items()
            if k not in {"method", "transition_illegal_penalty"}
        }
        results_df, accuracy_dict, _ = incremental_softmax_recovery(
            df=df,
            softmax_lst=softmax_list,
            **recovery_cfg,
        )
    elif method == "petri_transition_viterbi":
        results_df, accuracy_dict = run_transition_viterbi_recovery(
            df=df,
            softmax_list=softmax_list,
            eval_seq_ids=eval_seq_ids,
            allowed_transitions=allowed,
            prob_threshold=prob_threshold,
            illegal_penalty=transition_illegal_penalty,
            case_output_dir=out_dir / "case_outputs",
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    total_seconds = time.perf_counter() - start
    results_df, comparator_provenance = attach_diffact_comparators(
        results_df=results_df,
        combined_map_path=map_path,
        softmax_dir=softmax_dir,
        require_comparators=require_diffact_comparators,
    )
    run_scope = (
        "runtime_only_partial_pilot"
        if runtime_pilot_case_limit is not None
        else "complete_evaluation"
    )
    results_df["diffact_fraction"] = diffact_fraction
    results_df["petri_fraction"] = petri_fraction
    results_df["condition"] = condition
    results_df["petri_discovery_source"] = petri_discovery_source
    results_df["oracle_upper_bound"] = is_oracle
    results_df["is_oracle_condition"] = is_oracle
    results_df["run_scope"] = run_scope
    results_df.to_csv(results_csv, index=False)
    serializable_cfg["diffact_comparator_provenance"] = comparator_provenance
    config_path.write_text(
        json.dumps(serializable_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "accuracy_dict.json").write_text(
        json.dumps(accuracy_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_df = build_metric_rows(
        dataset=dataset,
        seed=seed,
        diffact_fraction=diffact_fraction,
        petri_fraction=petri_fraction,
        condition=condition,
        split=split,
        post_method=method,
        results_df=results_df,
        mapping_path=data_root / dataset / "mapping.txt",
        allowed_transitions=allowed,
        total_seconds=total_seconds,
        calibrated=calibrated,
    )
    metric_df["run_scope"] = run_scope
    metric_df["complete_evaluation_set"] = runtime_pilot_case_limit is None
    metric_df["runtime_pilot_case_limit"] = runtime_pilot_case_limit
    metric_df["petri_discovery_source"] = petri_discovery_source
    metric_df["oracle_upper_bound"] = is_oracle
    metric_df["is_oracle_condition"] = is_oracle
    metric_df.to_csv(metrics_csv, index=False)
    per_case_df = build_per_case_metric_rows(
        dataset=dataset,
        seed=seed,
        diffact_fraction=diffact_fraction,
        petri_fraction=petri_fraction,
        condition=condition,
        split=split,
        post_method=method,
        results_df=results_df,
        combined_map_path=map_path,
        mapping_path=data_root / dataset / "mapping.txt",
        allowed_transitions=allowed,
        calibrated=calibrated,
    )
    per_case_df["run_scope"] = run_scope
    per_case_df["complete_evaluation_set"] = runtime_pilot_case_limit is None
    per_case_df["runtime_pilot_case_limit"] = runtime_pilot_case_limit
    per_case_df["petri_discovery_source"] = petri_discovery_source
    per_case_df["oracle_upper_bound"] = is_oracle
    per_case_df["is_oracle_condition"] = is_oracle
    per_case_df.to_csv(per_case_metrics_csv, index=False)
    method_name = f"{method}_calibrated" if calibrated else method
    runtime_df, per_case_runtime_df = build_runtime_tables(
        seed=seed,
        diffact_fraction=diffact_fraction,
        petri_fraction=petri_fraction,
        condition=condition,
        split=split,
        method=method_name,
        results_df=results_df,
        combined_map_path=map_path,
        total_seconds=total_seconds,
    )
    for runtime_table in (runtime_df, per_case_runtime_df):
        runtime_table["run_scope"] = run_scope
        runtime_table["complete_evaluation_set"] = runtime_pilot_case_limit is None
        runtime_table["runtime_pilot_case_limit"] = runtime_pilot_case_limit
    runtime_df.to_csv(out_dir / "runtime.csv", index=False)
    per_case_runtime_df.to_csv(per_case_runtime_csv, index=False)
    print(f"Wrote {metrics_csv}")
    return metrics_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=int,
        default=None,
        help="Legacy selector; maps through --condition. Defaults to the coupled 25/50/75/100 curve.",
    )
    parser.add_argument(
        "--diffact-fractions",
        nargs="+",
        type=int,
        default=None,
        help="Explicit DiffAct softmax fractions, zipped with --petri-fractions.",
    )
    parser.add_argument(
        "--petri-fractions",
        nargs="+",
        type=int,
        default=None,
        help="Explicit Petri discovery fractions, zipped with --diffact-fractions.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        default=None,
        help="Defaults to the condition implied by each explicit fraction pair.",
    )
    parser.add_argument("--protocol", choices=PROTOCOL_CHOICES, default="auto")
    parser.add_argument(
        "--method",
        choices=["petri_transition_viterbi", "petri_conformance"],
        default=PAPER_DIFFACT_SKTR_METHOD,
        help=(
            "Postprocessing method. petri_conformance is the exact alignment-based "
            "SKTR path used by the prior full-data DiffAct recovery."
        ),
    )
    parser.add_argument(
        "--transition-illegal-penalty",
        type=float,
        default=2.0,
        help="Soft Viterbi penalty for a non-self transition absent from the training Petri/variant relation.",
    )
    parser.add_argument("--chunk-size", type=int, default=PAPER_DIFFACT_SKTR_CHUNK_SIZE)
    parser.add_argument("--prob-threshold", type=float, default=1e-6)
    parser.add_argument("--model-move-cost", type=float, default=1.0)
    parser.add_argument("--conformance-switch-penalty-weight", type=float, default=1.0)
    parser.add_argument(
        "--progress-log-interval-chunks",
        type=int,
        default=PAPER_DIFFACT_SKTR_PROGRESS_CHUNKS,
        help=(
            "Log conformance progress every N chunks inside each case. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--petri-discovery-representation",
        choices=["run_collapsed_segments", "frame_events"],
        default=PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION,
        help=(
            "Representation used for Petri-net discovery from the training subset. "
            "The prior full-data DiffAct+SKTR run used frame_events. Evaluation "
            "traces are always postprocessed and scored at frame level."
        ),
    )
    parser.add_argument(
        "--petri-discovery-source",
        choices=list(PETRI_DISCOVERY_SOURCES),
        default=None,
        help=(
            "Expected Petri discovery provenance. It must agree with --condition "
            "and normally should be omitted so it is inferred: "
            "nested_train_fraction (honest / D100+P25), full_train "
            "(structural_prior_full_train_petri), or official_test_fold "
            "(oracle_test_fold_petri)."
        ),
    )
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--temp-bounds", nargs=2, type=float, default=[1.0, 10.0])
    parser.add_argument("--workers", type=int, default=PAPER_DIFFACT_SKTR_WORKERS)
    parser.add_argument(
        "--inner-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use SKTR dataset-level parallelization across evaluation cases. "
            "Enabled by default for the low-data experiment; use "
            "--no-inner-parallel for sequential case processing."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--runtime-pilot-case-limit",
        type=int,
        default=None,
        help=(
            "Decode only the first N manifest-ordered cases for runtime estimation. "
            "Outputs are isolated and marked runtime_only_partial_pilot."
        ),
    )
    parser.add_argument(
        "--require-diffact-comparators",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require complete {case_id}_pred.npy and {case_id}_raw.npy streams. "
            "Defaults on for the primary no-validation protocol."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    data_root = args.data_root.resolve()
    metadata = json.loads((experiment_dir / "experiment_metadata.json").read_text())
    dataset = metadata["dataset"]
    protocol = resolve_protocol(metadata, args.protocol)
    fraction_pairs = resolve_fraction_pairs(
        legacy_fractions=args.fractions,
        diffact_fractions=args.diffact_fractions,
        petri_fractions=args.petri_fractions,
        condition=args.condition,
        petri_discovery_source=args.petri_discovery_source,
    )
    require_diffact_comparators = (
        protocol == PRIMARY_PROTOCOL
        if args.require_diffact_comparators is None
        else bool(args.require_diffact_comparators)
    )
    if args.runtime_pilot_case_limit is not None and args.runtime_pilot_case_limit < 1:
        parser.error("--runtime-pilot-case-limit must be positive")
    if protocol == PRIMARY_PROTOCOL:
        validate_primary_decoder_config(
            split=args.split,
            method=args.method,
            chunk_size=args.chunk_size,
            prob_threshold=args.prob_threshold,
            model_move_cost=args.model_move_cost,
            conformance_switch_penalty_weight=args.conformance_switch_penalty_weight,
            petri_discovery_representation=args.petri_discovery_representation,
            calibrated=args.calibrated,
        )
    setup_logging()

    commands: List[str] = []
    completed = 0
    requested_runs = list(iter_requested(args.seeds, fraction_pairs, args.max_runs))
    for seed, diffact_fraction, petri_fraction, condition in requested_runs:
        discovery_source = args.petri_discovery_source or default_petri_discovery_source(
            condition
        )
        parallel_flag = "--inner-parallel" if args.inner_parallel else "--no-inner-parallel"
        command_parts = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--experiment-dir",
            str(experiment_dir),
            "--data-root",
            str(data_root),
            "--seeds",
            str(seed),
            "--diffact-fractions",
            str(diffact_fraction),
        ]
        if petri_fraction is not None:
            command_parts += ["--petri-fractions", str(petri_fraction)]
        command_parts += [
            "--split",
            args.split,
            "--condition",
            condition,
            "--protocol",
            protocol,
            "--method",
            args.method,
            "--transition-illegal-penalty",
            str(args.transition_illegal_penalty),
            "--chunk-size",
            str(args.chunk_size),
            "--prob-threshold",
            str(args.prob_threshold),
            "--model-move-cost",
            str(args.model_move_cost),
            "--conformance-switch-penalty-weight",
            str(args.conformance_switch_penalty_weight),
            "--progress-log-interval-chunks",
            str(args.progress_log_interval_chunks),
            "--petri-discovery-representation",
            args.petri_discovery_representation,
            "--petri-discovery-source",
            discovery_source,
            "--workers",
            str(args.workers),
            parallel_flag,
            (
                "--require-diffact-comparators"
                if require_diffact_comparators
                else "--no-require-diffact-comparators"
            ),
            "--execute",
        ]
        if args.calibrated:
            command_parts += [
                "--calibrated",
                "--temp-bounds",
                str(args.temp_bounds[0]),
                str(args.temp_bounds[1]),
            ]
        if args.runtime_pilot_case_limit is not None:
            command_parts += [
                "--runtime-pilot-case-limit",
                str(args.runtime_pilot_case_limit),
            ]
        if args.force:
            command_parts.append("--force")
        command = shlex.join(command_parts)
        commands.append(command)
        if not args.execute:
            print(
                f"PREPARED seed={seed} diffact_frac={diffact_fraction} "
                f"petri_frac={petri_fraction} condition={condition}: {command}"
            )
            continue
        run_one(
            experiment_dir=experiment_dir,
            data_root=data_root,
            dataset=dataset,
            seed=seed,
            diffact_fraction=diffact_fraction,
            petri_fraction=petri_fraction,
            split=args.split,
            condition=condition,
            protocol=protocol,
            method=args.method,
            chunk_size=args.chunk_size,
            prob_threshold=args.prob_threshold,
            model_move_cost=args.model_move_cost,
            conformance_switch_penalty_weight=args.conformance_switch_penalty_weight,
            progress_log_interval_chunks=args.progress_log_interval_chunks,
            petri_discovery_representation=args.petri_discovery_representation,
            petri_discovery_source=discovery_source,
            transition_illegal_penalty=args.transition_illegal_penalty,
            calibrated=args.calibrated,
            temp_bounds=(float(args.temp_bounds[0]), float(args.temp_bounds[1])),
            workers=args.workers,
            inner_parallel=args.inner_parallel,
            runtime_pilot_case_limit=args.runtime_pilot_case_limit,
            require_diffact_comparators=require_diffact_comparators,
            force=args.force,
        )
        completed += 1

    cal_suffix = "_calibrated" if args.calibrated else ""
    pilot_suffix = (
        f"_runtime_pilot_n{args.runtime_pilot_case_limit}"
        if args.runtime_pilot_case_limit is not None
        else ""
    )
    if len(requested_runs) == 1:
        _, diffact_fraction, petri_fraction, condition = requested_runs[0]
        if petri_fraction is None:
            command_scope = f"{condition}_d{diffact_fraction}_ptest_test"
        else:
            command_scope = f"{condition}_d{diffact_fraction}_p{petri_fraction}"
    elif len({condition for _, _, _, condition in requested_runs}) == 1:
        command_scope = requested_runs[0][3]
    else:
        command_scope = "factorized"
    cmd_path = (
        experiment_dir
        / "petri"
        / f"run_{command_scope}_{args.split}{cal_suffix}{pilot_suffix}_commands.sh"
    )
    cmd_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n")
    cmd_path.chmod(0o755)
    print(f"Completed Petri runs: {completed}")
    print(f"Petri commands: {cmd_path}")
    if not args.execute:
        print("No Petri postprocessing was launched; pass --execute after softmax exports exist.")


if __name__ == "__main__":
    main()
