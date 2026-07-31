#!/usr/bin/env python3
"""Leakage-safe ensemble modeling, repair rules, and TAS metrics for v3."""

from __future__ import annotations

import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import (
    AGREEMENT_MEMBER_NAMES,
    BOOTSTRAP_SEEDS,
    CLASSIFIER_MAX_ITER,
    FEATURE_DIMENSION,
    LARGE_SPAN_THRESHOLD_INCREMENT,
    MAXIMUM_HIGHER_CONFIDENCE,
    MLP_HIDDEN_UNITS,
    MLP_SEEDS,
    OOF_HARM_FLOOR_PP,
    OUTER_FOLDS,
    PRIMARY_GATES,
    canonical_digest,
    canonical_feature_view,
    feature_orientation,
    file_sha256,
    pool_span_mean_std,
    stable_uint64,
)

from src.evaluation import _edit_score_asformer, _segmental_f1_counts_asformer


METRICS = ("acc", "edit", "f1@10", "f1@25", "f1@50")
N_CLASSES = 48


@dataclass(frozen=True)
class CaseData:
    outer_fold: int
    inner_fold: int | None
    case_id: str
    gt: np.ndarray
    baseline: np.ndarray

    @property
    def key(self) -> tuple[int, int | None, str]:
        return (self.outer_fold, self.inner_fold, self.case_id)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def verify_manifest_entries(
    manifest: Mapping[str, Any], workers: int = 4
) -> Dict[str, Any]:
    clean = dict(manifest)
    expected = clean.pop("manifest_digest")
    if canonical_digest(clean) != expected:
        raise RuntimeError("Input manifest self-digest mismatch")
    entries = [
        *manifest.get("fixed_artifacts", []),
        *manifest.get("ground_truth", []),
        *manifest.get("features", []),
    ]

    def verify(entry: Mapping[str, Any]) -> int:
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"Input size mismatch: {path}")
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Input hash mismatch: {path}")
        if "shape" in entry:
            array = np.load(path, mmap_mode="r")
            orientation, frames = feature_orientation(array.shape)
            if list(array.shape) != list(entry["shape"]):
                raise RuntimeError(f"Feature shape drift: {path}")
            if orientation != entry["orientation"] or frames != int(entry["time_frames"]):
                raise RuntimeError(f"Feature orientation/time drift: {path}")
        return path.stat().st_size

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        sizes = list(executor.map(verify, entries))
    return {
        "manifest_digest": expected,
        "files_verified": len(entries),
        "bytes_verified": int(sum(sizes)),
        "all_hashes_match": True,
    }


def fixed_paths(manifest: Mapping[str, Any]) -> Dict[str, Path]:
    return {str(row["role"]): Path(row["path"]) for row in manifest["fixed_artifacts"]}


def read_mapping(path: Path) -> tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    for line in path.read_text().splitlines():
        index, label = line.split(maxsplit=1)
        id_to_name[int(index)] = label
    if sorted(id_to_name) != list(range(N_CLASSES)):
        raise ValueError("Breakfast mapping must contain IDs 0..47")
    return id_to_name, {name: index for index, name in id_to_name.items()}


def load_ground_truth(
    manifest: Mapping[str, Any], name_to_id: Mapping[str, int]
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for row in manifest["ground_truth"]:
        labels = Path(row["path"]).read_text().splitlines()
        values = np.asarray([name_to_id[label] for label in labels], dtype=np.int16)
        if len(values) != int(row["time_frames"]):
            raise ValueError(f"GT length drift: {row['case_id']}")
        result[str(row["case_id"])] = values
    return result


def validate_segment_table(frame: pd.DataFrame, *, oof_only: bool) -> None:
    required = {
        "segment_id",
        "outer_fold",
        "inner_fold",
        "case_id",
        "segment_index",
        "start",
        "end",
        "length",
        "predicted_label",
        "correct_label",
        "correct_label_fraction",
        "base_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Segment table misses columns: {missing}")
    if frame["segment_id"].duplicated().any():
        raise ValueError("Segment IDs must be unique")
    if set(frame["outer_fold"].astype(int)) != set(OUTER_FOLDS):
        raise ValueError("Segment table does not contain four outer folds")
    if oof_only:
        if set(frame["scope"]) != {"oof_validation"} or set(frame["mode"]) != {"official"}:
            raise ValueError("OOF screening accepts official OOF rows only")
        if frame["inner_fold"].isna().any():
            raise ValueError("OOF rows require inner-fold identity")
    if frame["base_score"].isna().any():
        raise ValueError("Missing frozen base selector score")
    lengths = frame["end"].astype(int) - frame["start"].astype(int)
    if not lengths.equals(frame["length"].astype(int)):
        raise ValueError("Inconsistent span length")


def build_cases(
    segments: pd.DataFrame,
    ground_truth: Mapping[str, np.ndarray],
    *,
    include_inner: bool,
) -> list[CaseData]:
    group_columns = ["outer_fold", "case_id"]
    if include_inner:
        group_columns.insert(1, "inner_fold")
    cases: list[CaseData] = []
    for keys, group in segments.groupby(group_columns, sort=True):
        if include_inner:
            fold, inner, case_id = keys
            inner_value: int | None = int(inner)
        else:
            fold, case_id = keys
            inner_value = None
        ordered = group.sort_values(["segment_index", "start"], kind="mergesort")
        gt = ground_truth[str(case_id)]
        starts = ordered["start"].to_numpy(dtype=int)
        ends = ordered["end"].to_numpy(dtype=int)
        if starts[0] != 0 or ends[-1] != len(gt) or np.any(starts[1:] != ends[:-1]):
            raise ValueError(f"Non-contiguous prediction: outer={fold}/{case_id}")
        pred = np.empty(len(gt), dtype=np.int16)
        for row in ordered.itertuples():
            start, end = int(row.start), int(row.end)
            pred[start:end] = int(row.predicted_label)
            span_gt = gt[start:end]
            counts = np.bincount(span_gt, minlength=N_CLASSES)
            target = int(row.correct_label)
            if counts[target] != counts.max():
                raise ValueError(f"Invalid majority target: segment={row.segment_id}")
            if abs(float(row.correct_label_fraction) - counts[target] / len(span_gt)) > 1e-9:
                raise ValueError(f"Majority fraction drift: segment={row.segment_id}")
        cases.append(CaseData(int(fold), inner_value, str(case_id), gt, pred))
    return cases


def aggregate_metrics(
    cases: Sequence[CaseData], predictions: Mapping[tuple[int, int | None, str], np.ndarray]
) -> Dict[str, float]:
    frames = 0
    correct = 0
    edits: list[float] = []
    counts = {0.10: [0, 0, 0], 0.25: [0, 0, 0], 0.50: [0, 0, 0]}
    for case in cases:
        pred = np.asarray(predictions[case.key], dtype=str)
        gt = np.asarray(case.gt, dtype=str)
        frames += len(gt)
        correct += int(np.sum(pred == gt))
        edits.append(float(_edit_score_asformer(pred.tolist(), gt.tolist(), ["background"])))
        for threshold in counts:
            tp, fp, fn = _segmental_f1_counts_asformer(
                gt.tolist(), pred.tolist(), threshold, None
            )
            counts[threshold][0] += int(tp)
            counts[threshold][1] += int(fp)
            counts[threshold][2] += int(fn)
    result = {"acc": 100.0 * safe_div(correct, frames), "edit": float(np.mean(edits))}
    for threshold, (tp, fp, fn) in counts.items():
        result[f"f1@{int(100 * threshold)}"] = 100.0 * safe_div(
            2 * tp, 2 * tp + fp + fn
        )
    return result


def baseline_metrics(cases: Sequence[CaseData]) -> Dict[str, float]:
    return aggregate_metrics(cases, {case.key: case.baseline for case in cases})


def select_budget_rows(
    segments: pd.DataFrame, total_frames: int, budget: float
) -> pd.DataFrame:
    requested = max(1, int(round(float(budget) * int(total_frames))))
    scoped = segments.copy()
    scoped["tie_key"] = [
        int(
            stable_uint64(
                row.scope,
                row.mode,
                row.inner_fold,
                row.case_id,
                row.segment_index,
            )
        )
        for row in scoped.itertuples()
    ]
    scoped = scoped.sort_values(
        ["base_score", "tie_key"], ascending=[False, True], kind="mergesort"
    )
    remaining = requested
    rows: list[Dict[str, Any]] = []
    cutoffs = 0
    for row in scoped.itertuples():
        if remaining <= 0:
            break
        start, end = int(row.start), int(row.end)
        selected = min(end - start, remaining)
        selected_start = start
        if selected < end - start:
            selected_start = start + (end - start - selected) // 2
            cutoffs += 1
        rows.append(
            {
                "segment_id": int(row.segment_id),
                "outer_fold": int(row.outer_fold),
                # The frozen selector CSV encodes outer-test inner_fold as 0,
                # whereas CaseData correctly uses None. Scope is authoritative;
                # never let this serialization sentinel become a case key.
                "inner_fold": (
                    None
                    if str(row.scope) == "outer_test" or pd.isna(row.inner_fold)
                    else int(row.inner_fold)
                ),
                "scope": str(row.scope),
                "case_id": str(row.case_id),
                "segment_index": int(row.segment_index),
                "segment_start": start,
                "segment_end": end,
                "selected_start": selected_start,
                "selected_end": selected_start + selected,
                "selected_frames": selected,
                "is_partial_budget_cutoff": selected < end - start,
                "base_score": float(row.base_score),
                "tie_key": int(row.tie_key),
            }
        )
        remaining -= selected
    if remaining or cutoffs > 1:
        raise AssertionError(f"Budget fill failed: remaining={remaining}, cutoffs={cutoffs}")
    return pd.DataFrame(rows)


def pool_all_features(
    records: pd.DataFrame, feature_entries: Mapping[str, Mapping[str, Any]]
) -> np.ndarray:
    result = np.empty((len(records), 2 * FEATURE_DIMENSION), dtype=np.float32)
    for case_id, positions in records.groupby("case_id", sort=True).groups.items():
        entry = feature_entries[str(case_id)]
        array = np.load(entry["path"], mmap_mode="r")
        canonical = canonical_feature_view(array, int(entry["time_frames"]))
        cache: Dict[tuple[int, int], np.ndarray] = {}
        for position in positions:
            row = records.loc[position]
            key = (int(row.start), int(row.end))
            if key not in cache:
                cache[key] = pool_span_mean_std(
                    canonical, key[0], key[1], int(entry["time_frames"])
                )
            result[int(position)] = cache[key]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite visual features")
    return result


def _fit_logistic(
    x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(x_train)
    model = LogisticRegression(C=1.0, max_iter=CLASSIFIER_MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(transformed, y_train, sample_weight=weights)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(
            "Logistic repair head did not converge: "
            + "; ".join(str(item.message) for item in convergence)
        )
    return scaler, model


def _aligned_probabilities(
    model: Any, probabilities: np.ndarray, n_classes: int = N_CLASSES
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), n_classes), dtype=np.float64)
    aligned[:, np.asarray(model.classes_, dtype=int)] = probabilities
    return aligned


def fit_plain_probabilities(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    scaler, model = _fit_logistic(x_train, y_train, weights)
    probabilities = _aligned_probabilities(model, model.predict_proba(scaler.transform(x_eval)))
    return probabilities, {
        "scaler": scaler,
        "logistic": model,
        "max_n_iter": int(np.max(model.n_iter_)),
        "converged": True,
    }


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_units: int = MLP_HIDDEN_UNITS,
    random_state: int = 0,
) -> tuple[StandardScaler, MLPClassifier]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(x_train)
    model = MLPClassifier(
        hidden_layer_sizes=(int(hidden_units),),
        solver="adam",
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=int(random_state),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(transformed, y_train)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(
            "MLP repair head did not converge: "
            + "; ".join(str(item.message) for item in convergence)
        )
    return scaler, model


def case_bootstrap_positions(
    records: pd.DataFrame,
    train_positions: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Sample training videos with replacement and retain all of their spans."""

    train = records.iloc[train_positions]
    case_keys = sorted(
        {
            (int(row.outer_fold), int(row.inner_fold), str(row.case_id))
            for row in train.itertuples()
        }
    )
    positions_by_case: Dict[tuple[int, int, str], np.ndarray] = {}
    for key in case_keys:
        fold, inner, case_id = key
        mask = (
            (train["outer_fold"].astype(int).to_numpy() == fold)
            & (train["inner_fold"].astype(int).to_numpy() == inner)
            & (train["case_id"].astype(str).to_numpy() == case_id)
        )
        positions_by_case[key] = train_positions[np.flatnonzero(mask)]
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(case_keys), size=len(case_keys))
    sampled_keys = [case_keys[int(index)] for index in draws]
    sampled_positions = np.concatenate([positions_by_case[key] for key in sampled_keys])
    return sampled_positions.astype(int, copy=False), {
        "seed": int(seed),
        "drawn_cases": len(sampled_keys),
        "unique_cases": len(set(sampled_keys)),
        "sampled_segments": len(sampled_positions),
    }


def fit_agreement_ensemble_probabilities(
    records: pd.DataFrame,
    features: np.ndarray,
    train_positions: np.ndarray,
    eval_positions: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Fit the pre-registered seven independent heads and align probabilities."""

    train = records.iloc[train_positions]
    x_eval = features[eval_positions]
    y_train = train["correct_label"].to_numpy(dtype=int)
    weights = train["length"].to_numpy(dtype=float)
    members: list[np.ndarray] = []
    artifacts: Dict[str, Any] = {"member_order": list(AGREEMENT_MEMBER_NAMES)}

    full, full_artifacts = fit_plain_probabilities(
        features[train_positions], y_train, weights, x_eval
    )
    members.append(full)
    artifacts["full_logistic"] = full_artifacts

    for seed in BOOTSTRAP_SEEDS:
        sampled, bootstrap_audit = case_bootstrap_positions(
            records, train_positions, seed=seed
        )
        sampled_rows = records.iloc[sampled]
        probabilities, model_artifacts = fit_plain_probabilities(
            features[sampled],
            sampled_rows["correct_label"].to_numpy(dtype=int),
            sampled_rows["length"].to_numpy(dtype=float),
            x_eval,
        )
        name = f"bootstrap_logistic_seed_{seed}"
        members.append(probabilities)
        artifacts[name] = {**model_artifacts, "bootstrap": bootstrap_audit}

    for seed in MLP_SEEDS:
        scaler, model = _fit_mlp(
            features[train_positions],
            y_train,
            hidden_units=MLP_HIDDEN_UNITS,
            random_state=seed,
        )
        probabilities = _aligned_probabilities(
            model, model.predict_proba(scaler.transform(x_eval))
        )
        name = f"mlp256_seed_{seed}"
        members.append(probabilities)
        artifacts[name] = {
            "scaler": scaler,
            "mlp": model,
            "mlp_n_iter": int(model.n_iter_),
            "converged": True,
        }

    if len(members) != len(AGREEMENT_MEMBER_NAMES):
        raise AssertionError("Agreement ensemble member count drift")
    stacked = np.stack(members, axis=1)
    if not np.isfinite(stacked).all():
        raise RuntimeError("Non-finite agreement probabilities")
    if not np.allclose(stacked.sum(axis=2), 1.0, atol=1e-8):
        raise RuntimeError("Agreement probability rows do not sum to one")
    return stacked, artifacts


def crossfit_agreement_probabilities(
    records: pd.DataFrame, features: np.ndarray
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    outputs = np.full(
        (len(records), len(AGREEMENT_MEMBER_NAMES), N_CLASSES),
        np.nan,
        dtype=np.float64,
    )
    audit: list[Dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        fold_mask = records["outer_fold"].astype(int).to_numpy() == fold
        for held_inner in (1, 2, 3):
            eval_positions = np.flatnonzero(
                fold_mask & (records["inner_fold"].astype(int).to_numpy() == held_inner)
            )
            train_positions = np.flatnonzero(
                fold_mask & (records["inner_fold"].astype(int).to_numpy() != held_inner)
            )
            train = records.iloc[train_positions]
            evaluation = records.iloc[eval_positions]
            if set(train["participant"]) & set(evaluation["participant"]):
                raise ValueError(f"Participant leakage: outer={fold}, held_inner={held_inner}")
            probabilities, artifacts = fit_agreement_ensemble_probabilities(
                records, features, train_positions, eval_positions
            )
            outputs[eval_positions] = probabilities
            for member in AGREEMENT_MEMBER_NAMES:
                member_artifacts = artifacts[member]
                row: Dict[str, Any] = {
                    "outer_fold": fold,
                    "held_inner_fold": held_inner,
                    "member": member,
                    "train_segments": len(train_positions),
                    "eval_segments": len(eval_positions),
                    "train_cases": train["case_id"].nunique(),
                    "eval_cases": evaluation["case_id"].nunique(),
                    "participant_overlap": 0,
                    "converged": True,
                }
                if "max_n_iter" in member_artifacts:
                    row["n_iter"] = member_artifacts["max_n_iter"]
                if "mlp_n_iter" in member_artifacts:
                    row["n_iter"] = member_artifacts["mlp_n_iter"]
                if "bootstrap" in member_artifacts:
                    row.update(
                        {
                            f"bootstrap_{key}": value
                            for key, value in member_artifacts["bootstrap"].items()
                        }
                    )
                audit.append(row)
    if not np.isfinite(outputs).all():
        raise RuntimeError("Missing/non-finite cross-fitted agreement probabilities")
    if not np.allclose(outputs.sum(axis=2), 1.0, atol=1e-8):
        raise RuntimeError("Cross-fitted agreement probability rows do not sum to one")
    return outputs, audit


def agreement_consensus(
    member_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return consensus label, vote count, mean confidence, and mean matrix."""

    values = np.asarray(member_probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(AGREEMENT_MEMBER_NAMES), N_CLASSES):
        raise ValueError(
            "Agreement probabilities must have shape "
            f"(segments,{len(AGREEMENT_MEMBER_NAMES)},{N_CLASSES})"
        )
    proposals = np.argmax(values, axis=2)
    means = values.mean(axis=1)
    labels = np.empty(len(values), dtype=np.int16)
    votes = np.empty(len(values), dtype=np.int16)
    confidence = np.empty(len(values), dtype=np.float64)
    for position in range(len(values)):
        counts = np.bincount(proposals[position], minlength=N_CLASSES)
        maximum = int(counts.max())
        tied = np.flatnonzero(counts == maximum)
        best_mean = float(means[position, tied].max())
        label = int(tied[np.flatnonzero(means[position, tied] == best_mean)[0]])
        labels[position] = label
        votes[position] = maximum
        confidence[position] = means[position, label]
    return labels, votes, confidence, means


def reference_action_support(
    segments: pd.DataFrame,
    member_probabilities: np.ndarray,
    reference_ledger: pd.DataFrame,
    *,
    agreement_k: int,
    threshold: float,
) -> pd.DataFrame:
    """Measure whether agreement would abstain on each v2 reference span."""

    values = np.asarray(member_probabilities, dtype=np.float64)
    if values.shape != (len(segments), len(AGREEMENT_MEMBER_NAMES), N_CLASSES):
        raise ValueError("Reference-support probability shape drift")
    positions = {
        int(segment_id): position
        for position, segment_id in enumerate(segments["segment_id"].astype(int))
    }
    consensus_labels, consensus_votes, consensus_confidence, means = agreement_consensus(
        values
    )
    rows = []
    accepted = reference_ledger[reference_ledger["accepted"]]
    for reference in accepted.itertuples(index=False):
        position = positions[int(reference.segment_id)]
        label = int(reference.proposal_label)
        proposals = np.argmax(values[position], axis=1)
        reference_votes = int(np.sum(proposals == label))
        reference_mean_probability = float(values[position, :, label].mean())
        reference_supported = (
            reference_votes >= int(agreement_k)
            and reference_mean_probability >= float(threshold)
        )
        consensus_label = int(consensus_labels[position])
        votes = int(consensus_votes[position])
        mean_probability = float(consensus_confidence[position])
        incumbent = int(reference.incumbent_label)
        would_relabel = (
            votes >= int(agreement_k)
            and mean_probability >= float(threshold)
            and consensus_label != incumbent
        )
        rows.append(
            {
                "outer_fold": int(reference.outer_fold),
                "inner_fold": int(reference.inner_fold),
                "video_key_sha256": str(reference.video_key_sha256),
                "segment_id": int(reference.segment_id),
                "reference_label": label,
                "incumbent_label": incumbent,
                "reference_outcome": str(reference.outcome),
                "agreement_k": int(agreement_k),
                "threshold": float(threshold),
                "reference_label_supporting_members": reference_votes,
                "mean_reference_label_probability": reference_mean_probability,
                "reference_action_supported": bool(reference_supported),
                "agreement_consensus_label": consensus_label,
                "agreement_consensus_members": votes,
                "agreement_consensus_probability": mean_probability,
                "agreement_would_relabel": bool(would_relabel),
                "reference_action_vetoed": not bool(would_relabel),
            }
        )
    return pd.DataFrame(rows)


def _video_hash(case: CaseData) -> str:
    return hashlib.sha256(
        f"outer={case.outer_fold}|inner={case.inner_fold}|case={case.case_id}".encode()
    ).hexdigest()


def apply_configuration(
    cases: Sequence[CaseData],
    segments: pd.DataFrame,
    probabilities: np.ndarray,
    selected: pd.DataFrame,
    *,
    threshold: float,
    rule_name: str,
    rule_parameter: float | None,
    agreement_k: int | None = None,
) -> tuple[
    Dict[tuple[int, int | None, str], np.ndarray], pd.DataFrame, pd.DataFrame
]:
    predictions = {case.key: case.baseline.copy() for case in cases}
    case_lookup = {case.key: case for case in cases}
    segment_lookup = segments.set_index("segment_id", drop=False)
    values = np.asarray(probabilities)
    if values.ndim == 2:
        if values.shape != (len(segments), N_CLASSES):
            raise ValueError("Single-model probability shape drift")
        proposal_labels = np.argmax(values, axis=1).astype(np.int16)
        agreement_counts = np.ones(len(values), dtype=np.int16)
        mean_probabilities = values
        required_agreement = 1 if agreement_k is None else int(agreement_k)
        if required_agreement != 1:
            raise ValueError("A single-model configuration requires agreement_k=1")
    elif values.ndim == 3:
        proposal_labels, agreement_counts, _, mean_probabilities = agreement_consensus(values)
        if agreement_k is None:
            raise ValueError("Agreement ensemble configuration requires agreement_k")
        required_agreement = int(agreement_k)
        if not 4 <= required_agreement <= len(AGREEMENT_MEMBER_NAMES):
            raise ValueError(f"Invalid agreement_k={required_agreement}")
    else:
        raise ValueError("Probability tensor must be 2D or 3D")
    position_lookup = {
        int(segment_id): position
        for position, segment_id in enumerate(segments["segment_id"].astype(int))
    }
    candidates: list[Dict[str, Any]] = []
    for chosen in selected.itertuples(index=False):
        segment = segment_lookup.loc[int(chosen.segment_id)]
        position = position_lookup[int(chosen.segment_id)]
        vector = mean_probabilities[position]
        proposal = int(proposal_labels[position])
        agreement_count = int(agreement_counts[position])
        confidence = float(vector[proposal])
        incumbent = int(segment["predicted_label"])
        incumbent_probability = float(vector[incumbent])
        key = (
            int(chosen.outer_fold),
            None if chosen.inner_fold is None or pd.isna(chosen.inner_fold) else int(chosen.inner_fold),
            str(chosen.case_id),
        )
        case = case_lookup[key]
        eligible = (
            agreement_count >= required_agreement
            and confidence >= threshold
            and proposal != incumbent
        )
        reason = "eligible"
        if agreement_count < required_agreement:
            reason = "insufficient_agreement"
        elif confidence < threshold:
            reason = "below_tau"
        elif proposal == incumbent:
            reason = "same_as_incumbent"
        if eligible and rule_name == "incumbent_margin":
            eligible = confidence - incumbent_probability >= float(rule_parameter)
            if not eligible:
                reason = "below_incumbent_margin"
        if eligible and rule_name == "large_span_guard":
            segment_pct = 100.0 * int(segment["length"]) / len(case.gt)
            if segment_pct > float(rule_parameter):
                higher = min(
                    MAXIMUM_HIGHER_CONFIDENCE,
                    threshold + LARGE_SPAN_THRESHOLD_INCREMENT,
                )
                eligible = confidence >= higher
                if not eligible:
                    reason = "large_span_below_higher_tau"
        candidates.append(
            {
                **chosen._asdict(),
                "case_key": key,
                "video_key_sha256": _video_hash(case),
                "video_frames": len(case.gt),
                "incumbent_label": incumbent,
                "proposal_label": proposal,
                "agreement_count": agreement_count,
                "agreement_required": required_agreement,
                "proposal_probability": confidence,
                "incumbent_probability": incumbent_probability,
                "proposal_margin": confidence - incumbent_probability,
                "eligible_before_video_cap": eligible,
                "decision_reason": reason,
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame["repair_start"] = candidate_frame["selected_start"].astype(int)
    candidate_frame["repair_end"] = candidate_frame["selected_end"].astype(int)
    candidate_frame["accepted"] = candidate_frame["eligible_before_video_cap"].astype(bool)

    if rule_name == "video_cap":
        candidate_frame["accepted"] = False
        for key, positions in candidate_frame.groupby("case_key", sort=False).groups.items():
            group = candidate_frame.loc[positions]
            eligible = group[group["eligible_before_video_cap"]].sort_values(
                ["proposal_probability", "tie_key"],
                ascending=[False, True],
                kind="mergesort",
            )
            video_frames = int(group.iloc[0]["video_frames"])
            remaining = max(1, int(round(float(rule_parameter) / 100.0 * video_frames)))
            for index, row in eligible.iterrows():
                if remaining <= 0:
                    break
                length = int(row.selected_end) - int(row.selected_start)
                accepted = min(length, remaining)
                start = int(row.selected_start)
                if accepted < length:
                    start += (length - accepted) // 2
                candidate_frame.at[index, "repair_start"] = start
                candidate_frame.at[index, "repair_end"] = start + accepted
                candidate_frame.at[index, "accepted"] = True
                candidate_frame.at[index, "decision_reason"] = "accepted_under_video_cap"
                remaining -= accepted
            rejected = eligible.index[~candidate_frame.loc[eligible.index, "accepted"]]
            candidate_frame.loc[rejected, "decision_reason"] = "video_cap_exhausted"
    elif rule_name not in {"none", "incumbent_margin", "large_span_guard"}:
        raise ValueError(rule_name)

    ledger_rows: list[Dict[str, Any]] = []
    for row in candidate_frame.itertuples(index=False):
        case = case_lookup[row.case_key]
        start, end = int(row.repair_start), int(row.repair_end)
        before = case.baseline[start:end]
        gt = case.gt[start:end]
        if bool(row.accepted):
            predictions[case.key][start:end] = int(row.proposal_label)
            after = predictions[case.key][start:end]
        else:
            after = before
        fixed = int(np.sum((before != gt) & (after == gt)))
        broken = int(np.sum((before == gt) & (after != gt)))
        lateral = int(np.sum((before != gt) & (after != gt) & (before != after)))
        net = fixed - broken
        ledger_rows.append(
            {
                **row._asdict(),
                "relabelled_frames": end - start if bool(row.accepted) else 0,
                "fixed_frames": fixed,
                "broken_frames": broken,
                "lateral_changed_wrong_frames": lateral,
                "net_correct_frames": net,
                "outcome": (
                    "abstained"
                    if not bool(row.accepted)
                    else "helped"
                    if net > 0
                    else "hurt"
                    if net < 0
                    else "lateral"
                ),
            }
        )
    ledger = pd.DataFrame(ledger_rows)
    videos = video_metrics(cases, predictions)
    return predictions, ledger, videos


def video_metrics(
    cases: Sequence[CaseData],
    predictions: Mapping[tuple[int, int | None, str], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for case in cases:
        base_correct = int(np.sum(case.baseline == case.gt))
        repaired_correct = int(np.sum(predictions[case.key] == case.gt))
        rows.append(
            {
                "outer_fold": case.outer_fold,
                "inner_fold": case.inner_fold,
                "video_key_sha256": _video_hash(case),
                "n_frames": len(case.gt),
                "baseline_correct_frames": base_correct,
                "repair_correct_frames": repaired_correct,
                "net_correct_frames": repaired_correct - base_correct,
                "baseline_acc": 100.0 * base_correct / len(case.gt),
                "repair_acc": 100.0 * repaired_correct / len(case.gt),
                "delta_acc": 100.0 * (repaired_correct - base_correct) / len(case.gt),
            }
        )
    return pd.DataFrame(rows)


def evaluate_configuration(
    cases: Sequence[CaseData],
    segments: pd.DataFrame,
    probabilities: np.ndarray,
    selected: pd.DataFrame,
    *,
    threshold: float,
    rule_name: str,
    rule_parameter: float | None,
    agreement_k: int | None = None,
) -> tuple[Dict[str, Any], Dict[tuple[int, int | None, str], np.ndarray], pd.DataFrame, pd.DataFrame]:
    baseline = baseline_metrics(cases)
    predictions, ledger, videos = apply_configuration(
        cases,
        segments,
        probabilities,
        selected,
        threshold=threshold,
        rule_name=rule_name,
        rule_parameter=rule_parameter,
        agreement_k=agreement_k,
    )
    repaired = aggregate_metrics(cases, predictions)
    result: Dict[str, Any] = {
        "threshold": threshold,
        "agreement_k": 1 if agreement_k is None else int(agreement_k),
        "rule_name": rule_name,
        "rule_parameter": rule_parameter,
        "n_cases": len(cases),
        "n_frames": int(sum(len(case.gt) for case in cases)),
        "selected_frames": int(selected["selected_frames"].sum()),
        "accepted_spans": int(ledger["accepted"].sum()),
        "relabelled_frames": int(ledger["relabelled_frames"].sum()),
        "worst_video_delta_acc": float(videos["delta_acc"].min()),
        "videos_below_minus_5pp": int((videos["delta_acc"] < OOF_HARM_FLOOR_PP).sum()),
        "harm_constraint_pass": bool((videos["delta_acc"] >= OOF_HARM_FLOOR_PP).all()),
    }
    for metric in METRICS:
        result[f"baseline_{metric}"] = baseline[metric]
        result[f"repair_{metric}"] = repaired[metric]
        result[f"delta_{metric}"] = repaired[metric] - baseline[metric]
    return result, predictions, ledger, videos


def selection_sort_key(row: Mapping[str, Any], rule_order: Mapping[str, int]) -> tuple[Any, ...]:
    return (
        float(row["delta_acc"]),
        float(row["delta_f1@25"]),
        float(row["delta_edit"]),
        float(row["worst_video_delta_acc"]),
        -int(rule_order.get(str(row.get("rule_name")), 999)),
        -float(row.get("rule_parameter") or 0.0),
    )


def primary_gate_decision(
    pooled_delta: Mapping[str, float],
    fold_acc_deltas: Sequence[float],
    videos: pd.DataFrame,
) -> Dict[str, Any]:
    total_net = int(videos["net_correct_frames"].sum())
    largest = int(videos["net_correct_frames"].clip(lower=0).max())
    fraction = float(largest / total_net) if total_net > 0 else float("inf")
    checks = {
        "pooled_acc_gain": pooled_delta["acc"] >= PRIMARY_GATES["minimum_pooled_acc_gain_pp"],
        "positive_acc_folds": int(np.sum(np.asarray(fold_acc_deltas) > 0))
        >= PRIMARY_GATES["minimum_positive_acc_folds"],
        "pooled_edit_nonnegative": pooled_delta["edit"]
        >= PRIMARY_GATES["minimum_pooled_edit_gain_pp"],
        "pooled_f1_at_25_nonnegative": pooled_delta["f1@25"]
        >= PRIMARY_GATES["minimum_pooled_f1_at_25_gain_pp"],
        "no_video_acc_drop_over_limit": float(videos["delta_acc"].min())
        >= -PRIMARY_GATES["maximum_video_acc_drop_pp"],
        "single_video_contribution_bounded": fraction
        <= PRIMARY_GATES["maximum_single_video_gain_fraction"],
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "worst_video_delta_acc": float(videos["delta_acc"].min()),
        "positive_acc_folds": int(np.sum(np.asarray(fold_acc_deltas) > 0)),
        "pooled_net_correct_frames": total_net,
        "largest_single_video_gain_fraction": fraction,
    }
