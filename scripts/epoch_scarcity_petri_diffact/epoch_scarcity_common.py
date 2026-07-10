#!/usr/bin/env python3
"""Shared constants and provenance helpers for the epoch-scarcity study."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "data" / "data"))
DEFAULT_DIFFACT_ROOT = WORKSPACE_ROOT / "baselines" / "DiffAct"
PROTOCOL_VERSION = "epoch-scarcity-v1"
STUDY_TYPE = "epoch_scarcity"

PRIMARY_CONDITION = "official_full_train_petri"
ORACLE_CONDITION = "oracle_test_fold_petri"
CONDITIONS = (PRIMARY_CONDITION, ORACLE_CONDITION)
DISCOVERY_FULL_TRAIN = "full_train"
DISCOVERY_OFFICIAL_TEST = "official_test_fold"
DISCOVERY_BY_CONDITION = {
    PRIMARY_CONDITION: DISCOVERY_FULL_TRAIN,
    ORACLE_CONDITION: DISCOVERY_OFFICIAL_TEST,
}

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "gtea": {
        "folds": (1, 2, 3, 4),
        "train_cases": 21,
        "num_epochs": 10001,
        "final_epoch_index": 10000,
        "checkpoint_epochs": (200, 1000, 2000, 5000, 10000),
    },
    "50salads": {
        "folds": (1, 2, 3, 4, 5),
        "train_cases": 40,
        "num_epochs": 5001,
        "final_epoch_index": 5000,
        "checkpoint_epochs": (100, 500, 1000, 2500, 5000),
    },
}

CANONICAL_DECODER = {
    "method": "petri_conformance",
    "chunk_size": 11,
    "conditioning_state_mode": "topm",
    "conditioning_top_m": 1,
    "candidate_top_k": 3,
    "restrict_log_moves": True,
    "restrict_model_moves_to_tau": True,
    "max_consecutive_tau_moves": 8,
    "petri_discovery_representation": "frame_events",
}


@dataclass(frozen=True)
class CheckpointRecord:
    epoch_index: int
    completed_epochs: int
    schedule_fraction_pct: float
    training_item_presentations: int
    trainer_step: int
    optimizer_updates: int


def dataset_spec(dataset: str) -> Dict[str, Any]:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported epoch-scarcity dataset: {dataset!r}") from exc


def checkpoint_records(
    dataset: str,
    *,
    n_train_cases: int,
    batch_size: int,
) -> List[CheckpointRecord]:
    spec = dataset_spec(dataset)
    if n_train_cases != int(spec["train_cases"]):
        raise ValueError(
            f"{dataset} requires {spec['train_cases']} official training cases; "
            f"got {n_train_cases}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    final_epoch = int(spec["final_epoch_index"])
    records: List[CheckpointRecord] = []
    for epoch in spec["checkpoint_epochs"]:
        completed = int(epoch) + 1
        presentations = completed * n_train_cases
        records.append(
            CheckpointRecord(
                epoch_index=int(epoch),
                completed_epochs=completed,
                schedule_fraction_pct=100.0 * int(epoch) / final_epoch,
                training_item_presentations=presentations,
                trainer_step=1 + presentations,
                optimizer_updates=presentations // batch_size,
            )
        )
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid JSON: {path}") from exc


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_lines(path: Path, values: Iterable[str]) -> None:
    rows = [str(value).strip() for value in values if str(value).strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def validate_condition_source(condition: str, discovery_source: str) -> None:
    expected = DISCOVERY_BY_CONDITION.get(condition)
    if expected is None:
        raise ValueError(f"Unknown epoch-scarcity condition: {condition!r}")
    if discovery_source != expected:
        raise ValueError(
            f"{condition} requires petri_discovery_source={expected!r}; "
            f"got {discovery_source!r}"
        )


def checkpoint_dir(experiment_dir: Path, seed: int, epoch: int) -> Path:
    return experiment_dir / "trajectory" / f"seed_{seed}" / "checkpoints" / f"epoch_{epoch:05d}"


def checkpoint_metadata_path(experiment_dir: Path, seed: int, epoch: int) -> Path:
    return checkpoint_dir(experiment_dir, seed, epoch) / "checkpoint_metadata.json"


def checkpoint_model_path(experiment_dir: Path, seed: int, epoch: int) -> Path:
    return checkpoint_dir(experiment_dir, seed, epoch) / f"epoch-{epoch}.model"


def softmax_dir(experiment_dir: Path, seed: int, epoch: int) -> Path:
    return checkpoint_dir(experiment_dir, seed, epoch) / "softmax_test"


def record_to_dict(record: CheckpointRecord) -> Dict[str, Any]:
    return asdict(record)


def require_unique(values: Sequence[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique: {list(values)}")
