#!/usr/bin/env python3
"""Shared contracts for the staged GTEA class-specific onset-head pilot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")
DEFAULT_DIFFACT_ROOT = Path("/home/dsi/eli-bogdanov/DiffAct_onset_head")
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/gtea_class_specific_onset_fold1_v1"
)
DEFAULT_LOCKED_REFERENCE_CONFIG = Path(
    "/data1/eli-bogdanov/sktr_runs/epoch_scarcity_gtea_seed0_v1/"
    "experiments/gtea/fold_1/trajectory/seed_0/config.json"
)
DEFAULT_LOCKED_TRAIN_MANIFEST = Path(
    "/data1/eli-bogdanov/sktr_runs/low_data_decoding_study_gtea_seed0_bounds_v2/"
    "experiments/gtea/fold_1/manifests/seed_0/train_cases_frac_100.txt"
)
DEFAULT_LOCKED_TRAIN_BUNDLE = Path(
    "/data1/eli-bogdanov/sktr_runs/low_data_decoding_study_gtea_seed0_bounds_v2/"
    "experiments/gtea/fold_1/diffact_dataset_views/seed_0/frac_100/gtea/"
    "splits/train.split1.bundle"
)
DEFAULT_MULTIFOLD_DECISION = Path(
    "/data1/eli-bogdanov/sktr_runs/gtea_boundary_weight_multifold_v1/"
    "analysis/decision.json"
)

PROTOCOL_VERSION = "gtea-class-specific-onset-head-pilot-v1"
DATASET = "gtea"
FOLD = 1
TRAINING_SEED = 0
PHYSICAL_GPU = 3
DECODER_BOUNDARY_WEIGHT = 1.0
ONSET_LOSS_WEIGHTS = (0.0, 0.1, 0.3, 0.5)
PRIMARY_ONSET_LOSS_WEIGHT = 0.3
BASELINE_ONSET_LOSS_WEIGHT = 0.0
ONSET_SMOOTH = 1
ONSET_BUMP_POSITIVE_WEIGHT = 1.0
CHECKPOINT_EPOCHS = (200, 1000, 2000, 5000, 10000)
INFERENCE_SEEDS = (0, 1, 2)
INFERENCE_SEED_STRIDE = 1_000_000
FINAL_EPOCH = 10000
EXPECTED_NUM_EPOCHS = FINAL_EPOCH + 1
EXPECTED_TRAIN_CASES = 21
EXPECTED_TEST_CASES = 7
BOUNDARY_MATCH_MAX_DISTANCE = 50
SHORT_SEGMENT_MAX_LENGTH = 3

PROVENANCE_FILES = (
    "scripts/gtea_onset_head_pilot/README.md",
    "scripts/gtea_onset_head_pilot/common.py",
    "scripts/gtea_onset_head_pilot/prepare_study.py",
    "scripts/gtea_onset_head_pilot/run_variant.py",
    "scripts/gtea_onset_head_pilot/export_seeded_outputs.py",
    "scripts/gtea_onset_head_pilot/analyze.py",
    "scripts/gtea_onset_head_pilot/study_status.py",
    "scripts/gtea_onset_head_pilot/verify_launch_gate.py",
    "scripts/gtea_onset_head_pilot/tests/conftest.py",
    "scripts/gtea_onset_head_pilot/tests/test_gtea_onset_head_pilot.py",
    "scripts/gtea_boundary_weight_pilot/metrics.py",
    "src/evaluation.py",
)
DIFFACT_PROVENANCE_FILES = (
    "main.py",
    "model.py",
    "dataset.py",
    "utils.py",
    "tests/conftest.py",
    "tests/test_onset_head.py",
)


def normalize_case_id(value: str) -> str:
    text = str(value).strip()
    return text[:-4] if text.endswith(".txt") else text


def read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_lines(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [str(row) for row in rows]
    path.write_text(
        "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
    )


def read_bundle(path: Path) -> List[str]:
    return [normalize_case_id(line) for line in read_lines(path)]


def write_bundle(path: Path, cases: Sequence[str]) -> None:
    write_lines(path, [f"{normalize_case_id(case)}.txt" for case in cases])


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_provenance(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    status = run("status", "--short")
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "repo": str(repo.resolve()),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def current_source_hashes(diffact_root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative in PROVENANCE_FILES:
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = file_sha256(path)
    for relative in DIFFACT_PROVENANCE_FILES:
        path = diffact_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[f"diffact/{relative}"] = file_sha256(path)
    return hashes


def source_provenance(diffact_root: Path) -> Dict[str, Any]:
    hashes = current_source_hashes(diffact_root)
    return {
        "workspace_git": git_provenance(WORKSPACE_ROOT),
        "diffact_git": git_provenance(diffact_root),
        "file_sha256": hashes,
        "source_digest": canonical_digest(hashes),
    }


def verify_source_digest(metadata: Mapping[str, Any], diffact_root: Path) -> None:
    current = current_source_hashes(diffact_root)
    recorded = metadata["source_provenance"]["file_sha256"]
    if current != recorded:
        differing = sorted(
            key
            for key in set(current).union(recorded)
            if current.get(key) != recorded.get(key)
        )
        raise RuntimeError(
            f"Immutable onset-study source changed; generate a new study: {differing}"
        )
    if canonical_digest(current) != metadata["source_provenance"]["source_digest"]:
        raise RuntimeError("Recorded source digest is internally inconsistent")


def ensure_symlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def create_dataset_view(
    study_dir: Path,
    data_root: Path,
    train_cases: Sequence[str],
    test_cases: Sequence[str],
    locked_train_bundle: Path,
) -> Path:
    if read_bundle(locked_train_bundle) != list(train_cases):
        raise ValueError("Locked bundle order differs from requested train order")
    view_root = study_dir / "diffact_dataset_view"
    dataset_root = view_root / DATASET
    dataset_root.mkdir(parents=True, exist_ok=False)
    for name in ("features", "groundTruth"):
        ensure_symlink_or_copy(data_root / DATASET / name, dataset_root / name)
    ensure_symlink_or_copy(
        data_root / DATASET / "mapping.txt", dataset_root / "mapping.txt"
    )
    splits = dataset_root / "splits"
    splits.mkdir()
    shutil.copy2(locked_train_bundle, splits / "train.split1.bundle")
    write_bundle(splits / "test.split1.bundle", test_cases)
    return view_root


def load_mapping(path: Path) -> Tuple[Dict[str, int], List[str]]:
    label_to_index: Dict[str, int] = {}
    index_to_label: Dict[int, str] = {}
    for line in read_lines(path):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed mapping row: {line!r}")
        index = int(fields[0])
        label_to_index[fields[1]] = index
        index_to_label[index] = fields[1]
    labels = [index_to_label[index] for index in sorted(index_to_label)]
    if list(sorted(index_to_label)) != list(range(len(labels))):
        raise ValueError(f"Mapping indices are not contiguous: {path}")
    return label_to_index, labels


def create_alignment_dir(
    study_dir: Path, data_root: Path, test_cases: Sequence[str]
) -> Path:
    align_dir = study_dir / "align" / "test"
    align_dir.mkdir(parents=True, exist_ok=False)
    write_lines(
        align_dir / "video_index_map.txt",
        [f"{index}\t{case}" for index, case in enumerate(test_cases)],
    )
    label_to_index, _ = load_mapping(data_root / DATASET / "mapping.txt")
    rows = ["case:concept:name,concept:name"]
    for index, case in enumerate(test_cases):
        labels = read_lines(data_root / DATASET / "groundTruth" / f"{case}.txt")
        rows.extend(f"{index},{label_to_index[label]}" for label in labels)
    write_lines(align_dir / "ground_truth.csv", rows)
    return align_dir


def weight_slug(weight: float) -> str:
    return str(weight).replace(".", "p")


def variant_id(onset_weight: float) -> str:
    return f"gtea_fold1_trainseed0_bweight1p0_onset{weight_slug(onset_weight)}"


def training_invariant_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "dataset_name",
        "batch_size",
        "boundary_smooth",
        "class_specific_onset_smooth",
        "onset_bump_positive_weight",
        "class_weighting",
        "decoder_params",
        "diffusion_params",
        "encoder_params",
        "learning_rate",
        "log_freq",
        "num_epochs",
        "postprocess",
        "sample_rate",
        "set_sampling_seed",
        "soft_label",
        "temporal_aug",
        "weight_decay",
        "evaluate_during_training",
        "log_train_results",
        "random_seed",
        "initialization_seed",
    )
    payload = {key: deepcopy(config[key]) for key in keys}
    weights = deepcopy(config["loss_weights"])
    weights.pop("class_specific_onset_loss")
    payload["loss_weights_except_class_specific_onset"] = weights
    return payload


def training_invariant_digest(config: Mapping[str, Any]) -> str:
    return canonical_digest(training_invariant_payload(config))


def build_variant_config(
    reference: Mapping[str, Any],
    *,
    study_dir: Path,
    view_root: Path,
    train_manifest: Path,
    test_manifest: Path,
    onset_weight: float,
) -> Dict[str, Any]:
    config = deepcopy(dict(reference))
    task_id = variant_id(onset_weight)
    run_dir = study_dir / "runs" / task_id
    config["naming"] = task_id
    config["root_data_dir"] = str(view_root)
    config["split_id"] = FOLD
    config["result_dir"] = str(run_dir / "training")
    config["random_seed"] = TRAINING_SEED
    config["initialization_seed"] = TRAINING_SEED
    config["evaluate_during_training"] = False
    config["log_train_results"] = False
    config["training_subset_manifest"] = str(train_manifest)
    config["official_train_split_path"] = str(train_manifest)
    config["official_test_split_path"] = str(test_manifest)
    config["pre_specified_final_epoch"] = FINAL_EPOCH
    config["pre_specified_final_checkpoint"] = str(
        run_dir / "training" / task_id / f"epoch-{FINAL_EPOCH}.model"
    )
    config["checkpoint_selection"] = "pre_specified_grid_no_test_selection"
    config["gtea_onset_head_protocol_version"] = PROTOCOL_VERSION
    config["class_specific_onset_smooth"] = ONSET_SMOOTH
    config["onset_bump_positive_weight"] = ONSET_BUMP_POSITIVE_WEIGHT
    config["loss_weights"] = deepcopy(config["loss_weights"])
    config["loss_weights"]["decoder_boundary_loss"] = DECODER_BOUNDARY_WEIGHT
    config["loss_weights"]["class_specific_onset_loss"] = float(onset_weight)
    return config


def assert_variant_config(
    config: Mapping[str, Any], task: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    expectations = {
        "dataset_name": DATASET,
        "split_id": FOLD,
        "boundary_smooth": ONSET_SMOOTH,
        "class_specific_onset_smooth": ONSET_SMOOTH,
        "onset_bump_positive_weight": ONSET_BUMP_POSITIVE_WEIGHT,
        "num_epochs": EXPECTED_NUM_EPOCHS,
        "log_freq": 100,
        "random_seed": TRAINING_SEED,
        "initialization_seed": TRAINING_SEED,
        "evaluate_during_training": False,
        "log_train_results": False,
        "gtea_onset_head_protocol_version": PROTOCOL_VERSION,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expectations.items()
        if config.get(key) != expected
    }
    expected_weight = float(task["class_specific_onset_loss_weight"])
    if float(config["loss_weights"]["class_specific_onset_loss"]) != expected_weight:
        mismatches["loss_weights.class_specific_onset_loss"] = {
            "expected": expected_weight,
            "actual": config["loss_weights"]["class_specific_onset_loss"],
        }
    if float(config["loss_weights"]["decoder_boundary_loss"]) != DECODER_BOUNDARY_WEIGHT:
        mismatches["loss_weights.decoder_boundary_loss"] = {
            "expected": DECODER_BOUNDARY_WEIGHT,
            "actual": config["loss_weights"]["decoder_boundary_loss"],
        }
    if mismatches:
        raise ValueError(f"Onset variant changed a frozen field: {mismatches}")
    if training_invariant_digest(config) != metadata["training_invariant_digest"]:
        raise ValueError("Onset variant training-invariant digest mismatch")


def load_task(study_dir: Path, task_id: str) -> Dict[str, Any]:
    tasks = load_json(study_dir / "tasks.json")["tasks"]
    matches = [task for task in tasks if task["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one task {task_id!r}")
    return matches[0]


def checkpoint_path(task: Mapping[str, Any], epoch: int) -> Path:
    return Path(task["model_dir"]) / f"epoch-{int(epoch)}.model"


def export_dir(task: Mapping[str, Any], epoch: int, inference_seed: int) -> Path:
    return (
        Path(task["run_dir"])
        / "exports"
        / f"epoch_{int(epoch):05d}"
        / f"sampling_seed_{int(inference_seed)}"
    )


def map_rows(path: Path) -> List[Tuple[int, str]]:
    rows = []
    for line in read_lines(path):
        index, case = line.split(maxsplit=1)
        rows.append((int(index), normalize_case_id(case)))
    return sorted(rows)


def export_artifacts(output_dir: Path, expected_cases: int) -> List[Path]:
    paths = [
        output_dir / "video_index_map.txt",
        output_dir / "mapping.txt",
        output_dir / "ground_truth.csv",
    ]
    for index in range(expected_cases):
        paths.extend(
            [
                output_dir / f"{index}_raw.npy",
                output_dir / f"{index}.npy",
                output_dir / f"{index}_pred.npy",
                output_dir / f"{index}_onset.npy",
            ]
        )
    return paths


def verify_export(
    output_dir: Path,
    expected_cases: Sequence[str],
    *,
    verify_recorded_hashes: bool = True,
) -> Dict[str, Any]:
    expected = [normalize_case_id(case) for case in expected_cases]
    if map_rows(output_dir / "video_index_map.txt") != list(enumerate(expected)):
        raise ValueError(f"Export case map mismatch: {output_dir}")
    total_frames = 0
    class_count = None
    for index, case in enumerate(expected):
        raw = np.load(output_dir / f"{index}_raw.npy", mmap_mode="r")
        canonical = np.load(output_dir / f"{index}.npy", mmap_mode="r")
        pred = np.load(output_dir / f"{index}_pred.npy", mmap_mode="r")
        onset = np.load(output_dir / f"{index}_onset.npy", mmap_mode="r")
        if raw.ndim != 2 or canonical.shape != raw.shape or onset.shape != raw.shape:
            raise ValueError(
                f"{case}: invalid activity/onset shapes "
                f"{raw.shape}/{canonical.shape}/{onset.shape}"
            )
        if pred.shape != (raw.shape[1],):
            raise ValueError(f"{case}: invalid prediction shape {pred.shape}")
        if not np.isfinite(onset).all() or onset.min() < 0 or onset.max() > 1:
            raise ValueError(f"{case}: onset curves are outside [0,1]")
        class_count = int(raw.shape[0]) if class_count is None else class_count
        if raw.shape[0] != class_count:
            raise ValueError(f"{case}: class count changed")
        total_frames += int(raw.shape[1])
    artifacts = export_artifacts(output_dir, len(expected))
    for path in artifacts:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    hashes = {
        path.relative_to(output_dir).as_posix(): file_sha256(path)
        for path in artifacts
    }
    if verify_recorded_hashes:
        complete = load_json(output_dir / "export_complete.json")
        if complete.get("artifact_sha256") != hashes:
            raise ValueError(f"Export hashes changed: {output_dir}")
    return {
        "case_count": len(expected),
        "frame_count": total_frames,
        "class_count": class_count,
        "artifact_sha256": hashes,
    }


def highest_saved_epoch(model_dir: Path) -> int | None:
    epochs = []
    for path in model_dir.glob("epoch-*.model"):
        try:
            epochs.append(int(path.stem.split("-")[-1]))
        except ValueError:
            pass
    return max(epochs) if epochs else None
