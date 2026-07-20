#!/usr/bin/env python3
"""Shared contracts for the immutable GTEA boundary-weight pilot."""

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
DEFAULT_DIFFACT_ROOT = Path(
    "/home/dsi/eli-bogdanov/sktr_for_long_traces/baselines/DiffAct"
)
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/gtea_boundary_weight_fold1_seed0_v1"
)
DEFAULT_LOCKED_REFERENCE_CONFIG = Path(
    "/data1/eli-bogdanov/sktr_runs/epoch_scarcity_gtea_seed0_v1/"
    "experiments/gtea/fold_1/trajectory/seed_0/config.json"
)
DEFAULT_LOCKED_REFERENCE_EXPORT = Path(
    "/data1/eli-bogdanov/sktr_runs/epoch_scarcity_gtea_seed0_v1/"
    "experiments/gtea/fold_1/trajectory/seed_0/checkpoints/"
    "epoch_10000/softmax_test"
)
DEFAULT_LOCKED_REFERENCE_TRAIN_MANIFEST = Path(
    "/data1/eli-bogdanov/sktr_runs/low_data_decoding_study_gtea_seed0_bounds_v2/"
    "experiments/gtea/fold_1/manifests/seed_0/train_cases_frac_100.txt"
)
DEFAULT_LOCKED_REFERENCE_TRAIN_BUNDLE = Path(
    "/data1/eli-bogdanov/sktr_runs/low_data_decoding_study_gtea_seed0_bounds_v2/"
    "experiments/gtea/fold_1/diffact_dataset_views/seed_0/frac_100/gtea/"
    "splits/train.split1.bundle"
)

PROTOCOL_VERSION = "gtea-boundary-weight-pilot-v1"
DATASET = "gtea"
FOLD = 1
TRAINING_SEED = 0
PHYSICAL_GPU = 3
BOUNDARY_WEIGHTS = (0.1, 0.3, 0.5, 1.0)
PRIMARY_BOUNDARY_WEIGHT = 0.5
CHECKPOINT_EPOCHS = (200, 1000, 2000, 5000, 10000)
INFERENCE_SEEDS = (0, 1, 2)
INFERENCE_SEED_STRIDE = 1_000_000
FINAL_EPOCH = 10000
EXPECTED_NUM_EPOCHS = FINAL_EPOCH + 1
EXPECTED_TRAIN_CASES = 21
EXPECTED_TEST_CASES = 7
RECONCILIATION_METRIC_TOLERANCE = 0.5
RECONCILIATION_FRAME_DISAGREEMENT_TOLERANCE = 0.01
BOUNDARY_MATCH_MAX_DISTANCE = 50
SHORT_SEGMENT_MAX_LENGTH = 3


PROVENANCE_FILES = (
    "scripts/gtea_boundary_weight_pilot/README.md",
    "scripts/gtea_boundary_weight_pilot/common.py",
    "scripts/gtea_boundary_weight_pilot/metrics.py",
    "scripts/gtea_boundary_weight_pilot/prepare_study.py",
    "scripts/gtea_boundary_weight_pilot/export_seeded_softmax.py",
    "scripts/gtea_boundary_weight_pilot/run_variant.py",
    "scripts/gtea_boundary_weight_pilot/reconcile_baseline.py",
    "scripts/gtea_boundary_weight_pilot/study_status.py",
    "scripts/gtea_boundary_weight_pilot/analyze.py",
    "src/evaluation.py",
)
DIFFACT_PROVENANCE_FILES = (
    "main.py",
    "model.py",
    "dataset.py",
    "utils.py",
    "export_softmax.py",
    "configs/GTEA-Trained-S1.json",
)


def normalize_case_id(value: str) -> str:
    text = str(value).strip()
    return text[:-4] if text.endswith(".txt") else text


def read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [str(row) for row in rows]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def read_bundle(path: Path) -> List[str]:
    return [normalize_case_id(line) for line in read_lines(path)]


def write_bundle(path: Path, cases: Sequence[str]) -> None:
    write_lines(path, [f"{normalize_case_id(case)}.txt" for case in cases])


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
            raise FileNotFoundError(f"Missing pilot source: {path}")
        hashes[relative] = file_sha256(path)
    for relative in DIFFACT_PROVENANCE_FILES:
        path = diffact_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing DiffAct source: {path}")
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
            key for key in set(current).union(recorded) if current.get(key) != recorded.get(key)
        )
        raise RuntimeError(
            "Immutable pilot source digest mismatch. Generate a new study directory; "
            f"changed files: {differing}"
        )
    digest = canonical_digest(current)
    if digest != metadata["source_provenance"]["source_digest"]:
        raise RuntimeError("Immutable pilot source digest is internally inconsistent")


def ensure_symlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace generated data view path: {destination}")
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
    *,
    locked_train_bundle: Path | None = None,
) -> Path:
    view_root = study_dir / "diffact_dataset_view"
    dataset_root = view_root / DATASET
    dataset_root.mkdir(parents=True, exist_ok=False)
    for name in ("features", "groundTruth"):
        ensure_symlink_or_copy(data_root / DATASET / name, dataset_root / name)
    ensure_symlink_or_copy(data_root / DATASET / "mapping.txt", dataset_root / "mapping.txt")
    splits = dataset_root / "splits"
    splits.mkdir()
    train_bundle = splits / "train.split1.bundle"
    if locked_train_bundle is None:
        write_bundle(train_bundle, train_cases)
    else:
        locked_cases = read_bundle(locked_train_bundle)
        if locked_cases != [normalize_case_id(case) for case in train_cases]:
            raise ValueError(
                "Locked DiffAct train bundle order differs from requested training order"
            )
        shutil.copy2(locked_train_bundle, train_bundle)
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
        missing = sorted(set(labels) - set(label_to_index))
        if missing:
            raise ValueError(f"{case}: labels missing from mapping: {missing}")
        rows.extend(f"{index},{label_to_index[label]}" for label in labels)
    write_lines(align_dir / "ground_truth.csv", rows)
    return align_dir


def boundary_weight_slug(weight: float) -> str:
    return str(weight).replace(".", "p")


def variant_id(weight: float, training_seed: int = TRAINING_SEED) -> str:
    return f"gtea_fold{FOLD}_trainseed{training_seed}_boundary{boundary_weight_slug(weight)}"


def training_invariant_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Fields that must remain identical across the four training variants."""
    keys = (
        "dataset_name",
        "batch_size",
        "boundary_smooth",
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
        "random_seed",
        "evaluate_during_training",
        "log_train_results",
    )
    payload = {key: deepcopy(config[key]) for key in keys}
    weights = deepcopy(config["loss_weights"])
    weights.pop("decoder_boundary_loss")
    payload["loss_weights_except_decoder_boundary"] = weights
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
    weight: float,
    training_seed: int = TRAINING_SEED,
) -> Dict[str, Any]:
    config = deepcopy(dict(reference))
    task_id = variant_id(weight, training_seed)
    run_dir = study_dir / "runs" / task_id
    config["naming"] = task_id
    config["root_data_dir"] = str(view_root)
    config["split_id"] = 1
    config["result_dir"] = str(run_dir / "training")
    config["random_seed"] = int(training_seed)
    config["initialization_seed"] = int(training_seed)
    config["evaluate_during_training"] = False
    config["log_train_results"] = False
    config["training_subset_manifest"] = str(train_manifest)
    config["official_train_split_path"] = str(train_manifest)
    config["official_test_split_path"] = str(test_manifest)
    config["pre_specified_final_epoch"] = FINAL_EPOCH
    config["pre_specified_final_checkpoint"] = str(
        run_dir / "training" / task_id / f"epoch-{FINAL_EPOCH}.model"
    )
    config["checkpoint_selection"] = "pre_specified_final_epoch_no_test_selection"
    config["gtea_boundary_weight_protocol_version"] = PROTOCOL_VERSION
    config["gtea_boundary_weight_variant"] = float(weight)
    config["loss_weights"] = deepcopy(config["loss_weights"])
    config["loss_weights"]["decoder_boundary_loss"] = float(weight)
    return config


def assert_variant_config(
    config: Mapping[str, Any], task: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    expected_weight = float(task["decoder_boundary_loss"])
    actual_weight = float(config["loss_weights"]["decoder_boundary_loss"])
    if actual_weight != expected_weight:
        raise ValueError(
            f"Runtime boundary weight mismatch: task={expected_weight}, config={actual_weight}"
        )
    fixed_expectations = {
        "dataset_name": DATASET,
        "boundary_smooth": 1,
        "soft_label": 1.4,
        "num_epochs": EXPECTED_NUM_EPOCHS,
        "log_freq": 100,
        "random_seed": int(task["training_seed"]),
        "evaluate_during_training": False,
        "log_train_results": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in fixed_expectations.items()
        if config.get(key) != expected
    }
    if config.get("postprocess") != {"type": "purge", "value": 3}:
        mismatches["postprocess"] = {
            "expected": {"type": "purge", "value": 3},
            "actual": config.get("postprocess"),
        }
    expected_conditions = ["full", "zero", "boundary03-", "segment=1", "segment=1"]
    actual_conditions = config.get("diffusion_params", {}).get("cond_types")
    if actual_conditions != expected_conditions:
        mismatches["diffusion_params.cond_types"] = {
            "expected": expected_conditions,
            "actual": actual_conditions,
        }
    if float(config["loss_weights"].get("encoder_boundary_loss", -1)) != 0.0:
        mismatches["loss_weights.encoder_boundary_loss"] = {
            "expected": 0.0,
            "actual": config["loss_weights"].get("encoder_boundary_loss"),
        }
    if mismatches:
        raise ValueError(f"Variant changed a frozen training field: {mismatches}")
    digest = training_invariant_digest(config)
    if digest != metadata["training_invariant_digest"]:
        raise ValueError(
            "Variant training-invariant digest mismatch; more than decoder_boundary_loss changed"
        )


def load_task(study_dir: Path, task_id: str) -> Dict[str, Any]:
    tasks = load_json(study_dir / "tasks.json")["tasks"]
    matches = [task for task in tasks if task["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one task {task_id!r}; found {len(matches)}")
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
    rows: List[Tuple[int, str]] = []
    for line in read_lines(path):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed video map row in {path}: {line!r}")
        rows.append((int(fields[0]), normalize_case_id(fields[1])))
    rows.sort()
    return rows


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
            ]
        )
    return paths


def verify_export(
    output_dir: Path,
    expected_cases: Sequence[str],
    *,
    verify_recorded_hashes: bool = True,
) -> Dict[str, Any]:
    rows = map_rows(output_dir / "video_index_map.txt")
    expected = [normalize_case_id(case) for case in expected_cases]
    if rows != list(enumerate(expected)):
        raise ValueError(
            f"Export case map does not match immutable test manifest: {output_dir}"
        )
    total_frames = 0
    class_count: int | None = None
    for index, case in rows:
        raw = np.load(output_dir / f"{index}_raw.npy", mmap_mode="r")
        canonical = np.load(output_dir / f"{index}.npy", mmap_mode="r")
        pred = np.load(output_dir / f"{index}_pred.npy", mmap_mode="r")
        if raw.ndim != 2 or canonical.shape != raw.shape:
            raise ValueError(f"{case}: invalid probability shapes {raw.shape}/{canonical.shape}")
        if pred.shape != (raw.shape[1],):
            raise ValueError(f"{case}: prediction shape {pred.shape}, expected {(raw.shape[1],)}")
        if class_count is None:
            class_count = int(raw.shape[0])
        if raw.shape[0] != class_count:
            raise ValueError(f"{case}: class-count mismatch")
        total_frames += int(raw.shape[1])
    artifacts = export_artifacts(output_dir, len(expected))
    for path in artifacts:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    hashes = {path.relative_to(output_dir).as_posix(): file_sha256(path) for path in artifacts}
    complete_path = output_dir / "export_complete.json"
    if verify_recorded_hashes:
        complete = load_json(complete_path)
        if complete.get("artifact_sha256") != hashes:
            raise ValueError(f"Export artifact hashes changed after completion: {output_dir}")
    return {
        "case_count": len(expected),
        "frame_count": total_frames,
        "class_count": class_count,
        "artifact_sha256": hashes,
    }


def highest_saved_epoch(model_dir: Path) -> int | None:
    epochs: List[int] = []
    for path in model_dir.glob("epoch-*.model"):
        try:
            epochs.append(int(path.stem.split("-")[-1]))
        except ValueError:
            continue
    return max(epochs) if epochs else None
