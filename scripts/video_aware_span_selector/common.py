#!/usr/bin/env python3
"""Shared, provenance-aware helpers for the four-fold selector scale-up."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "data" / "data"))
DEFAULT_DIFFACT_ROOT = Path(
    os.environ.get(
        "DIFFACT_ROOT",
        Path.home() / "sktr_for_long_traces" / "baselines" / "DiffAct",
    )
)
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/"
    "breakfast_video_selector_oof_allfolds_seed0_v2"
)
DEFAULT_V1_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/"
    "breakfast_video_selector_oof_outer1_seed0_v1"
)
DEFAULT_OUTER_EXPORT_ROOT = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_diffact_raw_exports_v1"
)

DATASET = "breakfast"
OUTER_FOLDS = (1, 2, 3, 4)
INNER_FOLDS = 3
SEED = 0
FINAL_EPOCH = 1000
PROTOCOL_VERSION = "breakfast-video-aware-selector-allfolds-v2"
FRAME_BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
BASE_NUMERIC_FEATURES = (
    "uncertainty_mean",
    "uncertainty_q90",
    "entropy_mean",
    "top1_uncertainty_mean",
    "margin_mean",
    "pred_probability_mean",
    "official_override_gap_mean",
    "duration_raw_abs_z",
    "duration_norm_abs_z",
    "segment_log_length",
    "segment_fraction",
    "video_progress_mid",
    "class_frame_rarity",
    "class_segment_rarity",
    "neighbor_class_rarity",
)
SHAPE_FEATURES = (
    "confidence_slope",
    "edge_vs_core_margin",
    "flicker_rate",
    "runner_up_gap",
    "runner_up_consistency",
)
REPAIR_CANDIDATE_TOP_K = 5
REPAIR_CANDIDATE_FIELDS = ("class_id", "label", "mean_probability")
FORBIDDEN_PRIMARY_FEATURE_PREFIXES = ("task__", "camera__")

PROVENANCE_FILES = (
    "scripts/video_aware_span_selector/README.md",
    "scripts/video_aware_span_selector/common.py",
    "scripts/video_aware_span_selector/prepare_study.py",
    "scripts/video_aware_span_selector/run_oof_task.py",
    "scripts/video_aware_span_selector/study_status.py",
    "scripts/video_aware_span_selector/analyze_selector.py",
)
DIFFACT_PROVENANCE_FILES = (
    "main.py",
    "export_softmax.py",
    "dataset.py",
    "utils.py",
    "model.py",
    "configs/Breakfast-Trained-S1.json",
)


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    participant: str
    task: str
    camera: str
    n_frames: int


def normalize_case_id(value: str) -> str:
    text = str(value).strip()
    return text[:-4] if text.endswith(".txt") else text


def repair_candidate_columns() -> Tuple[str, ...]:
    return tuple(
        f"candidate_rank_{rank}_{field}"
        for rank in range(1, REPAIR_CANDIDATE_TOP_K + 1)
        for field in REPAIR_CANDIDATE_FIELDS
    )


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_provenance(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return proc.stdout.strip()

    status = run("status", "--short")
    diff = subprocess.run(
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
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def source_provenance(diffact_root: Path) -> Dict[str, Any]:
    hashes: Dict[str, str] = {}
    for relative in PROVENANCE_FILES:
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required selector source is missing: {path}")
        hashes[relative] = file_sha256(path)
    for relative in DIFFACT_PROVENANCE_FILES:
        path = diffact_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required DiffAct source is missing: {path}")
        hashes[f"diffact/{relative}"] = file_sha256(path)
    return {
        "selector_git": git_provenance(WORKSPACE_ROOT),
        "diffact_git": git_provenance(diffact_root),
        "file_sha256": hashes,
        "source_digest": canonical_digest(hashes),
    }


def current_source_hashes(diffact_root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative in PROVENANCE_FILES:
        hashes[relative] = file_sha256(WORKSPACE_ROOT / relative)
    for relative in DIFFACT_PROVENANCE_FILES:
        hashes[f"diffact/{relative}"] = file_sha256(diffact_root / relative)
    return hashes


def parse_case(case_id: str, n_frames: int) -> CaseInfo:
    case_id = normalize_case_id(case_id)
    pieces = case_id.split("_")
    if not pieces or not re.fullmatch(r"P\d+", pieces[0]):
        raise ValueError(f"Cannot parse Breakfast participant from {case_id!r}")
    participant = pieces[0]
    task = pieces[-1]
    camera_token = pieces[1] if len(pieces) >= 3 else "unknown"
    camera = re.sub(r"\d+$", "", camera_token).lower() or camera_token.lower()
    return CaseInfo(case_id, participant, task, camera, int(n_frames))


def ground_truth_path(data_root: Path, case_id: str) -> Path:
    return data_root / DATASET / "groundTruth" / f"{normalize_case_id(case_id)}.txt"


def load_case_infos(data_root: Path, cases: Sequence[str]) -> List[CaseInfo]:
    infos: List[CaseInfo] = []
    for case in cases:
        n_frames = len(read_lines(ground_truth_path(data_root, case)))
        if n_frames <= 0:
            raise ValueError(f"Empty Breakfast ground truth: {case}")
        infos.append(parse_case(case, n_frames))
    return infos


def official_splits(data_root: Path, outer_fold: int) -> Tuple[List[str], List[str]]:
    split_dir = data_root / DATASET / "splits"
    train = read_bundle(split_dir / f"train.split{outer_fold}.bundle")
    test = read_bundle(split_dir / f"test.split{outer_fold}.bundle")
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError("Official Breakfast split contains duplicate case IDs")
    overlap = set(train).intersection(test)
    if overlap:
        raise ValueError(f"Official train/test overlap: {sorted(overlap)[:5]}")
    return train, test


def make_subject_disjoint_inner_folds(
    infos: Sequence[CaseInfo], n_folds: int = INNER_FOLDS, seed: int = SEED
) -> Dict[int, List[str]]:
    """Greedily balance participant groups while retaining exactly 13 per fold."""
    by_participant: Dict[str, List[CaseInfo]] = defaultdict(list)
    for info in infos:
        by_participant[info.participant].append(info)
    participants = sorted(by_participant)
    if len(participants) % n_folds:
        raise ValueError(
            f"Expected participant count divisible by {n_folds}, got {len(participants)}"
        )
    capacity = len(participants) // n_folds
    all_tasks = sorted({info.task for info in infos})
    target_frames = sum(info.n_frames for info in infos) / n_folds
    target_cases = len(infos) / n_folds
    task_totals = Counter(info.task for info in infos)
    target_task_cases = {task: task_totals[task] / n_folds for task in all_tasks}

    participant_frames = {
        participant: sum(info.n_frames for info in group)
        for participant, group in by_participant.items()
    }
    participant_cases = {participant: len(group) for participant, group in by_participant.items()}
    participant_tasks = {
        participant: Counter(info.task for info in group)
        for participant, group in by_participant.items()
    }

    def objective(assignment: Mapping[int, Sequence[str]]) -> float:
        total = 0.0
        for people in assignment.values():
            frames = sum(participant_frames[person] for person in people)
            cases = sum(participant_cases[person] for person in people)
            task_counts: Counter[str] = Counter()
            for person in people:
                task_counts.update(participant_tasks[person])
            total += 4.0 * ((frames - target_frames) / max(target_frames, 1.0)) ** 2
            total += 2.0 * ((cases - target_cases) / max(target_cases, 1.0)) ** 2
            total += 0.5 * sum(
                ((task_counts[task] - target) / max(target, 1.0)) ** 2
                for task, target in target_task_cases.items()
            )
        return total

    best_assignment: Dict[int, List[str]] | None = None
    best_score = float("inf")
    # Multiple deterministic starts plus exact-capacity pair swaps avoid a
    # brittle split driven by the first few unusually long participants.
    for restart in range(24):
        rng = random.Random(seed * 10_000 + restart)
        order = sorted(
            participants,
            key=lambda participant: (
                -(participant_frames[participant] * (0.95 + 0.10 * rng.random())),
                participant,
            ),
        )
        assignment: Dict[int, List[str]] = {
            fold: [] for fold in range(1, n_folds + 1)
        }
        for participant in order:
            choices: List[Tuple[float, float, int]] = []
            for fold in assignment:
                if len(assignment[fold]) >= capacity:
                    continue
                candidate = {key: list(value) for key, value in assignment.items()}
                candidate[fold].append(participant)
                choices.append((objective(candidate), rng.random(), fold))
            assignment[min(choices)[2]].append(participant)

        for _ in range(100):
            current_score = objective(assignment)
            best_swap: Tuple[float, int, str, int, str] | None = None
            for left_fold in range(1, n_folds + 1):
                for right_fold in range(left_fold + 1, n_folds + 1):
                    for left_person in sorted(assignment[left_fold]):
                        for right_person in sorted(assignment[right_fold]):
                            candidate = {
                                key: list(value) for key, value in assignment.items()
                            }
                            candidate[left_fold].remove(left_person)
                            candidate[right_fold].remove(right_person)
                            candidate[left_fold].append(right_person)
                            candidate[right_fold].append(left_person)
                            score = objective(candidate)
                            swap = (
                                score,
                                left_fold,
                                left_person,
                                right_fold,
                                right_person,
                            )
                            if best_swap is None or swap < best_swap:
                                best_swap = swap
            if best_swap is None or best_swap[0] >= current_score - 1e-12:
                break
            _, left_fold, left_person, right_fold, right_person = best_swap
            assignment[left_fold].remove(left_person)
            assignment[right_fold].remove(right_person)
            assignment[left_fold].append(right_person)
            assignment[right_fold].append(left_person)
        score = objective(assignment)
        canonical = tuple(tuple(sorted(assignment[fold])) for fold in sorted(assignment))
        best_canonical = (
            tuple(tuple(sorted(best_assignment[fold])) for fold in sorted(best_assignment))
            if best_assignment is not None
            else None
        )
        if score < best_score - 1e-12 or (
            abs(score - best_score) <= 1e-12
            and (best_canonical is None or canonical < best_canonical)
        ):
            best_score = score
            best_assignment = {fold: list(people) for fold, people in assignment.items()}
    assert best_assignment is not None
    result = {
        fold: sorted(
            info.case_id
            for participant in best_assignment[fold]
            for info in by_participant[participant]
        )
        for fold in best_assignment
    }
    validate_inner_folds(infos, result, n_folds=n_folds)
    return result


def validate_inner_folds(
    infos: Sequence[CaseInfo], folds: Mapping[int, Sequence[str]], n_folds: int = INNER_FOLDS
) -> None:
    expected = {info.case_id for info in infos}
    seen: set[str] = set()
    participant_seen: set[str] = set()
    participant_by_case = {info.case_id: info.participant for info in infos}
    expected_people = len({info.participant for info in infos}) // n_folds
    if sorted(folds) != list(range(1, n_folds + 1)):
        raise ValueError(f"Inner-fold IDs must be 1..{n_folds}")
    for fold, cases in folds.items():
        case_set = set(cases)
        if len(case_set) != len(cases):
            raise ValueError(f"Inner fold {fold} has duplicate cases")
        overlap = seen.intersection(case_set)
        if overlap:
            raise ValueError(f"Cases appear in multiple inner folds: {sorted(overlap)[:5]}")
        people = {participant_by_case[case] for case in cases}
        if len(people) != expected_people:
            raise ValueError(
                f"Inner fold {fold} has {len(people)} people; expected {expected_people}"
            )
        people_overlap = participant_seen.intersection(people)
        if people_overlap:
            raise ValueError(f"Participants cross inner folds: {sorted(people_overlap)}")
        seen.update(case_set)
        participant_seen.update(people)
    if seen != expected:
        raise ValueError(
            f"Inner-fold case coverage mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}"
        )


def fold_summary(infos: Sequence[CaseInfo], cases: Sequence[str]) -> Dict[str, Any]:
    by_case = {info.case_id: info for info in infos}
    selected = [by_case[case] for case in cases]
    return {
        "case_count": len(selected),
        "frame_count": sum(info.n_frames for info in selected),
        "participant_count": len({info.participant for info in selected}),
        "participants": sorted({info.participant for info in selected}),
        "task_case_counts": dict(sorted(Counter(info.task for info in selected).items())),
        "camera_case_counts": dict(sorted(Counter(info.camera for info in selected).items())),
    }


def ensure_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError(f"Existing symlink {path} points to {path.resolve()}, not {target}")
        return
    if path.exists():
        raise FileExistsError(path)
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def create_dataset_view(
    view_root: Path,
    data_root: Path,
    train_cases: Sequence[str],
    heldout_cases: Sequence[str],
) -> None:
    dataset_source = data_root / DATASET
    dataset_view = view_root / DATASET
    ensure_symlink(dataset_view / "features", dataset_source / "features")
    ensure_symlink(dataset_view / "groundTruth", dataset_source / "groundTruth")
    ensure_symlink(dataset_view / "mapping.txt", dataset_source / "mapping.txt")
    split_dir = dataset_view / "splits"
    write_bundle(split_dir / "train.split1.bundle", train_cases)
    write_bundle(split_dir / "test.split1.bundle", heldout_cases)


def create_alignment_dir(
    align_dir: Path, data_root: Path, cases: Sequence[str]
) -> None:
    """Write the ASFormer-shaped index/GT bundle expected by export_softmax.py."""
    align_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = data_root / DATASET / "mapping.txt"
    label_to_index: Dict[str, int] = {}
    for line in read_lines(mapping_path):
        index, label = line.split(maxsplit=1)
        label_to_index[label] = int(index)
    write_lines(
        align_dir / "video_index_map.txt",
        [f"{index}\t{case}" for index, case in enumerate(cases)],
    )
    gt_rows = ["case:concept:name,concept:name"]
    for index, case in enumerate(cases):
        for label in read_lines(ground_truth_path(data_root, case)):
            if label not in label_to_index:
                raise ValueError(f"{case}: label {label!r} is missing from mapping.txt")
            gt_rows.append(f"{index},{label_to_index[label]}")
    write_lines(align_dir / "ground_truth.csv", gt_rows)


def verify_source_digest(metadata: Mapping[str, Any], diffact_root: Path) -> None:
    expected = metadata["source_provenance"]["file_sha256"]
    current = current_source_hashes(diffact_root)
    if current != expected:
        changed = sorted(
            key for key in set(expected).union(current) if expected.get(key) != current.get(key)
        )
        raise RuntimeError(
            "Study source digest no longer matches the immutable metadata. "
            f"Changed files: {changed}. Generate a new study directory."
        )
