#!/usr/bin/env python3
"""
Ceiling and GT-fitness diagnostics for DiffAct + SKTR runs.

This script is intentionally a thin wrapper around the production evaluation
path.  It reconstructs the train/test split for an existing run, rediscovers
the Petri net with the same repository function, and computes an oracle by
calling ``incremental_softmax_recovery`` with one-hot ground-truth softmaxes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_diffact_sktr_fold1_paper import (  # noqa: E402
    DATASET_HP_DEFAULTS,
    HP_STRATEGIES,
    build_base_config_for_fold,
    get_variant_info_fast,
    load_diffact_softmax_and_aligned_df,
    resolve_diffact_softmax_dir,
    select_train_test_cases,
    verify_softmax_list,
)
from src.cv_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_video_to_case_mapping,
    get_dataset_cv_config,
    load_fold_case_ids,
)
from src.data_processing import split_train_test  # noqa: E402
from src.incremental_softmax_recovery import incremental_softmax_recovery  # noqa: E402
from src.petri_model import collapse_runs, discover_petri_net, build_probability_dict  # noqa: E402
from src.utils import linear_prob_combiner, make_cost_function  # noqa: E402


@dataclass(frozen=True)
class RunPreset:
    dataset: str
    run_dir: Path
    workers: int
    unique_only: bool = False
    train_k: Optional[int] = None


RUN_PRESETS: Dict[str, RunPreset] = {
    "50salads": RunPreset(
        dataset="50salads",
        run_dir=Path("/data1/eli-bogdanov/sktr_runs/diffact_50salads_allfolds_resumable_6ba8868_chunk11"),
        workers=10,
    ),
    "gtea": RunPreset(
        dataset="gtea",
        run_dir=Path("/data1/eli-bogdanov/sktr_runs/diffact_gtea_allfolds_resumable_6ba8868_chunk11_w7"),
        workers=7,
    ),
    "breakfast": RunPreset(
        dataset="breakfast",
        run_dir=Path("/data1/eli-bogdanov/sktr_runs/diffact_breakfast_unique199_f14fd99_chunk11_w10"),
        workers=10,
        unique_only=True,
        train_k=199,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether a discovered Petri net can help SKTR beat DiffAct argmax."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(RUN_PRESETS),
        required=True,
        help="Datasets to analyze.",
    )
    parser.add_argument("--fold", type=int, default=1, help="Single fold to analyze unless --all-folds is set.")
    parser.add_argument("--all-folds", action="store_true", help="Analyze all CV folds for each dataset.")
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional explicit test case IDs to analyze after reconstructing the fold split.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional prefix limit on reconstructed test cases. Useful for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/data1/eli-bogdanov/sktr_runs/sktr_ceiling_analysis",
    )
    parser.add_argument(
        "--oracle-workers",
        type=int,
        default=1,
        help=(
            "Workers for oracle recovery. Values >1 enable dataset-level "
            "parallelization; this changes only scheduling, not algorithmic flags."
        ),
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help=(
            "Compute only train/test reconstruction, net parity, existing-case "
            "consistency, and collapsed-sequence fitness. This is useful for a "
            "cheap first pass; ceiling conclusions require oracle enabled."
        ),
    )
    parser.add_argument(
        "--fitness-chunking",
        choices=["full", "run"],
        default="full",
        help=(
            "How to chunk collapsed GT/argmax/SKTR sequences for fitness. "
            "'full' uses one exact whole-sequence conformance problem; 'run' "
            "uses the original recovery chunk size for bounded full-scale scans."
        ),
    )
    parser.add_argument(
        "--oracle-epsilon",
        type=float,
        default=0.0,
        help="Probability assigned to non-GT classes in oracle softmax. 0 gives exact one-hot.",
    )
    parser.add_argument(
        "--progress-log-interval-chunks",
        type=int,
        default=0,
        help="Progress interval for oracle recovery. Default keeps smoke output compact.",
    )
    parser.add_argument(
        "--keep-oracle-records",
        action="store_true",
        help="Write per-frame oracle recovery CSVs in addition to per-case summary.",
    )
    return parser.parse_args()


def case_ids_for_fold(dataset: str, fold: int, softmax_dir: Path, data_root: str) -> Tuple[List[str], List[str]]:
    video_map = build_video_to_case_mapping(
        dataset,
        "diffact",
        video_index_map_path=softmax_dir / "video_index_map.txt",
    )
    split = load_fold_case_ids(dataset, fold, video_map, data_root=data_root)
    return [str(c) for c in split["train"]], [str(c) for c in split["test"]]


def load_run_command(preset: RunPreset) -> str:
    def normalize_command(text: str) -> str:
        text = text.replace("\\\n", " ")
        text = re.sub(r"\\\s+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        for marker in (
            " status=$?",
            "; status=$?",
            " } >>",
            "; echo EXIT_STATUS",
            " EXIT_STATUS=",
            " FINISHED_AT=",
        ):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()
        text = re.sub(r"\s+2>&1.*$", "", text).strip()
        text = re.sub(r"\s+>>\s+.*$", "", text).strip()
        return text.replace('"$OUT"', str(preset.run_dir)).replace("'$OUT'", str(preset.run_dir)).replace(
            "$OUT", str(preset.run_dir)
        )

    def extract_python_command(line: str) -> Optional[str]:
        normalized = normalize_command(line)
        candidates = [
            "python -u eval_diffact_sktr_fold1_paper.py",
            "python3 -u eval_diffact_sktr_fold1_paper.py",
            "python eval_diffact_sktr_fold1_paper.py",
            "python3 eval_diffact_sktr_fold1_paper.py",
            "eval_diffact_sktr_fold1_paper.py",
        ]
        positions = [normalized.find(marker) for marker in candidates if normalized.find(marker) != -1]
        if not positions:
            return None
        return normalize_command(normalized[min(positions) :])

    command_path = preset.run_dir / "run_command.txt"
    if command_path.is_file():
        command = extract_python_command(command_path.read_text())
        if command:
            return command

    watch_log = preset.run_dir / "watch.log"
    if watch_log.is_file():
        candidates: List[str] = []
        for line in watch_log.read_text(errors="ignore").splitlines():
            command = extract_python_command(line)
            if command and "--datasets" in command and "--out-dir" in command:
                candidates.append(command)
        if candidates:
            def score(command: str) -> Tuple[int, int]:
                return (
                    0 if "$" in command else 1,
                    -len(command),
                )

            return max(candidates, key=score)
    return ""


def build_eval_arg_parser() -> argparse.ArgumentParser:
    # Source of truth is the inline parser in eval_diffact_sktr_fold1_paper.main().
    # It cannot be imported directly without refactoring that entry point, so this
    # copy is guarded by unknown_run_args/config_matches_run in every summary.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--chunk-size", type=int, default=11)
    parser.add_argument("--prob-threshold", type=float, default=1e-6)
    parser.add_argument("--model-move-cost", type=float, default=1.0)
    parser.add_argument("--state-mode", type=str, default="topm", choices=["exact", "topm"])
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--candidate-top-k", type=int, default=3)
    parser.add_argument("--candidate-top-p", type=float, default=1.0)
    parser.add_argument("--candidate-min-k", type=int, default=1)
    parser.add_argument("--conformance-switch-penalty-weight", type=float, default=1.0)
    parser.add_argument("--restrict-log-moves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restrict-model-moves-to-tau", action="store_true")
    parser.add_argument("--max-consecutive-tau-moves", type=int, default=8)
    parser.add_argument("--progress-log-interval-chunks", type=int, default=0)
    parser.add_argument("--enabled-cache-size", type=int, default=100000)
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--inner-parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sktr-log-level", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume-case-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--case-output-dir", type=str, default=None)
    parser.add_argument("--unique-only", action="store_true")
    parser.add_argument("--train-k", type=int, default=None)
    parser.add_argument("--train-sample-seed", type=int, default=None)
    parser.add_argument("--case-manifest-dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["50salads", "gtea"])
    parser.add_argument("--all-folds", action="store_true")
    return parser


def parse_eval_run_args(command: str) -> Tuple[argparse.Namespace, List[str]]:
    if not command:
        raise ValueError("Could not find eval_diffact_sktr_fold1_paper.py command in run artifacts")
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("Run command parsed to no tokens")
    if tokens[0].startswith("python"):
        tokens = tokens[1:]
        if tokens and tokens[0] == "-u":
            tokens = tokens[1:]
    if tokens and tokens[0].endswith("eval_diffact_sktr_fold1_paper.py"):
        tokens = tokens[1:]
    parser = build_eval_arg_parser()
    parsed, unknown = parser.parse_known_args(tokens)
    return parsed, unknown


def _path_equal(left: Optional[str], right: Path) -> bool:
    if left is None:
        return False
    left_path = Path(os.path.expandvars(left)).expanduser().resolve()
    return left_path == right.expanduser().resolve()


def validate_run_args(
    *,
    dataset: str,
    fold: int,
    preset: RunPreset,
    run_args: argparse.Namespace,
    unknown: Sequence[str],
) -> Tuple[bool, List[str]]:
    mismatches: List[str] = []
    if unknown:
        mismatches.append(f"unknown parsed args: {list(unknown)}")
    if dataset not in [str(d) for d in run_args.datasets]:
        mismatches.append(f"dataset {dataset!r} not present in run --datasets={run_args.datasets!r}")
    if not run_args.all_folds and int(run_args.fold) != fold:
        mismatches.append(f"fold {fold} not covered by run --fold={run_args.fold}")
    if run_args.out_dir is not None and not _path_equal(run_args.out_dir, preset.run_dir):
        mismatches.append(f"--out-dir {run_args.out_dir!r} != preset run dir {preset.run_dir}")
    expected_case_root = preset.run_dir / "case_outputs"
    if run_args.case_output_dir is not None and not _path_equal(run_args.case_output_dir, expected_case_root):
        mismatches.append(
            f"--case-output-dir {run_args.case_output_dir!r} != expected {expected_case_root}"
        )
    if bool(run_args.unique_only) != bool(preset.unique_only):
        mismatches.append(f"--unique-only {run_args.unique_only} != preset {preset.unique_only}")
    if run_args.train_k != preset.train_k:
        mismatches.append(f"--train-k {run_args.train_k} != preset {preset.train_k}")
    if int(run_args.workers) != int(preset.workers):
        mismatches.append(f"--workers {run_args.workers} != preset {preset.workers}")
    return not mismatches, mismatches


def run_args_to_dict(run_args: argparse.Namespace) -> Dict[str, Any]:
    keys = [
        "datasets",
        "all_folds",
        "fold",
        "seed",
        "data_root",
        "chunk_size",
        "prob_threshold",
        "model_move_cost",
        "state_mode",
        "top_m",
        "candidate_top_k",
        "candidate_top_p",
        "candidate_min_k",
        "conformance_switch_penalty_weight",
        "restrict_log_moves",
        "restrict_model_moves_to_tau",
        "max_consecutive_tau_moves",
        "progress_log_interval_chunks",
        "enabled_cache_size",
        "use_calibration",
        "inner_parallel",
        "workers",
        "unique_only",
        "train_k",
        "train_sample_seed",
        "out_dir",
        "case_output_dir",
    ]
    return {key: getattr(run_args, key) for key in keys}


def parse_run_model_size(run_log: Path, dataset: str, fold: int) -> Tuple[Optional[int], Optional[int]]:
    if not run_log.is_file():
        return None, None
    current_fold: Optional[int] = None
    fold_re = re.compile(rf"--- {re.escape(dataset)} fold (\d+)/\d+ ---")
    size_re = re.compile(r"Discovered Petri net model: (\d+) places, (\d+) transitions")
    for line in run_log.read_text(errors="ignore").splitlines():
        fold_match = fold_re.search(line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            continue
        size_match = size_re.search(line)
        if size_match and current_fold == fold:
            return int(size_match.group(1)), int(size_match.group(2))
    return None, None


def make_base_cfg(
    *,
    run_args: argparse.Namespace,
    workers: int,
    dataset_parallelization: bool,
    progress_log_interval_chunks: int,
) -> Dict[str, Any]:
    max_consecutive_tau_moves = (
        None
        if int(run_args.max_consecutive_tau_moves) == 0
        else int(run_args.max_consecutive_tau_moves)
    )
    return build_base_config_for_fold(
        seed=int(run_args.seed),
        chunk_size=int(run_args.chunk_size),
        prob_threshold=float(run_args.prob_threshold),
        model_move_cost=float(run_args.model_move_cost),
        state_mode=str(run_args.state_mode),
        top_m=int(run_args.top_m),
        candidate_top_k=int(run_args.candidate_top_k),
        candidate_top_p=float(run_args.candidate_top_p),
        candidate_min_k=int(run_args.candidate_min_k),
        conformance_switch_penalty_weight=float(run_args.conformance_switch_penalty_weight),
        restrict_log_moves=bool(run_args.restrict_log_moves),
        restrict_model_moves_to_tau=bool(run_args.restrict_model_moves_to_tau),
        max_consecutive_tau_moves=max_consecutive_tau_moves,
        progress_log_interval_chunks=progress_log_interval_chunks,
        enabled_cache_size=int(run_args.enabled_cache_size),
        use_calibration=bool(run_args.use_calibration),
        workers=workers,
        dataset_parallelization=dataset_parallelization,
    )


def select_cases_for_run(
    *,
    dataset: str,
    fold: int,
    run_args: argparse.Namespace,
    df: pd.DataFrame,
    softmax_dir: Path,
    data_root: str,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    full_train, full_test = case_ids_for_fold(dataset, fold, softmax_dir, data_root)
    variant_df = None
    if run_args.unique_only or run_args.train_k is not None:
        variant_df = get_variant_info_fast(df, use_collapsed=True)
    train_selection_seed = (
        int(run_args.seed)
        if run_args.train_sample_seed is None
        else int(run_args.train_sample_seed)
    )
    train_cases, test_cases, meta = select_train_test_cases(
        train_cases=full_train,
        test_cases=full_test,
        variant_df=variant_df,
        unique_only=bool(run_args.unique_only),
        train_k=run_args.train_k,
        seed=train_selection_seed,
        fold=fold,
    )
    return train_cases, test_cases, meta


def maybe_restrict_test_cases(
    test_cases: List[str],
    case_ids: Optional[Sequence[str]],
    case_limit: Optional[int],
) -> List[str]:
    selected = list(test_cases)
    if case_ids is not None:
        wanted = [str(c) for c in case_ids]
        missing = [c for c in wanted if c not in set(selected)]
        if missing:
            raise ValueError(f"Requested case IDs are not in reconstructed test split: {missing}")
        selected = wanted
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("--case-limit must be positive")
        selected = selected[:case_limit]
    return selected


def extract_train_test_df(
    df: pd.DataFrame,
    train_cases: List[str],
    test_cases: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return split_train_test(
        df,
        n_train_traces=len(train_cases),
        n_test_traces=len(test_cases),
        train_cases=train_cases,
        test_cases=test_cases,
        random_seed=42,
    )


def one_hot_from_df(df: pd.DataFrame, n_classes: int, epsilon: float) -> List[np.ndarray]:
    if epsilon < 0:
        raise ValueError("--oracle-epsilon must be non-negative")
    if n_classes <= 1 and epsilon:
        raise ValueError("epsilon oracle requires at least two classes")
    if epsilon * max(n_classes - 1, 0) >= 1.0:
        raise ValueError("epsilon is too large for the number of classes")

    matrices: List[np.ndarray] = []
    low = float(epsilon)
    high = 1.0 - low * (n_classes - 1)
    for _, group in df.groupby("case:concept:name", sort=False):
        labels = group["concept:name"].astype(int).tolist()
        mat = np.full((n_classes, len(labels)), low, dtype=np.float32)
        for t, label in enumerate(labels):
            mat[int(label), t] = high
        matrices.append(mat)
    return matrices


def load_existing_case_outputs(case_dir: Path, test_cases: Iterable[str]) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    for case in test_cases:
        path = case_dir / f"{case}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed case output: {path}")
        df = pd.read_csv(path)
        required = {"case:concept:name", "sktr_activity", "argmax_activity", "ground_truth", "is_correct"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        outputs[str(case)] = df
    return outputs


def accuracy(pred: Sequence[Any], gt: Sequence[Any]) -> float:
    if len(pred) != len(gt):
        raise ValueError(f"Length mismatch: pred={len(pred)} gt={len(gt)}")
    if not gt:
        return 0.0
    p = np.asarray([str(x) for x in pred])
    g = np.asarray([str(x) for x in gt])
    return float((p == g).mean())


def collapsed_from_frame_df(frame_df: pd.DataFrame, column: str) -> List[str]:
    return collapse_runs(frame_df[column].astype(str).tolist())


def matrix_for_sequence(sequence: Sequence[str], n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, len(sequence)), dtype=np.float32)
    for t, label in enumerate(sequence):
        mat[int(label), t] = 1.0
    return mat


def tau_reaches_final_marking(
    model: Any,
    marking: Any,
    *,
    max_tau_depth: int = 1000,
    max_states: int = 100000,
) -> Tuple[bool, Optional[int], bool]:
    if model.final_mark is None:
        return True, 0, False

    target = model.final_mark.places
    start = marking.places
    if start == target:
        return True, 0, False

    seen = {start}
    queue = deque([(start, 0)])
    truncated = False
    while queue:
        places, depth = queue.popleft()
        if depth >= max_tau_depth:
            truncated = True
            continue
        for transition in model._find_directly_enabled_transitions(places):
            if transition.label is not None:
                continue
            next_places = model._fire_transition(places, transition).places
            if next_places == target:
                return True, depth + 1, False
            if next_places not in seen:
                seen.add(next_places)
                if len(seen) >= max_states:
                    return False, None, True
                queue.append((next_places, depth + 1))
    return False, None, truncated


def align_collapsed_sequence(
    *,
    model: Any,
    sequence: Sequence[str],
    n_classes: int,
    cost_fn: Any,
    prob_dict_uncollapsed: Dict,
    prob_dict_collapsed: Dict,
    cfg: Dict[str, Any],
    chunk_size: Optional[int],
) -> Dict[str, Any]:
    if not sequence:
        return {
            "sync_moves": 0,
            "log_moves": 0,
            "model_moves": 0,
            "tau_moves": 0,
            "alignment_cost": 0.0,
            "fitness": 1.0,
            "accepted_exact": True,
            "final_marking_is_final": True,
            "accepted_exact_final": True,
            "tau_reaches_final": True,
            "tau_completion_depth": 0,
            "tau_search_truncated": False,
            "accepted_exact_tau_completed": True,
        }

    result = model.conformance_chunked(
        softmax_matrix=matrix_for_sequence(sequence, n_classes),
        initial_marking=model.init_mark,
        cost_fn=cost_fn,
        chunk_size=max(1, int(chunk_size) if chunk_size is not None else len(sequence)),
        eps=cfg["prob_threshold"],
        inline_progress=False,
        prob_dict_uncollapsed=prob_dict_uncollapsed,
        prob_dict_collapsed=prob_dict_collapsed,
        switch_penalty_weight=cfg["conformance_switch_penalty_weight"],
        use_state_caching=True,
        merge_mismatched_boundaries=False,
        conditioning_alpha=cfg["conditioning_alpha"],
        conditioning_combine_fn=cfg["conditioning_combine_fn"],
        conditioning_n_prev_labels=cfg["conditioning_n_prev_labels"],
        conditioning_interpolation_weights=cfg["conditioning_interpolation_weights"],
        conditioning_state_mode=cfg["conditioning_state_mode"],
        conditioning_top_m=cfg["conditioning_top_m"],
        candidate_top_p=cfg["candidate_top_p"],
        candidate_top_k=cfg["candidate_top_k"],
        candidate_min_k=cfg["candidate_min_k"],
        candidate_source=cfg["candidate_source"],
        candidate_apply_to_sync=cfg["candidate_apply_to_sync"],
        restrict_log_moves=cfg["restrict_log_moves"],
        restrict_model_moves_to_tau=cfg["restrict_model_moves_to_tau"],
        max_consecutive_tau_moves=cfg["max_consecutive_tau_moves"],
        progress_log_interval_chunks=0,
    )
    alignment = result["alignment"]
    sync_moves = sum(1 for move_type, _, _ in alignment if move_type == "sync")
    log_moves = sum(1 for move_type, _, _ in alignment if move_type == "log")
    model_moves = sum(1 for move_type, _, _ in alignment if move_type == "model")
    tau_moves = sum(1 for move_type, _, _ in alignment if move_type == "tau")
    denom = sync_moves + log_moves + model_moves
    fitness = 1.0 if denom == 0 else sync_moves / denom
    final_marking_is_final = (
        model.final_mark is None
        or result["final_marking"].places == model.final_mark.places
    )
    accepted_exact = bool(log_moves == 0 and model_moves == 0)
    tau_reaches_final, tau_completion_depth, tau_search_truncated = tau_reaches_final_marking(
        model, result["final_marking"]
    )
    return {
        "sync_moves": sync_moves,
        "log_moves": log_moves,
        "model_moves": model_moves,
        "tau_moves": tau_moves,
        "alignment_cost": float(result["total_cost"]),
        "fitness": float(fitness),
        "accepted_exact": accepted_exact,
        "final_marking_is_final": bool(final_marking_is_final),
        "accepted_exact_final": bool(accepted_exact and final_marking_is_final),
        "tau_reaches_final": bool(tau_reaches_final),
        "tau_completion_depth": tau_completion_depth,
        "tau_search_truncated": bool(tau_search_truncated),
        "accepted_exact_tau_completed": bool(accepted_exact and tau_reaches_final),
    }


def prefix_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def run_oracle_recovery(
    *,
    dataset: str,
    fold: int,
    df: pd.DataFrame,
    one_hot_softmax: List[np.ndarray],
    train_cases: List[str],
    test_cases: List[str],
    run_args: argparse.Namespace,
    out_dir: Path,
    oracle_workers: int,
    progress_log_interval_chunks: int,
) -> pd.DataFrame:
    hp = DATASET_HP_DEFAULTS[(dataset, "diffact")]
    cfg = make_base_cfg(
        run_args=run_args,
        workers=oracle_workers,
        dataset_parallelization=oracle_workers > 1,
        progress_log_interval_chunks=progress_log_interval_chunks,
    )
    cfg.update(
        {
            "train_cases": train_cases,
            "test_cases": test_cases,
            "conditioning_alpha": hp["alpha"],
            "conditioning_interpolation_weights": HP_STRATEGIES[hp["strategy"]],
            "verbose": True,
            "log_level": logging.INFO,
            "case_output_dir": out_dir / "oracle_case_outputs" / f"{dataset}_fold{fold}",
            "resume_case_outputs": True,
        }
    )
    if oracle_workers < 1:
        raise ValueError("--oracle-workers must be positive")
    results_df, _, _ = incremental_softmax_recovery(df=df, softmax_lst=one_hot_softmax, **cfg)
    return results_df


def analyze_fold(
    *,
    dataset: str,
    fold: int,
    preset: RunPreset,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    run_command = load_run_command(preset)
    run_args, unknown_run_args = parse_eval_run_args(run_command)
    config_matches_run, config_mismatches = validate_run_args(
        dataset=dataset,
        fold=fold,
        preset=preset,
        run_args=run_args,
        unknown=unknown_run_args,
    )
    if not config_matches_run:
        raise ValueError(
            f"{dataset} fold {fold}: recovered run command does not match preset: "
            + "; ".join(config_mismatches)
        )

    diffact_root = REPO_ROOT / "baselines" / "DiffAct"
    softmax_dir = resolve_diffact_softmax_dir(
        diffact_root, dataset, fold, disallow_legacy=True
    )
    df, softmax_lst, _ = load_diffact_softmax_and_aligned_df(
        dataset, softmax_dir, Path(run_args.data_root)
    )
    verify_softmax_list(softmax_lst, dataset)

    full_train_cases, full_test_cases, selection_meta = select_cases_for_run(
        dataset=dataset,
        fold=fold,
        run_args=run_args,
        df=df,
        softmax_dir=softmax_dir,
        data_root=run_args.data_root,
    )
    selected_test_cases = maybe_restrict_test_cases(
        full_test_cases, args.case_ids, args.case_limit
    )
    train_df, test_df = extract_train_test_df(df, full_train_cases, selected_test_cases)

    hp = DATASET_HP_DEFAULTS[(dataset, "diffact")]
    fitness_cfg = make_base_cfg(
        run_args=run_args,
        workers=1,
        dataset_parallelization=False,
        progress_log_interval_chunks=0,
    )
    fitness_cfg.update(
        {
            "conditioning_alpha": hp["alpha"],
            "conditioning_interpolation_weights": HP_STRATEGIES[hp["strategy"]],
            "conditioning_combine_fn": linear_prob_combiner,
        }
    )
    fitness_chunk_size = (
        int(fitness_cfg["chunk_size"])
        if args.fitness_chunking == "run"
        else None
    )

    model = discover_petri_net(train_df)
    model.enable_caching(True, max_size=fitness_cfg["enabled_cache_size"])
    model._allow_lazy_map_build = False
    cost_fn = make_cost_function(
        base=fitness_cfg["cost_function"],
        model_move=fitness_cfg["model_move_cost"],
        log_move=fitness_cfg["log_move_cost"],
        tau_move=fitness_cfg["tau_move_cost"],
        round_precision=fitness_cfg["round_precision"],
    )
    prob_dict_uncollapsed = build_probability_dict(
        train_df, max_hist_len=fitness_cfg["max_hist_len"], use_collapsed=False
    )
    prob_dict_collapsed = build_probability_dict(
        train_df, max_hist_len=fitness_cfg["max_hist_len"], use_collapsed=True
    )

    n_classes = int(softmax_lst[0].shape[0])
    oracle_by_case: Dict[str, pd.DataFrame] = {}
    if not args.skip_oracle:
        one_hot_softmax = one_hot_from_df(df, n_classes, args.oracle_epsilon)
        oracle_df = run_oracle_recovery(
            dataset=dataset,
            fold=fold,
            df=df,
            one_hot_softmax=one_hot_softmax,
            train_cases=full_train_cases,
            test_cases=selected_test_cases,
            run_args=run_args,
            out_dir=out_dir,
            oracle_workers=args.oracle_workers,
            progress_log_interval_chunks=args.progress_log_interval_chunks,
        )
        if args.keep_oracle_records:
            oracle_df.to_csv(out_dir / f"{dataset}_fold{fold}_oracle_records.csv", index=False)

        oracle_by_case = {
            str(case): g.reset_index(drop=True)
            for case, g in oracle_df.groupby("case:concept:name", sort=False)
        }
    case_output_dir = preset.run_dir / "case_outputs" / f"{dataset}_fold{fold}"
    existing_outputs = load_existing_case_outputs(case_output_dir, selected_test_cases)

    rows: List[Dict[str, Any]] = []
    for case_id in selected_test_cases:
        case_id = str(case_id)
        existing = existing_outputs[case_id].reset_index(drop=True)
        gt = existing["ground_truth"].astype(str).tolist()
        argmax = existing["argmax_activity"].astype(str).tolist()
        sktr = existing["sktr_activity"].astype(str).tolist()
        oracle = oracle_by_case.get(case_id)
        oracle_pred = oracle["sktr_activity"].astype(str).tolist() if oracle is not None else []

        gt_collapsed = collapse_runs(gt)
        argmax_collapsed = collapse_runs(argmax)
        sktr_collapsed = collapse_runs(sktr)

        row: Dict[str, Any] = {
            "dataset": dataset,
            "fold": fold,
            "case_id": case_id,
            "n_frames": len(gt),
            "n_gt_segments": len(gt_collapsed),
            "n_argmax_segments": len(argmax_collapsed),
            "n_sktr_segments": len(sktr_collapsed),
            "argmax_acc": accuracy(argmax, gt),
            "sktr_acc": accuracy(sktr, gt),
            "sktr_minus_argmax_acc": accuracy(sktr, gt) - accuracy(argmax, gt),
            "oracle_acc": accuracy(oracle_pred, gt) if oracle is not None else np.nan,
            "oracle_minus_argmax_acc": (
                accuracy(oracle_pred, gt) - accuracy(argmax, gt)
                if oracle is not None
                else np.nan
            ),
            "oracle_argmax_is_gt": (
                bool(oracle["argmax_activity"].astype(str).tolist() == gt)
                if oracle is not None
                else None
            ),
            "case_output_final_sktr_cumacc": float(existing["cumulative_accuracy"].iloc[-1]),
            "case_output_is_correct_mean": float(existing["is_correct"].astype(bool).mean()),
        }
        row["case_output_consistent"] = bool(
            abs(row["sktr_acc"] - row["case_output_final_sktr_cumacc"]) < 1e-12
            and abs(row["sktr_acc"] - row["case_output_is_correct_mean"]) < 1e-12
        )

        for prefix, seq in (
            ("gt", gt_collapsed),
            ("argmax", argmax_collapsed),
            ("sktr", sktr_collapsed),
        ):
            metrics = align_collapsed_sequence(
                model=model,
                sequence=seq,
                n_classes=n_classes,
                cost_fn=cost_fn,
                prob_dict_uncollapsed=prob_dict_uncollapsed,
                prob_dict_collapsed=prob_dict_collapsed,
                cfg=fitness_cfg,
                chunk_size=fitness_chunk_size,
            )
            row.update(prefix_metrics(prefix, metrics))

        rows.append(row)

    case_df = pd.DataFrame(rows)
    run_places, run_transitions = parse_run_model_size(preset.run_dir / "run.log", dataset, fold)
    gt_prefix_exact_mask = case_df["gt_accepted_exact"].astype(bool)
    gt_final_exact_mask = case_df["gt_accepted_exact_final"].astype(bool)
    gt_tau_completed_exact_mask = case_df["gt_accepted_exact_tau_completed"].astype(bool)
    gt_tau_search_truncated_mask = case_df["gt_tau_search_truncated"].astype(bool)
    prefix_exact_oracle_min_acc = (
        float(case_df.loc[gt_prefix_exact_mask, "oracle_acc"].min())
        if (not args.skip_oracle and bool(gt_prefix_exact_mask.any()))
        else None
    )
    final_exact_oracle_min_acc = (
        float(case_df.loc[gt_final_exact_mask, "oracle_acc"].min())
        if (not args.skip_oracle and bool(gt_final_exact_mask.any()))
        else None
    )
    tau_completed_exact_oracle_min_acc = (
        float(case_df.loc[gt_tau_completed_exact_mask, "oracle_acc"].min())
        if (not args.skip_oracle and bool(gt_tau_completed_exact_mask.any()))
        else None
    )
    oracle_floor_exercised = bool(not args.skip_oracle and gt_tau_completed_exact_mask.any())
    final_exact_oracle_floor_ok = (
        bool(final_exact_oracle_min_acc >= 1.0 - 1e-12)
        if (not args.skip_oracle and bool(gt_final_exact_mask.any()) and final_exact_oracle_min_acc is not None)
        else None
    )
    tau_completed_exact_oracle_floor_ok = (
        bool(tau_completed_exact_oracle_min_acc >= 1.0 - 1e-12)
        if oracle_floor_exercised and tau_completed_exact_oracle_min_acc is not None
        else None
    )
    model_size_matches = (
        run_places == len(model.places) and run_transitions == len(model.transitions)
        if run_places is not None and run_transitions is not None
        else None
    )
    oracle_argmax_is_gt_all = (
        bool(case_df["oracle_argmax_is_gt"].all())
        if not args.skip_oracle
        else None
    )
    case_output_consistent_all = bool(case_df["case_output_consistent"].all())
    gate_ready = bool(
        config_matches_run
        and model_size_matches
        and case_output_consistent_all
        and not args.skip_oracle
        and oracle_argmax_is_gt_all
        and oracle_floor_exercised
        and tau_completed_exact_oracle_floor_ok
    )
    summary = {
        "dataset": dataset,
        "fold": fold,
        "run_dir": str(preset.run_dir),
        "softmax_dir": str(softmax_dir),
        "run_command": run_command,
        "parsed_run_args": run_args_to_dict(run_args),
        "unknown_run_args": list(unknown_run_args),
        "config_matches_run": bool(config_matches_run),
        "config_mismatches": config_mismatches,
        "gate_ready": gate_ready,
        "gate_exact_fit_basis": "tau_completed_final_marking",
        "oracle_floor_exercised": oracle_floor_exercised,
        "final_marking_enforced_by_conformance": False,
        "final_marking_semantics": (
            "diagnostic_only; partial_trace_conformance stops at timestamp==n_ts "
            "and does not require model.final_mark"
        ),
        "fitness_chunking": args.fitness_chunking,
        "fitness_chunk_size": fitness_chunk_size,
        "train_cases": len(full_train_cases),
        "test_cases_full": len(full_test_cases),
        "test_cases_analyzed": len(selected_test_cases),
        "unique_only": bool(run_args.unique_only),
        "train_k": run_args.train_k,
        "data_root": run_args.data_root,
        "selection_meta": selection_meta,
        "model_places": len(model.places),
        "model_transitions": len(model.transitions),
        "run_model_places": run_places,
        "run_model_transitions": run_transitions,
        "model_size_matches_run_log": model_size_matches,
        "argmax_acc_frame": float(np.average(case_df["argmax_acc"], weights=case_df["n_frames"])),
        "sktr_acc_frame": float(np.average(case_df["sktr_acc"], weights=case_df["n_frames"])),
        "oracle_acc_frame": (
            float(np.average(case_df["oracle_acc"], weights=case_df["n_frames"]))
            if not args.skip_oracle
            else None
        ),
        "gt_prefix_exact_fit_cases": int(gt_prefix_exact_mask.sum()),
        "gt_final_exact_fit_cases": int(gt_final_exact_mask.sum()),
        "gt_tau_completed_exact_fit_cases": int(gt_tau_completed_exact_mask.sum()),
        "gt_tau_search_truncated_cases": int(gt_tau_search_truncated_mask.sum()),
        "argmax_tau_search_truncated_cases": int(case_df["argmax_tau_search_truncated"].astype(bool).sum()),
        "sktr_tau_search_truncated_cases": int(case_df["sktr_tau_search_truncated"].astype(bool).sum()),
        "gt_exact_fit_cases": int(gt_tau_completed_exact_mask.sum()),
        "gt_exact_fit_final_cases": int(gt_final_exact_mask.sum()),
        "case_output_consistent_all": case_output_consistent_all,
        "oracle_argmax_is_gt_all": oracle_argmax_is_gt_all,
        "prefix_exact_oracle_min_acc": prefix_exact_oracle_min_acc,
        "final_exact_oracle_min_acc": final_exact_oracle_min_acc,
        "tau_completed_exact_oracle_min_acc": tau_completed_exact_oracle_min_acc,
        "final_exact_oracle_floor_ok": final_exact_oracle_floor_ok,
        "tau_completed_exact_oracle_floor_ok": tau_completed_exact_oracle_floor_ok,
        "exact_fit_oracle_min_acc": tau_completed_exact_oracle_min_acc,
    }

    return case_df, summary


def write_summary(out_dir: Path, summaries: List[Dict[str, Any]]) -> None:
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    all_case_dfs: List[pd.DataFrame] = []

    for dataset in args.datasets:
        preset = RUN_PRESETS[dataset]
        n_folds = int(get_dataset_cv_config(dataset, args.data_root)["n_folds"])
        folds = range(1, n_folds + 1) if args.all_folds else [args.fold]
        for fold in folds:
            print(f"Analyzing {dataset} fold {fold}...", flush=True)
            case_df, summary = analyze_fold(
                dataset=dataset,
                fold=fold,
                preset=preset,
                args=args,
                out_dir=out_dir,
            )
            case_path = out_dir / f"{dataset}_fold{fold}_ceiling_cases.csv"
            case_df.to_csv(case_path, index=False)
            summary_path = out_dir / f"{dataset}_fold{fold}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            all_case_dfs.append(case_df)
            all_summaries.append(summary)

            if summary["oracle_acc_frame"] is None:
                acc_line = (
                    f"  frame acc: argmax={summary['argmax_acc_frame']:.4f}, "
                    f"sktr={summary['sktr_acc_frame']:.4f}, oracle=SKIPPED"
                )
            else:
                acc_line = (
                    f"  frame acc: argmax={summary['argmax_acc_frame']:.4f}, "
                    f"sktr={summary['sktr_acc_frame']:.4f}, "
                    f"oracle={summary['oracle_acc_frame']:.4f}"
                )
            print(
                f"  wrote {case_path}\n"
                f"{acc_line}",
                flush=True,
            )
            print(
                f"  GT prefix-exact cases: {summary['gt_prefix_exact_fit_cases']}/"
                f"{summary['test_cases_analyzed']}; "
                f"final-exact={summary['gt_final_exact_fit_cases']}/"
                f"{summary['test_cases_analyzed']}; "
                f"tau-completed={summary['gt_tau_completed_exact_fit_cases']}/"
                f"{summary['test_cases_analyzed']}; "
                f"tau-truncated={summary['gt_tau_search_truncated_cases']}; "
                f"config parity={summary['config_matches_run']}; "
                f"model parity={summary['model_size_matches_run_log']}; "
                f"case consistency={summary['case_output_consistent_all']}; "
                f"oracle_floor_exercised={summary['oracle_floor_exercised']}; "
                f"gate_ready={summary['gate_ready']}",
                flush=True,
            )

    if all_case_dfs:
        pd.concat(all_case_dfs, ignore_index=True).to_csv(
            out_dir / "all_ceiling_cases.csv", index=False
        )
    write_summary(out_dir, all_summaries)
    print(f"Done. Summary written under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
