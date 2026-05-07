#!/usr/bin/env python3
"""
Profile DiffAct + SKTR on one fold/test case without running a full fold.

The default command profiles the same recovery entry point used by
eval_diffact_sktr_fold1_paper.py, but with test_cases restricted to a single
case and multiprocessing disabled. Use --target conformance to profile only
process_trace_chunked after model/probability setup has completed.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import pickle
import pstats
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_diffact_sktr_fold1_paper import (  # noqa: E402
    DATASET_HP_DEFAULTS,
    HP_STRATEGIES,
    build_base_config_for_fold,
    load_diffact_softmax_and_aligned_df,
    resolve_diffact_softmax_dir,
    verify_softmax_list,
)
from src.conformance_checking import process_trace_chunked  # noqa: E402
from src.cv_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_video_to_case_mapping,
    load_fold_case_ids,
)
from src.data_processing import (  # noqa: E402
    filter_indices,
    prepare_softmax,
    select_softmax_matrices,
    split_train_test,
    validate_sequential_case_ids,
)
from src.incremental_softmax_recovery import incremental_softmax_recovery  # noqa: E402
from src.petri_model import build_probability_dict, discover_petri_net  # noqa: E402
from src.utils import make_cost_function  # noqa: E402


class ProfileBudgetExpired(TimeoutError):
    """Internal stop signal used to end bounded profiling runs cleanly."""


def _parse_log_level(value: str) -> int:
    if value.isdigit():
        return int(value)
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Unknown log level: {value}")
    return level


def _parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative; use 0 to disable")
    return parsed


def _parse_nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected float, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative")
    return parsed


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _case_lengths(case_ids: List[str], softmax_lst: List[np.ndarray]) -> Dict[str, int]:
    return {str(case_id): int(softmax_lst[int(case_id)].shape[1]) for case_id in case_ids}


def _select_case(
    test_cases: List[str],
    softmax_lst: List[np.ndarray],
    case_id: Optional[str],
    selector: str,
) -> str:
    test_cases = [str(case) for case in test_cases]
    if not test_cases:
        raise ValueError("No test cases found for fold")

    if case_id is not None:
        case_id = str(case_id)
        if case_id not in test_cases:
            raise ValueError(f"case {case_id!r} is not in fold test cases: {test_cases}")
        return case_id

    lengths = _case_lengths(test_cases, softmax_lst)
    if selector == "first":
        return test_cases[0]
    if selector == "shortest":
        return min(test_cases, key=lambda c: (lengths[c], int(c)))
    if selector == "longest":
        return max(test_cases, key=lambda c: (lengths[c], -int(c)))
    raise ValueError(f"Unknown case selector: {selector}")


def _trim_case_frames(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    case_id: str,
    max_frames: Optional[int],
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    if max_frames is None:
        return df, softmax_lst
    if max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    case_col = "case:concept:name"
    mask = df[case_col].astype(str) == str(case_id)
    case_len = int(mask.sum())
    keep_count = min(max_frames, case_len)
    if keep_count == case_len:
        return df, softmax_lst

    case_indices = df.index[mask].tolist()
    drop_indices = case_indices[keep_count:]
    trimmed_df = df.drop(index=drop_indices).reset_index(drop=True)

    trimmed_softmax = list(softmax_lst)
    case_idx = int(case_id)
    trimmed_softmax[case_idx] = np.asarray(trimmed_softmax[case_idx])[:, :keep_count]
    return trimmed_df, trimmed_softmax


def _run_profile(
    func: Callable[[], Any],
    stats_path: Path,
    text_path: Path,
    sort_key: str,
    top: int,
    timeout_seconds: Optional[float] = None,
) -> Tuple[Any, float, Dict[str, Any]]:
    profile = cProfile.Profile()
    started = time.perf_counter()
    result = None
    status: Dict[str, Any] = {"profile_interrupted": False, "profile_error": None}

    old_handler = None
    old_timer = None

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise ProfileBudgetExpired(f"profile time budget expired after {timeout_seconds:.3f}s")

    try:
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise ValueError("--profile-seconds must be positive when set")
            old_handler = signal.getsignal(signal.SIGALRM)
            old_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            signal.signal(signal.SIGALRM, _timeout_handler)
        result = profile.runcall(func)
    except ProfileBudgetExpired as exc:
        status["profile_interrupted"] = True
        status["profile_error"] = str(exc)
    finally:
        if timeout_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
            if old_timer is not None and old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
        elapsed = time.perf_counter() - started
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        profile.dump_stats(str(stats_path))
        with text_path.open("w") as fh:
            stats = pstats.Stats(profile, stream=fh).strip_dirs().sort_stats(sort_key)
            stats.print_stats(top)
    return result, elapsed, status


def _build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    base = build_base_config_for_fold(
        seed=args.seed,
        chunk_size=args.chunk_size,
        prob_threshold=args.prob_threshold,
        model_move_cost=args.model_move_cost,
        state_mode=args.state_mode,
        top_m=args.top_m,
        candidate_top_k=args.candidate_top_k,
        candidate_top_p=args.candidate_top_p,
        candidate_min_k=args.candidate_min_k,
        conformance_switch_penalty_weight=args.conformance_switch_penalty_weight,
        restrict_log_moves=args.restrict_log_moves,
        restrict_model_moves_to_tau=args.restrict_model_moves_to_tau,
        max_consecutive_tau_moves=(
            None if args.max_consecutive_tau_moves == 0 else args.max_consecutive_tau_moves
        ),
        dijkstra_beam_width=(
            None if args.dijkstra_beam_width == 0 else args.dijkstra_beam_width
        ),
        dijkstra_beam_cost_delta=args.dijkstra_beam_cost_delta,
        progress_log_interval_chunks=args.progress_log_interval_chunks,
        enabled_cache_size=args.enabled_cache_size,
        use_calibration=args.use_calibration,
        workers=1,
        dataset_parallelization=False,
    )
    base.update(
        {
            "parallel_processing": False,
            "dataset_parallelization": False,
            "max_workers": 1,
            "verbose": True,
            "log_level": args.log_level,
        }
    )
    return base


def _setup_cache_metadata(
    args: argparse.Namespace,
    fold_inputs: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset": args.dataset,
        "fold": args.fold,
        "softmax_dir": str(fold_inputs["softmax_dir"]),
        "train_cases": list(fold_inputs["train_cases"]),
        "hp": dict(fold_inputs["hp"]),
        "weights": list(fold_inputs["weights"]),
        "max_hist_len": cfg["max_hist_len"],
        "compute_marking_transition_map": cfg.get("compute_marking_transition_map", True),
        "precompute_marking_transition_map": cfg.get("precompute_marking_transition_map", False),
    }


def _load_setup_cache(
    cache_path: Optional[Path],
    metadata: Dict[str, Any],
    refresh: bool,
) -> Optional[Dict[str, Any]]:
    if cache_path is None or refresh or not cache_path.exists():
        return None
    with cache_path.open("rb") as fh:
        payload = pickle.load(fh)
    if payload.get("metadata") != metadata:
        return None
    return payload


def _save_setup_cache(
    cache_path: Optional[Path],
    metadata: Dict[str, Any],
    model: Any,
    prob_dict_uncollapsed: Dict[Tuple[str, ...], Dict[str, float]],
    prob_dict_collapsed: Dict[Tuple[str, ...], Dict[str, float]],
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(
            {
                "metadata": metadata,
                "model": model,
                "prob_dict_uncollapsed": prob_dict_uncollapsed,
                "prob_dict_collapsed": prob_dict_collapsed,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _prepare_fold_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    diffact_root = Path(args.diffact_root)
    softmax_dir = resolve_diffact_softmax_dir(
        diffact_root,
        args.dataset,
        args.fold,
        disallow_legacy=args.disallow_legacy,
    )
    df, softmax_lst, entries = load_diffact_softmax_and_aligned_df(
        args.dataset,
        softmax_dir,
        Path(args.data_root),
    )
    verify_softmax_list(softmax_lst, args.dataset)

    video_map = build_video_to_case_mapping(
        args.dataset,
        "diffact",
        video_index_map_path=softmax_dir / "video_index_map.txt",
    )
    splits = load_fold_case_ids(args.dataset, args.fold, video_map, data_root=args.data_root)
    train_cases = [str(case) for case in splits["train"]]
    test_cases = [str(case) for case in splits["test"]]
    selected_case = _select_case(test_cases, softmax_lst, args.case_id, args.case_selector)
    df, softmax_lst = _trim_case_frames(df, softmax_lst, selected_case, args.max_frames)

    hp = DATASET_HP_DEFAULTS[(args.dataset, "diffact")]
    weights = HP_STRATEGIES[hp["strategy"]]
    case_to_stem = {str(case): stem for case, stem in entries}
    lengths = _case_lengths(test_cases, softmax_lst)

    return {
        "df": df,
        "softmax_lst": softmax_lst,
        "softmax_dir": softmax_dir,
        "train_cases": train_cases,
        "test_cases": test_cases,
        "selected_case": selected_case,
        "selected_stem": case_to_stem.get(selected_case),
        "selected_frames": lengths[selected_case],
        "hp": hp,
        "weights": weights,
    }


def _profile_recovery(
    fold_inputs: Dict[str, Any],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    run_cfg = {
        **cfg,
        "train_cases": fold_inputs["train_cases"],
        "test_cases": [fold_inputs["selected_case"]],
        "conditioning_alpha": fold_inputs["hp"]["alpha"],
        "conditioning_interpolation_weights": fold_inputs["weights"],
    }

    def run() -> Tuple[pd.DataFrame, Dict[str, List[float]], Any]:
        return incremental_softmax_recovery(
            df=fold_inputs["df"],
            softmax_lst=fold_inputs["softmax_lst"],
            **run_cfg,
        )

    result, elapsed, profile_status = _run_profile(
        run,
        args.stats_path,
        args.text_path,
        args.sort,
        args.top,
        args.profile_seconds,
    )
    summary = {
        "target": "recovery",
        "elapsed_seconds": elapsed,
        **profile_status,
    }
    if result is None:
        summary.update(
            {
                "records": None,
                "sktr_accuracy": None,
                "argmax_accuracy": None,
            }
        )
        return summary

    results_df, accuracy_dict, _ = result
    if args.csv_path is not None:
        results_df.to_csv(args.csv_path, index=False)

    summary.update(
        {
            "records": int(len(results_df)),
            "sktr_accuracy": accuracy_dict.get("sktr_accuracy"),
            "argmax_accuracy": accuracy_dict.get("argmax_accuracy"),
        }
    )
    return summary


def _build_conformance_setup(
    fold_inputs: Dict[str, Any],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if cfg["use_calibration"]:
        raise ValueError("--use-calibration is only supported with --target recovery")

    setup_start = time.perf_counter()
    df = fold_inputs["df"]
    softmax_lst = fold_inputs["softmax_lst"]
    validate_sequential_case_ids(df, softmax_lst)

    softmax_np = prepare_softmax(softmax_lst)
    filtered_log, filtered_softmax = filter_indices(
        df,
        softmax_np,
        n_indices=cfg["n_indices"],
        n_per_run=cfg["n_per_run"],
        sequential_sampling=cfg["sequential_sampling"],
        independent_sampling=cfg["independent_sampling"],
        random_seed=cfg["random_seed"],
    )
    train_df, test_df = split_train_test(
        filtered_log,
        cfg["n_train_traces"],
        cfg["n_test_traces"],
        fold_inputs["train_cases"],
        [fold_inputs["selected_case"]],
        ensure_train_variant_diversity=cfg["ensure_train_variant_diversity"],
        ensure_test_variant_diversity=cfg["ensure_test_variant_diversity"],
        allow_train_cases_in_test=cfg["allow_train_cases_in_test"],
        random_seed=cfg["random_seed"],
    )

    cache_metadata = _setup_cache_metadata(args, fold_inputs, cfg)
    cache_payload = _load_setup_cache(
        args.setup_cache_path,
        cache_metadata,
        refresh=args.refresh_setup_cache,
    )
    setup_cache_hit = cache_payload is not None
    setup_cache_path = str(args.setup_cache_path) if args.setup_cache_path is not None else None

    if cache_payload is not None:
        model_start = time.perf_counter()
        model = cache_payload["model"]
        model.enable_caching(True, max_size=args.enabled_cache_size)
        prob_dict_uncollapsed = cache_payload["prob_dict_uncollapsed"]
        prob_dict_collapsed = cache_payload["prob_dict_collapsed"]
        model_elapsed = time.perf_counter() - model_start
        prob_elapsed = 0.0
    else:
        model_start = time.perf_counter()
        model = discover_petri_net(train_df)
        model.enable_caching(True, max_size=args.enabled_cache_size)
        if cfg.get("compute_marking_transition_map", True):
            if cfg.get("precompute_marking_transition_map", False):
                model.build_marking_transition_map()
                model._allow_lazy_map_build = False
            else:
                model._allow_lazy_map_build = True
                model.get_or_build_marking_transition_map(max_tau_depth=100)
                model.get_tau_reachable_transitions_initial(max_tau_depth=100)
        else:
            model._allow_lazy_map_build = False
            model.marking_transition_map = None
        model_elapsed = time.perf_counter() - model_start

        prob_start = time.perf_counter()
        prob_dict_uncollapsed: Dict[Tuple[str, ...], Dict[str, float]] = {}
        prob_dict_collapsed: Dict[Tuple[str, ...], Dict[str, float]] = {}
        if cfg["conformance_switch_penalty_weight"] > 0.0 or fold_inputs["hp"]["alpha"] is not None:
            prob_dict_uncollapsed = build_probability_dict(
                train_df,
                max_hist_len=cfg["max_hist_len"],
                use_collapsed=False,
            )
            prob_dict_collapsed = build_probability_dict(
                train_df,
                max_hist_len=cfg["max_hist_len"],
                use_collapsed=True,
            )
        prob_elapsed = time.perf_counter() - prob_start
        _save_setup_cache(
            args.setup_cache_path,
            cache_metadata,
            model,
            prob_dict_uncollapsed,
            prob_dict_collapsed,
        )

    if filtered_softmax is None:
        raise ValueError("filtered softmax matrices are required")
    test_softmax = select_softmax_matrices(filtered_softmax, test_df)[0]
    ground_truth = (
        test_df.groupby("case:concept:name", sort=False)["concept:name"].apply(list).to_dict()[
            fold_inputs["selected_case"]
        ]
    )
    cost_fn = make_cost_function(
        base=cfg["cost_function"],
        model_move=cfg["model_move_cost"],
        log_move=cfg["log_move_cost"],
        tau_move=cfg["tau_move_cost"],
        round_precision=cfg["round_precision"],
    )
    setup_elapsed = time.perf_counter() - setup_start

    return {
        "model": model,
        "cost_fn": cost_fn,
        "test_softmax": test_softmax,
        "ground_truth": ground_truth,
        "prob_dict_uncollapsed": prob_dict_uncollapsed,
        "prob_dict_collapsed": prob_dict_collapsed,
        "setup_elapsed_seconds": setup_elapsed,
        "model_elapsed_seconds": model_elapsed,
        "prob_elapsed_seconds": prob_elapsed,
        "train_events": int(len(train_df)),
        "test_events": int(len(test_df)),
        "model_places": int(len(model.places)),
        "model_transitions": int(len(model.transitions)),
        "prob_histories_uncollapsed": int(len(prob_dict_uncollapsed)),
        "prob_histories_collapsed": int(len(prob_dict_collapsed)),
        "setup_cache_hit": setup_cache_hit,
        "setup_cache_path": setup_cache_path,
        "enabled_cache_size": int(args.enabled_cache_size),
    }


def _profile_conformance(
    fold_inputs: Dict[str, Any],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    setup = _build_conformance_setup(fold_inputs, cfg, args)
    partial_call_counter = {"completed": 0}
    search_stats: Dict[str, Any] = {}
    if args.max_partial_calls is not None:
        if args.max_partial_calls <= 0:
            raise ValueError("--max-partial-calls must be positive when set")
        original_partial = setup["model"].partial_trace_conformance

        def limited_partial_trace_conformance(*partial_args: Any, **partial_kwargs: Any) -> Dict[str, Any]:
            result = original_partial(*partial_args, **partial_kwargs)
            partial_call_counter["completed"] += 1
            if partial_call_counter["completed"] >= args.max_partial_calls:
                raise ProfileBudgetExpired(
                    f"partial-call budget expired after {partial_call_counter['completed']} call(s)"
                )
            return result

        setup["model"].partial_trace_conformance = limited_partial_trace_conformance

    def run() -> Tuple[List[str], List[float]]:
        return process_trace_chunked(
            softmax_matrix=setup["test_softmax"],
            model=setup["model"],
            cost_fn=setup["cost_fn"],
            chunk_size=cfg["chunk_size"],
            eps=cfg["prob_threshold"],
            inline_progress=True,
            progress_prefix=f"case {fold_inputs['selected_case']}",
            prob_dict_uncollapsed=setup["prob_dict_uncollapsed"],
            prob_dict_collapsed=setup["prob_dict_collapsed"],
            switch_penalty_weight=cfg["conformance_switch_penalty_weight"],
            use_state_caching=cfg.get("use_state_caching", True),
            merge_mismatched_boundaries=cfg.get("merge_mismatched_boundaries", True),
            conditioning_alpha=fold_inputs["hp"]["alpha"],
            conditioning_combine_fn=cfg["conditioning_combine_fn"],
            conditioning_n_prev_labels=cfg["conditioning_n_prev_labels"],
            conditioning_interpolation_weights=fold_inputs["weights"],
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
            dijkstra_beam_width=cfg["dijkstra_beam_width"],
            dijkstra_beam_cost_delta=cfg["dijkstra_beam_cost_delta"],
            progress_log_interval_chunks=cfg["progress_log_interval_chunks"],
            profile_stats=search_stats,
        )

    result, elapsed, profile_status = _run_profile(
        run,
        args.stats_path,
        args.text_path,
        args.sort,
        args.top,
        args.profile_seconds,
    )
    summary = {
        "target": "conformance",
        "elapsed_seconds": elapsed,
        "max_partial_calls": args.max_partial_calls,
        "partial_calls_completed": partial_call_counter["completed"],
        **profile_status,
        **{
            key: value
            for key, value in setup.items()
            if key
            in {
                "setup_elapsed_seconds",
                "model_elapsed_seconds",
                "prob_elapsed_seconds",
                "train_events",
                "test_events",
                "model_places",
                "model_transitions",
                "prob_histories_uncollapsed",
                "prob_histories_collapsed",
                "setup_cache_hit",
                "setup_cache_path",
                "enabled_cache_size",
            }
        },
        "search_stats": search_stats,
    }
    if result is None:
        summary.update({"predictions": None, "move_costs": None})
        return summary

    sktr_preds, move_costs = result
    ground_truth = [str(x) for x in setup["ground_truth"]]
    argmax_preds = [str(idx) for idx in np.argmax(setup["test_softmax"], axis=0)]
    n_compare = min(len(ground_truth), len(sktr_preds), len(argmax_preds))
    sktr_accuracy = (
        sum(1 for pred, true in zip(sktr_preds[:n_compare], ground_truth[:n_compare]) if pred == true) / n_compare
        if n_compare
        else None
    )
    argmax_accuracy = (
        sum(1 for pred, true in zip(argmax_preds[:n_compare], ground_truth[:n_compare]) if pred == true) / n_compare
        if n_compare
        else None
    )
    summary.update(
        {
            "predictions": int(len(sktr_preds)),
            "move_costs": int(len(move_costs)),
            "total_move_cost": float(sum(move_costs)),
            "sktr_accuracy": sktr_accuracy,
            "argmax_accuracy": argmax_accuracy,
            "accuracy_frames_compared": int(n_compare),
        }
    )
    return summary


def _default_output_paths(args: argparse.Namespace, case_id: str) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = (
        f"sktr_{args.dataset}_fold{args.fold}_case{case_id}_"
        f"{args.target}_{stamp}"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.stats_path = Path(args.stats_path) if args.stats_path else out_dir / f"{stem}.prof"
    args.text_path = Path(args.text_path) if args.text_path else out_dir / f"{stem}.txt"
    args.summary_path = (
        Path(args.summary_path) if args.summary_path else out_dir / f"{stem}.summary.json"
    )
    args.csv_path = Path(args.csv_path) if args.csv_path else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile SKTR incremental softmax recovery on one DiffAct fold case."
    )
    parser.add_argument("--dataset", choices=["50salads", "gtea"], default="50salads")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--case-id", default=None, help="Fold test case id to profile")
    parser.add_argument(
        "--case-selector",
        choices=["longest", "shortest", "first"],
        default="longest",
        help="Case selection when --case-id is omitted",
    )
    parser.add_argument(
        "--target",
        choices=["recovery", "conformance"],
        default="recovery",
        help="Profile whole incremental_softmax_recovery or just process_trace_chunked",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", default=str(REPO_ROOT / "baselines" / "DiffAct"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "profiling"))
    parser.add_argument("--setup-cache-path", default=None)
    parser.add_argument("--no-setup-cache", action="store_true")
    parser.add_argument("--refresh-setup-cache", action="store_true")
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--text-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--csv-path", default=None, help="Optional CSV for recovery target results")
    parser.add_argument("--sort", default="cumulative", choices=["cumulative", "time", "calls", "name", "filename", "line"])
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--log-level", type=_parse_log_level, default=logging.INFO)
    parser.add_argument(
        "--profile-seconds",
        type=float,
        default=None,
        help="Interrupt the profiled region after this many seconds and still write partial cProfile stats",
    )
    parser.add_argument(
        "--max-partial-calls",
        type=int,
        default=None,
        help="Conformance target only: stop after this many partial_trace_conformance calls",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Trim selected test case to first N frames for smoke profiling")
    parser.add_argument("--enabled-cache-size", type=int, default=10000, help="Max entries for PetriNet enabled-transition cache")
    parser.add_argument("--disallow-legacy", action="store_true", help="Require softmax_foldK instead of fold-1 legacy fallback")

    parser.add_argument("--chunk-size", type=int, default=11)
    parser.add_argument("--prob-threshold", type=float, default=1e-6)
    parser.add_argument("--model-move-cost", type=float, default=1.0)
    parser.add_argument("--state-mode", choices=["exact", "topm"], default="topm")
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--candidate-top-k", type=int, default=3)
    parser.add_argument("--candidate-top-p", type=float, default=1.0)
    parser.add_argument("--candidate-min-k", type=int, default=1)
    parser.add_argument(
        "--conformance-switch-penalty-weight",
        type=_parse_nonnegative_float,
        default=1.0,
        help="Penalty added when a sync/log move changes label; 0 disables.",
    )
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument(
        "--restrict-log-moves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict log moves to top-1 observed label plus previous label; enabled by default.",
    )
    parser.add_argument("--restrict-model-moves-to-tau", action="store_true")
    parser.add_argument(
        "--max-consecutive-tau-moves",
        type=_parse_nonnegative_int,
        default=8,
        help="Cap consecutive direct tau/model-quiet moves; use 0 to disable.",
    )
    parser.add_argument(
        "--dijkstra-beam-width",
        type=_parse_nonnegative_int,
        default=0,
        help="Keep at most this many states per timestamp/label beam bucket; 0 disables.",
    )
    parser.add_argument(
        "--dijkstra-beam-cost-delta",
        type=_parse_nonnegative_float,
        default=None,
        help="Keep states within this cost delta of the best state in each beam bucket.",
    )
    parser.add_argument(
        "--progress-log-interval-chunks",
        type=_parse_nonnegative_int,
        default=0,
        help="Log conformance progress every N temporal chunks; 0 disables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    fold_inputs = _prepare_fold_inputs(args)
    _default_output_paths(args, fold_inputs["selected_case"])
    cfg = _build_base_config(args)
    if args.no_setup_cache:
        args.setup_cache_path = None
    elif args.setup_cache_path is None:
        args.setup_cache_path = (
            Path(args.out_dir)
            / "setup_cache"
            / f"{args.dataset}_fold{args.fold}_diffact_sktr_setup.pkl"
        )
    else:
        args.setup_cache_path = Path(args.setup_cache_path)

    print(
        "Profiling "
        f"{args.dataset} fold {args.fold} case {fold_inputs['selected_case']} "
        f"({fold_inputs['selected_stem']}, {fold_inputs['selected_frames']} frames), "
        f"target={args.target}",
        flush=True,
    )
    print(f"Writing cProfile binary stats to {args.stats_path}", flush=True)
    print(f"Writing pstats text report to {args.text_path}", flush=True)

    if args.target == "recovery":
        summary = _profile_recovery(fold_inputs, cfg, args)
    else:
        summary = _profile_conformance(fold_inputs, cfg, args)

    summary.update(
        {
            "dataset": args.dataset,
            "fold": args.fold,
            "case_id": fold_inputs["selected_case"],
            "case_stem": fold_inputs["selected_stem"],
            "case_frames": fold_inputs["selected_frames"],
            "train_cases": len(fold_inputs["train_cases"]),
            "fold_test_cases": len(fold_inputs["test_cases"]),
            "softmax_dir": str(fold_inputs["softmax_dir"]),
            "stats_path": str(args.stats_path),
            "text_path": str(args.text_path),
            "hp": fold_inputs["hp"],
            "weights": fold_inputs["weights"],
            "chunk_size": cfg["chunk_size"],
            "candidate_top_k": cfg["candidate_top_k"],
            "conformance_switch_penalty_weight": cfg["conformance_switch_penalty_weight"],
            "conditioning_state_mode": cfg["conditioning_state_mode"],
            "conditioning_top_m": cfg["conditioning_top_m"],
            "max_consecutive_tau_moves": cfg["max_consecutive_tau_moves"],
            "dijkstra_beam_width": cfg["dijkstra_beam_width"],
            "dijkstra_beam_cost_delta": cfg["dijkstra_beam_cost_delta"],
            "progress_log_interval_chunks": cfg["progress_log_interval_chunks"],
            "max_frames": args.max_frames,
        }
    )
    args.summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"Done in {summary['elapsed_seconds']:.3f}s; summary: {args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
