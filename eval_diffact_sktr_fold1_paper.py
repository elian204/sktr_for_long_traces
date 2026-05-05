#!/usr/bin/env python3
"""
Evaluate SKTR on DiffAct softmax vs frame-wise argmax (per CV fold train/test).

Default: **fold 1** only. With ``--all-folds``, runs every fold for each selected dataset
(4 for GTEA, 5 for 50 Salads). For each **fold** `k`, loads softmax from
``baselines/DiffAct/results/<dataset>/softmax_fold{k}/`` (exported with
``*-Trained-S{k}.json``), or for **fold 1** only falls back to legacy ``.../softmax/`` if
the per-fold directory is missing.

Metrics use the same ASFormer-compatible evaluators as the rest of sktr_for_long_traces.
Pretrained weights are the official DiffAct ``release.model`` per split.
"""

from __future__ import annotations
from src.evaluation import compute_tas_metrics_asformer, compute_sktr_vs_argmax_metrics
from src.incremental_softmax_recovery import incremental_softmax_recovery
from src.cv_utils import (
    build_video_to_case_mapping,
    load_fold_case_ids,
    DEFAULT_DATA_ROOT,
    get_dataset_cv_config,
)
from src.utils import linear_prob_combiner, map_to_string_numbers

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re

import numpy as np
import pandas as pd

workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


# Defaults aligned with kfold_learning_curve_experiment / DATASET_HP_DEFAULTS for diffact
HP_STRATEGIES = {
    "trigram_heavy": [0.1, 0.15, 0.75],
    "unigram_super_heavy": [0.75, 0.15, 0.1],
}

DATASET_HP_DEFAULTS = {
    ("50salads", "diffact"): {"alpha": 0.3, "strategy": "unigram_super_heavy"},
    ("gtea", "diffact"): {"alpha": 0.95, "strategy": "trigram_heavy"},
}


def parse_log_level(value: str) -> int:
    if value.isdigit():
        return int(value)
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Unknown log level: {value}")
    return level


def resolve_background_label(dataset_name: str) -> None:
    return None


def resolve_diffact_softmax_dir(
    diffact_root: Path,
    dataset_name: str,
    fold: int,
    *,
    disallow_legacy: bool = False,
) -> Path:
    """
    Directory with DiffAct ``.npy`` bundle for this CV fold.

    Must match ``export_softmax.py`` run with ``configs/<...>-Trained-S{fold}.json`` and
    the matching ``trained_models/<...>-Trained-S{fold}/release.model`` checkpoint.
    Default export layout is ``.../results/<dataset>/softmax_fold{fold}``.

    When ``disallow_legacy`` is True (e.g. ``--all-folds``), only ``softmax_fold{fold}`` is
    accepted — never the legacy single ``softmax/`` tree — so every fold uses an explicitly
    aligned bundle.

    Otherwise, fold 1 may fall back to legacy ``softmax/`` if ``softmax_fold1`` is absent.
    """
    fold_dir = diffact_root / "results" / dataset_name / f"softmax_fold{fold}"
    legacy = diffact_root / "results" / dataset_name / "softmax"
    if fold_dir.is_dir() and (fold_dir / "video_index_map.txt").is_file():
        return fold_dir.resolve()
    if (
        not disallow_legacy
        and fold == 1
        and legacy.is_dir()
        and (legacy / "video_index_map.txt").is_file()
    ):
        return legacy.resolve()
    hint_legacy = ""
    if disallow_legacy and fold == 1 and legacy.is_dir():
        hint_legacy = (
            f"\n  (Legacy {legacy} exists but is ignored when evaluating all folds; "
            "export or symlink to softmax_fold1.)\n"
        )
    ex = (
        f"DiffAct softmax bundle not found for {dataset_name} fold {fold}.\n"
        f"  Expected: {fold_dir}\n"
        + ("" if disallow_legacy else f"  (fold 1 fallback): {legacy}\n")
        + hint_legacy
        + "Export with the matching checkpoint, e.g.:\n"
        f"  python3 export_softmax.py --config configs/...-Trained-S{fold}.json "
        f'--output_dir {diffact_root / "results" / dataset_name / f"softmax_fold{fold}"} '
        f"--root_data_dir <DATA_ROOT> --device 1\n"
    )
    raise FileNotFoundError(ex)


def build_base_config_for_fold(
    *,
    seed: int,
    chunk_size: int,
    prob_threshold: float,
    model_move_cost: float,
    state_mode: str,
    top_m: int,
    candidate_top_k: int,
    candidate_top_p: float,
    candidate_min_k: int,
    restrict_log_moves: bool,
    restrict_model_moves_to_tau: bool,
    enabled_cache_size: int,
    use_calibration: bool,
    workers: int,
    dataset_parallelization: bool,
) -> Dict[str, Any]:
    """Mirror kfold_learning_curve_experiment.build_base_config defaults."""
    return {
        "n_train_traces": None,
        "n_test_traces": None,
        "train_cases": None,
        "test_cases": None,
        "ensure_train_variant_diversity": False,
        "ensure_test_variant_diversity": False,
        "use_same_traces_for_train_test": False,
        "allow_train_cases_in_test": False,
        "compute_marking_transition_map": False,
        "sequential_sampling": False,
        "n_indices": 10**9,
        "n_per_run": None,
        "independent_sampling": True,
        "prob_threshold": prob_threshold,
        "chunk_size": chunk_size,
        "conformance_switch_penalty_weight": 1.0,
        "merge_mismatched_boundaries": False,
        "conditioning_combine_fn": linear_prob_combiner,
        "max_hist_len": 3,
        "conditioning_n_prev_labels": 3,
        "use_collapsed_runs": True,
        "cost_function": "linear",
        "model_move_cost": model_move_cost,
        "log_move_cost": 1.0,
        "tau_move_cost": 1e-6,
        "non_sync_penalty": 1.0,
        "use_calibration": use_calibration,
        "temp_bounds": (1.0, 10.0),
        "temperature": None,
        "verbose": False,
        "log_level": logging.ERROR,
        "round_precision": 2,
        "random_seed": seed,
        "save_model_path": None,
        "save_model": False,
        "parallel_processing": False,
        "dataset_parallelization": dataset_parallelization,
        "dataset_parallelization_context": None,
        "max_workers": workers if dataset_parallelization else 1,
        "conditioning_state_mode": state_mode,
        "conditioning_top_m": top_m,
        "candidate_top_p": candidate_top_p,
        "candidate_top_k": candidate_top_k,
        "candidate_min_k": candidate_min_k,
        "candidate_source": "conditioned",
        "candidate_apply_to_sync": True,
        "restrict_log_moves": restrict_log_moves,
        "restrict_model_moves_to_tau": restrict_model_moves_to_tau,
        "enabled_cache_size": enabled_cache_size,
    }


def parse_video_index_map(path: Path) -> List[Tuple[str, str]]:
    """Lines case_id<TAB>stem ordered by integer case id."""
    rows_lc: List[Tuple[int, str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = re.split(r"\s+", line, maxsplit=1)
            case_id, name = parts[0].strip(), parts[1].strip()
            rows_lc.append((int(case_id), name))
    rows_lc.sort(key=lambda x: x[0])
    return [(str(i), name) for i, name in rows_lc]


def load_diffact_softmax_and_aligned_df(
    dataset_name: str,
    softmax_dir: Path,
    data_root: Path,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[Tuple[str, str]]]:
    """
    Load .npy softmasks and build the event log from ``groundTruth/*.txt``.

    The ASFormer ``ground_truth.csv`` copied into the DiffAct export can have a
    different number of frames per case than the DiffAct decoder output; this
    path guarantees agreement with the exported softmax.
    """
    mapping_path = softmax_dir / "mapping.txt"
    label_to_idx: Dict[str, int] = {}
    with open(mapping_path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx, lab = parts
                label_to_idx[lab] = int(idx)

    entries = parse_video_index_map(softmax_dir / "video_index_map.txt")
    softmax_lst: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    gt_root = data_root / dataset_name / "groundTruth"

    for case_str, stem in entries:
        mat = np.load(softmax_dir / f"{case_str}.npy")
        softmax_lst.append(mat)
        gf = gt_root / f"{stem}.txt"
        raw = np.atleast_1d(np.loadtxt(gf, dtype=str))
        raw = np.asarray(raw, dtype=str)
        if len(raw) != mat.shape[1]:
            raise ValueError(
                f"{dataset_name} {stem}: len(gt)={len(raw)} vs softmax T={mat.shape[1]}"
            )
        for lab in raw:
            if lab not in label_to_idx:
                raise ValueError(f"Unknown label {lab!r} in {stem}")
            rows.append(
                {
                    "case:concept:name": str(case_str),
                    "concept:name": label_to_idx[lab],
                }
            )

    df = pd.DataFrame(rows)
    df, _ = map_to_string_numbers(df)
    return df, softmax_lst, entries


def verify_softmax_list(softmax_lst: List[np.ndarray], name: str) -> None:
    for i, m in enumerate(softmax_lst):
        if m.ndim != 2:
            raise ValueError(
                f"{name}: case {i} expected 2D softmax, got {m.shape}")
        col_sums = m.sum(axis=0)
        err = float(np.abs(col_sums - 1.0).max())
        if err > 0.02:
            raise ValueError(
                f"{name}: case {i} columns do not sum to ~1 (max err={err})")


def softmax_map_from_entries(
    entries: List[Tuple[str, str]], softmax_lst: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    if len(entries) != len(softmax_lst):
        raise ValueError("entries / softmax length mismatch")
    return {str(e[0]): softmax_lst[i] for i, e in enumerate(entries)}


def build_argmax_test_dataframe(
    df: pd.DataFrame,
    case_to_mat: Dict[str, np.ndarray],
    test_cases: List[str],
) -> pd.DataFrame:
    """Frame-wise argmax on exported probabilities; test videos only."""
    rows = []
    for case in test_cases:
        case_str = str(case)
        sub = df[df["case:concept:name"].astype(str) == case_str]
        if sub.empty:
            raise ValueError(f"No rows for test case {case}")
        mat = case_to_mat[case_str]
        gt = sub["concept:name"].astype(str).tolist()
        if mat.shape[1] != len(gt):
            raise ValueError(
                f"case {case}: softmax T={mat.shape[1]} vs gt len={len(gt)}"
            )
        pred_idx = np.argmax(mat, axis=0)
        pred = [str(int(p)) for p in pred_idx]
        for t in range(len(gt)):
            rows.append(
                {
                    "case:concept:name": case_str,
                    "ground_truth": gt[t],
                    "prediction": pred[t],
                }
            )
    return pd.DataFrame(rows)


def find_official_diffact_predictions(diffact_root: Path) -> Optional[Path]:
    """Discrete DiffAct predictions if any prediction/*.txt exists under the repo."""
    found = list(diffact_root.glob("**/prediction/*.txt"))
    return found[0].parent if found else None


def sanity_length_check(results_csv: Path) -> None:
    df = pd.read_csv(results_csv)
    for vid, g in df.groupby("case:concept:name", sort=False):
        n = len(g)
        if g["sktr_activity"].notna().sum() != n:
            raise ValueError(f"case {vid}: invalid SKTR rows")
        if len(g["ground_truth"]) != len(g["sktr_activity"]):
            raise ValueError(f"case {vid}: length mismatch gt vs sktr")


def print_sample_comparison(results_csv: Path, n_cases: int = 3, width: int = 48) -> None:
    df = pd.read_csv(results_csv)
    df["case:concept:name"] = df["case:concept:name"].astype(str)
    cases = df["case:concept:name"].unique().tolist()[:n_cases]
    lines = []
    for c in cases:
        g = df[df["case:concept:name"] == c].reset_index(drop=True)
        gt = g["ground_truth"].astype(str).tolist()
        sk = g["sktr_activity"].astype(str).tolist()
        am = g["argmax_activity"].astype(str).tolist()
        span = min(width, len(gt))
        lines.append(f"case {c} (first {span} frames):")
        lines.append(f"  GT    : {' '.join(gt[:span])}")
        lines.append(f"  Argmax: {' '.join(am[:span])}")
        lines.append(f"  SKTR  : {' '.join(sk[:span])}")
    print("\nSample argmax vs SKTR (test fold):\n" + "\n".join(lines))


def main():
    print("eval_diffact_sktr_fold1_paper: starting", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str,
                        default=os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--chunk-size", type=int, default=11,
                        help="Conformance chunk size (kfold default: 11)")
    parser.add_argument("--prob-threshold", type=float, default=1e-6)
    parser.add_argument("--model-move-cost", type=float, default=1.0)
    parser.add_argument("--state-mode", type=str,
                        default="topm", choices=["exact", "topm"])
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--candidate-top-k", type=int, default=3)
    parser.add_argument("--candidate-top-p", type=float, default=1.0)
    parser.add_argument("--candidate-min-k", type=int, default=1)
    parser.add_argument(
        "--restrict-log-moves",
        action="store_true",
        help="Approximate SKTR: restrict log moves to top-1 observed label plus previous label.",
    )
    parser.add_argument(
        "--restrict-model-moves-to-tau",
        action="store_true",
        help="Approximate SKTR: allow only tau transitions as model moves.",
    )
    parser.add_argument(
        "--enabled-cache-size",
        type=int,
        default=100000,
        help="Max PetriNet enabled-transition cache entries per worker.",
    )
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--inner-parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sktr-log-level",
        type=parse_log_level,
        default=None,
        help="Opt in to SKTR logging, e.g. INFO or DEBUG. Default preserves quiet eval behavior.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["50salads", "gtea"],
        choices=["50salads", "gtea"],
        help="Which datasets to run (default: both)",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run every CV fold per dataset (GTEA: 4 folds, 50 Salads: 5). Ignores single --fold.",
    )
    args = parser.parse_args()

    diffact_root = workspace_root / "baselines" / "DiffAct"
    out_dir = Path(args.out_dir or workspace_root /
                   "results" / "paper_diffact_fold1")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(args.datasets)
    table_rows = []

    official_note = find_official_diffact_predictions(diffact_root)
    if official_note:
        print(
            f"Note: found prediction artifacts under {official_note} (not loaded; export-only pipeline).",
            flush=True,
        )
    else:
        print(
            "Note: no separate DiffAct official prediction folder found; skipping official discrete baseline.",
            flush=True,
        )

    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)

        n_folds = int(get_dataset_cv_config(ds, args.data_root)["n_folds"])
        if args.all_folds:
            folds_to_run = list(range(1, n_folds + 1))
        else:
            if args.fold < 1 or args.fold > n_folds:
                raise ValueError(
                    f"{ds}: --fold must be in 1..{n_folds} (got {args.fold})"
                )
            folds_to_run = [args.fold]

        for fold in folds_to_run:
            softmax_dir = resolve_diffact_softmax_dir(
                diffact_root, ds, fold, disallow_legacy=args.all_folds
            )
            print(
                f"Using DiffAct softmax bundle: {softmax_dir} "
                f"(checkpoint naming: *-Trained-S{fold})",
                flush=True,
            )

            df, softmax_lst, entries = load_diffact_softmax_and_aligned_df(
                ds, softmax_dir, Path(args.data_root)
            )
            verify_softmax_list(softmax_lst, ds)

            case_to_mat = softmax_map_from_entries(entries, softmax_lst)

            video_map = build_video_to_case_mapping(
                ds,
                "diffact",
                video_index_map_path=softmax_dir / "video_index_map.txt",
            )

            print(f"\n--- {ds} fold {fold}/{n_folds} ---", flush=True)
            splits = load_fold_case_ids(
                ds, fold, video_map, data_root=args.data_root
            )
            train_cases = splits["train"]
            test_cases = splits["test"]
            print(
                f"Fold {fold}: {len(train_cases)} train cases, "
                f"{len(test_cases)} test cases",
                flush=True,
            )

            hp = DATASET_HP_DEFAULTS[(ds, "diffact")]
            weights = HP_STRATEGIES[hp["strategy"]]

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
                restrict_log_moves=args.restrict_log_moves,
                restrict_model_moves_to_tau=args.restrict_model_moves_to_tau,
                enabled_cache_size=args.enabled_cache_size,
                use_calibration=args.use_calibration,
                workers=args.workers,
                dataset_parallelization=args.inner_parallel,
            )
            if args.sktr_log_level is not None:
                base["verbose"] = True
                base["log_level"] = args.sktr_log_level
            cfg = {
                **base,
                "train_cases": train_cases,
                "test_cases": test_cases,
                "conditioning_alpha": hp["alpha"],
                "conditioning_interpolation_weights": weights,
            }

            results_csv = out_dir / f"{ds}_diffact_fold{fold}_sktr.csv"
            results_df, _, _ = incremental_softmax_recovery(
                df=df, softmax_lst=softmax_lst, **cfg
            )
            results_df.to_csv(results_csv, index=False)
            sanity_length_check(results_csv)
            print_sample_comparison(results_csv)

            metrics_bundle = compute_sktr_vs_argmax_metrics(
                str(results_csv),
                case_col="case:concept:name",
                sktr_pred_col="sktr_activity",
                argmax_pred_col="argmax_activity",
                gt_col="ground_truth",
                background=resolve_background_label(ds),
                dataset_name=ds,
                mapping_path=softmax_dir / "mapping.txt",
            )

            argmax_df = build_argmax_test_dataframe(
                df, case_to_mat, test_cases)
            argmax_metrics = compute_tas_metrics_asformer(
                argmax_df,
                pred_col="prediction",
                gt_col="ground_truth",
                case_col="case:concept:name",
                background=resolve_background_label(ds),
                dataset_name=ds,
                mapping_path=softmax_dir / "mapping.txt",
            )

            # Table: use standalone argmax-on-exported-softmax for "DiffAct (argmax)";
            # SKTR from bundle; optional: show CSV argmax column vs standalone (debug)
            table_rows.append(
                {
                    "Fold": fold,
                    "Dataset": ds.replace("50salads", "50Salads"),
                    "Method": "DiffAct (argmax)",
                    "Edit": round(argmax_metrics["edit"], 2),
                    "F1@10": round(argmax_metrics["f1@10"], 2),
                    "F1@25": round(argmax_metrics["f1@25"], 2),
                    "F1@50": round(argmax_metrics["f1@50"], 2),
                    "Acc": round(argmax_metrics["acc"], 2),
                }
            )
            sk = metrics_bundle["sktr"]
            table_rows.append(
                {
                    "Fold": fold,
                    "Dataset": ds.replace("50salads", "50Salads"),
                    "Method": "DiffAct + SKTR",
                    "Edit": round(sk["edit"], 2),
                    "F1@10": round(sk["f1@10"], 2),
                    "F1@25": round(sk["f1@25"], 2),
                    "F1@50": round(sk["f1@50"], 2),
                    "Acc": round(sk["acc"], 2),
                }
            )

            with open(out_dir / f"{ds}_metrics_fold{fold}.json", "w") as f:
                json.dump(
                    {
                        "argmax_export_softmax_test_fold": argmax_metrics,
                        "sktr": sk,
                        "argmax_from_sktr_csv": metrics_bundle["argmax"],
                        "hp": hp,
                        "train_cases": len(train_cases),
                        "test_cases": len(test_cases),
                        "diffact_softmax_dir": str(softmax_dir),
                        "diffact_checkpoint_naming": f"<Dataset>-Trained-S{fold} (see baselines/DiffAct/configs/)",
                    },
                    f,
                    indent=2,
                )

    tag = "all_folds" if args.all_folds else f"fold{args.fold}"
    summary_df = pd.DataFrame(table_rows)
    summary_path = out_dir / f"table_diffact_{tag}.md"
    summary_csv = out_dir / f"table_diffact_{tag}.csv"
    summary_df.to_csv(summary_csv, index=False)
    try:
        md = summary_df.to_markdown(index=False)
    except Exception:
        md = summary_df.to_string(index=False)
    with open(summary_path, "w") as f:
        f.write(md)

    salads_only = summary_df[summary_df["Dataset"] == "50Salads"].copy()
    if not salads_only.empty:
        salads_csv = out_dir / f"table_50salads_{tag}_only.csv"
        salads_only.to_csv(salads_csv, index=False)
        try:
            md_s = salads_only.to_markdown(index=False)
        except Exception:
            md_s = salads_only.to_string(index=False)
        with open(out_dir / f"table_50salads_{tag}_only.md", "w") as f:
            f.write(md_s)

    print("\n=== Summary table ===\n")
    print(md)
    extra = ""
    if not salads_only.empty:
        extra = f", {salads_csv.name} and table_50salads_{tag}_only.md"
    print(
        f"\nWrote {summary_path}, {summary_csv.name}{extra}, and CSV/JSON details under {out_dir}")


if __name__ == "__main__":
    main()
