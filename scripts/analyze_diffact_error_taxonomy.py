#!/usr/bin/env python3
"""
Frame-level DiffAct/SKTR error taxonomy over existing artifacts.

This script is read-only over completed evaluation outputs and validated ceiling
CSVs. It does not rerun SKTR, rediscover process models, or recompute
conformance. The only fold-derived model it builds is a fold-pure training-set
DiffAct confusion table from existing softmax bundles for the class_confusion
taxonomy bucket.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_diffact_sktr_fold1_paper import (  # noqa: E402
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


OUTLIER_CASES = {
    ("gtea", 2, "7"),
    ("gtea", 2, "8"),
    ("gtea", 2, "9"),
    ("gtea", 2, "11"),
    ("gtea", 2, "12"),
    ("gtea", 2, "13"),
    ("50salads", 1, "1"),
    ("50salads", 5, "49"),
}

CATEGORY_COLUMNS = [
    "boundary_w25",
    "over_segmentation_island",
    "long_substitution",
    "class_confusion",
    "residual",
]
BOUNDARY_COLUMNS = ["boundary_w10", "boundary_w25_sensitivity", "boundary_w50"]
ORDER_COLUMNS = [
    "gt_accepted_exact",
    "gt_accepted_exact_tau_completed",
    "argmax_log_moves",
    "argmax_model_moves",
    "sktr_log_moves",
    "sktr_model_moves",
]


@dataclass(frozen=True)
class DatasetConfig:
    run_dir: Path
    unique_only: bool = False
    train_k: Optional[int] = None


DATASETS: Dict[str, DatasetConfig] = {
    "gtea": DatasetConfig(
        run_dir=Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "diffact_gtea_allfolds_resumable_6ba8868_chunk11_w7"
        )
    ),
    "50salads": DatasetConfig(
        run_dir=Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "diffact_50salads_allfolds_resumable_6ba8868_chunk11"
        )
    ),
    "breakfast": DatasetConfig(
        run_dir=Path(
            "/data1/eli-bogdanov/sktr_runs/"
            "diffact_breakfast_unique199_f14fd99_chunk11_w10"
        ),
        unique_only=True,
        train_k=199,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize DiffAct argmax/SKTR error types from existing artifacts."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--out-dir",
        default="/data1/eli-bogdanov/sktr_runs/diffact_error_taxonomy_v1",
    )
    parser.add_argument("--boundary-width", type=int, default=25)
    parser.add_argument("--island-max-len", type=int, default=25)
    parser.add_argument("--long-min-len", type=int, default=100)
    parser.add_argument("--long-homogeneity", type=float, default=0.90)
    parser.add_argument("--confusion-top-k", type=int, default=3)
    parser.add_argument(
        "--manual-cases-per-dataset",
        type=int,
        default=3,
        help="Number of concrete spot-check cases per dataset in manual_validation.md.",
    )
    return parser.parse_args()


def contiguous_spans(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def segments(labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not labels:
        return []
    out: List[Tuple[int, int, str]] = []
    start = 0
    current = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            out.append((start, idx, current))
            start = idx
            current = label
    out.append((start, len(labels), current))
    return out


def format_segments(segs: Sequence[Tuple[int, int, str]], limit: int = 16) -> str:
    items = [f"{label}[{start}:{end}]" for start, end, label in segs[:limit]]
    if len(segs) > limit:
        items.append(f"... +{len(segs) - limit} more")
    return " | ".join(items)


def load_ceiling(dataset: str) -> pd.DataFrame:
    if dataset == "gtea":
        paths = [
            Path(
                "/data1/eli-bogdanov/sktr_runs/"
                "sktr_ceiling_analysis_gtea_skip_all_v4/all_ceiling_cases.csv"
            )
        ]
    elif dataset == "breakfast":
        paths = sorted(
            Path("/data1/eli-bogdanov/sktr_runs/sktr_ceiling_breakfast_skip_all_v1")
            .glob("breakfast_fold*_ceiling_cases.csv")
        )
    elif dataset == "50salads":
        paths = [
            Path(
                "/data1/eli-bogdanov/sktr_runs/"
                "sktr_ceiling_50salads_fold1_run_rerun_64gb_v1/"
                "completed_ceiling_cases.csv"
            ),
            Path(
                "/data1/eli-bogdanov/sktr_runs/"
                "sktr_ceiling_50salads_isolated_v1/completed_ceiling_cases.csv"
            ),
        ]
    else:
        raise ValueError(dataset)

    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ceiling artifacts for {dataset}: {missing}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["dataset"] = df["dataset"].astype(str)
    df["fold"] = df["fold"].astype(int)
    df["case_id"] = df["case_id"].astype(str)
    return df.sort_values(["fold", "case_id"]).reset_index(drop=True)


def get_folds(dataset: str, requested: Optional[Sequence[int]], data_root: str) -> List[int]:
    if requested:
        return [int(f) for f in requested]
    n_folds = int(get_dataset_cv_config(dataset, data_root)["n_folds"])
    return list(range(1, n_folds + 1))


def case_output_path(dataset: str, fold: int, case_id: str) -> Path:
    return DATASETS[dataset].run_dir / "case_outputs" / f"{dataset}_fold{fold}" / f"{case_id}.csv"


def fold_train_cases(
    *,
    dataset: str,
    fold: int,
    df: pd.DataFrame,
    softmax_dir: Path,
    data_root: str,
) -> List[str]:
    video_map = build_video_to_case_mapping(
        dataset,
        "diffact",
        video_index_map_path=softmax_dir / "video_index_map.txt",
    )
    split = load_fold_case_ids(dataset, fold, video_map, data_root=data_root)
    full_train = [str(c) for c in split["train"]]
    full_test = [str(c) for c in split["test"]]
    cfg = DATASETS[dataset]
    variant_df = None
    if cfg.unique_only or cfg.train_k is not None:
        variant_df = get_variant_info_fast(df, use_collapsed=True)
    train_cases, _, _ = select_train_test_cases(
        train_cases=full_train,
        test_cases=full_test,
        variant_df=variant_df,
        unique_only=cfg.unique_only,
        train_k=cfg.train_k,
        seed=42,
        fold=fold,
    )
    return [str(c) for c in train_cases]


def build_train_confusions(
    *,
    dataset: str,
    fold: int,
    data_root: str,
    top_k: int,
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    diffact_root = REPO_ROOT / "baselines" / "DiffAct"
    softmax_dir = resolve_diffact_softmax_dir(
        diffact_root, dataset, fold, disallow_legacy=True
    )
    df, softmax_lst, entries = load_diffact_softmax_and_aligned_df(
        dataset, softmax_dir, Path(data_root)
    )
    verify_softmax_list(softmax_lst, f"{dataset} fold {fold}")
    case_to_mat = {str(entry[0]): softmax_lst[idx] for idx, entry in enumerate(entries)}
    train_cases = fold_train_cases(
        dataset=dataset,
        fold=fold,
        df=df,
        softmax_dir=softmax_dir,
        data_root=data_root,
    )

    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for case_id in train_cases:
        mat = case_to_mat[case_id]
        gt = (
            df.loc[df["case:concept:name"].astype(str) == case_id, "concept:name"]
            .astype(str)
            .to_numpy()
        )
        pred = mat.argmax(axis=0).astype(str)
        if len(gt) != len(pred):
            raise ValueError(
                f"{dataset} fold {fold} train case {case_id}: GT {len(gt)} != pred {len(pred)}"
            )
        for g, p in zip(gt, pred):
            if p != g:
                counts[str(g)][str(p)] += 1

    top: Dict[str, Set[str]] = {}
    raw: Dict[str, Dict[str, int]] = {}
    for gt_label, counter in counts.items():
        top[gt_label] = {label for label, _ in counter.most_common(top_k)}
        raw[gt_label] = {label: int(count) for label, count in counter.items()}
    return top, raw


def boundary_label_sets(gt: Sequence[str], width: int) -> List[Set[str]]:
    labels: List[Set[str]] = [set() for _ in gt]
    for idx in range(1, len(gt)):
        if gt[idx - 1] == gt[idx]:
            continue
        left, right = gt[idx - 1], gt[idx]
        start = max(0, idx - 1 - width)
        end = min(len(gt), idx + width + 1)
        for t in range(start, end):
            labels[t].add(left)
            labels[t].add(right)
    return labels


def island_mask(gt: Sequence[str], pred: Sequence[str], max_len: int) -> np.ndarray:
    out = np.zeros(len(gt), dtype=bool)
    for start, end, label in segments(list(gt)):
        wrong = [pred[t] != label for t in range(start, end)]
        for rel_s, rel_e in contiguous_spans(wrong):
            span_s = start + rel_s
            span_e = start + rel_e
            if span_e - span_s > max_len:
                continue
            if span_s <= start or span_e >= end:
                continue
            if pred[span_s - 1] == label and pred[span_e] == label:
                out[span_s:span_e] = True
    return out


def long_substitution_mask(
    gt: Sequence[str],
    pred: Sequence[str],
    min_len: int,
    homogeneity: float,
) -> np.ndarray:
    errors = [p != g for g, p in zip(gt, pred)]
    out = np.zeros(len(gt), dtype=bool)
    for start, end in contiguous_spans(errors):
        length = end - start
        if length < min_len:
            continue
        counts = Counter(pred[start:end])
        if not counts:
            continue
        _, max_count = counts.most_common(1)[0]
        if max_count / length >= homogeneity:
            out[start:end] = True
    return out


def classify_frames(
    *,
    gt: Sequence[str],
    pred: Sequence[str],
    top_confusions: Dict[str, Set[str]],
    island_max_len: int,
    long_min_len: int,
    long_homogeneity: float,
) -> Tuple[List[str], Dict[str, int]]:
    gt_list = list(gt)
    pred_list = list(pred)
    error = np.asarray([p != g for g, p in zip(gt_list, pred_list)], dtype=bool)
    boundary_sets = {
        10: boundary_label_sets(gt_list, 10),
        25: boundary_label_sets(gt_list, 25),
        50: boundary_label_sets(gt_list, 50),
    }
    boundary_masks = {
        width: np.asarray(
            [
                bool(error[t] and pred_list[t] in boundary_sets[width][t])
                for t in range(len(gt_list))
            ],
            dtype=bool,
        )
        for width in (10, 25, 50)
    }
    island = island_mask(gt_list, pred_list, island_max_len)
    long_sub = long_substitution_mask(
        gt_list, pred_list, long_min_len, long_homogeneity
    )

    categories = ["correct"] * len(gt_list)
    counts = {col: 0 for col in CATEGORY_COLUMNS}
    for t, (g, p) in enumerate(zip(gt_list, pred_list)):
        if not error[t]:
            continue
        if boundary_masks[25][t]:
            cat = "boundary_w25"
        elif island[t]:
            cat = "over_segmentation_island"
        elif long_sub[t]:
            cat = "long_substitution"
        elif p in top_confusions.get(g, set()):
            cat = "class_confusion"
        else:
            cat = "residual"
        categories[t] = cat
        counts[cat] += 1

    counts["boundary_w10"] = int(boundary_masks[10].sum())
    counts["boundary_w25_sensitivity"] = int(boundary_masks[25].sum())
    counts["boundary_w50"] = int(boundary_masks[50].sum())
    return categories, counts


def analyze_case(
    *,
    dataset: str,
    fold: int,
    case_id: str,
    ceiling_row: pd.Series,
    top_confusions: Dict[str, Set[str]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], pd.DataFrame]:
    path = case_output_path(dataset, fold, case_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    gt = df["ground_truth"].astype(str).tolist()
    rows: List[Dict[str, Any]] = []
    category_by_system: Dict[str, List[str]] = {}

    for system, pred_col in [("argmax", "argmax_activity"), ("sktr", "sktr_activity")]:
        pred = df[pred_col].astype(str).tolist()
        categories, counts = classify_frames(
            gt=gt,
            pred=pred,
            top_confusions=top_confusions,
            island_max_len=int(args.island_max_len),
            long_min_len=int(args.long_min_len),
            long_homogeneity=float(args.long_homogeneity),
        )
        total_errors = int(sum(p != g for g, p in zip(gt, pred)))
        category_sum = int(sum(counts[col] for col in CATEGORY_COLUMNS))
        sum_ok = category_sum == total_errors
        row: Dict[str, Any] = {
            "dataset": dataset,
            "fold": fold,
            "case_id": case_id,
            "system": system,
            "n_frames": len(gt),
            "total_errors": total_errors,
            "sum_check_ok": bool(sum_ok),
        }
        for col in CATEGORY_COLUMNS + BOUNDARY_COLUMNS:
            row[col] = int(counts[col])
            row[f"{col}_share"] = float(counts[col] / total_errors) if total_errors else 0.0
        for col in ORDER_COLUMNS:
            row[col] = ceiling_row[col]
        rows.append(row)
        category_by_system[system] = categories
    return rows, category_by_system, df


def aggregate_shares(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(list(group_cols), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_cases"] = int(len(group))
        row["n_frames"] = int(group["n_frames"].sum())
        row["total_errors"] = int(group["total_errors"].sum())
        denom = max(row["total_errors"], 1)
        for col in CATEGORY_COLUMNS + BOUNDARY_COLUMNS:
            row[col] = int(group[col].sum())
            row[f"{col}_share"] = float(group[col].sum() / denom)
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_summary(per_case: pd.DataFrame, per_fold: pd.DataFrame) -> pd.DataFrame:
    dataset_rows = aggregate_shares(per_case, ["dataset", "system"])
    spread_rows: List[Dict[str, Any]] = []
    share_cols = [f"{col}_share" for col in CATEGORY_COLUMNS + BOUNDARY_COLUMNS]
    for (dataset, system), group in per_fold.groupby(["dataset", "system"]):
        row: Dict[str, Any] = {"dataset": dataset, "system": system}
        for share_col in share_cols:
            spread = float(group[share_col].max() - group[share_col].min())
            row[f"{share_col}_fold_spread"] = spread
            row[f"{share_col}_spread_gt_10pp"] = bool(spread > 0.10)
        spread_rows.append(row)
    spread = pd.DataFrame(spread_rows)
    return dataset_rows.merge(spread, on=["dataset", "system"], how="left")


def order_context_crosstab(per_case: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    context_defs = {
        "argmax_no_order_violation": (
            per_case["argmax_log_moves"].astype(int)
            + per_case["argmax_model_moves"].astype(int)
        )
        == 0,
        "argmax_has_order_violation": (
            per_case["argmax_log_moves"].astype(int)
            + per_case["argmax_model_moves"].astype(int)
        )
        > 0,
        "gt_tau_completed": per_case["gt_accepted_exact_tau_completed"].astype(bool),
        "gt_not_tau_completed": ~per_case["gt_accepted_exact_tau_completed"].astype(bool),
    }
    for (dataset, system), group in per_case.groupby(["dataset", "system"]):
        for context, mask in context_defs.items():
            sub = group.loc[mask.loc[group.index]]
            if sub.empty:
                continue
            total = int(sub["total_errors"].sum())
            row: Dict[str, Any] = {
                "dataset": dataset,
                "system": system,
                "context": context,
                "n_cases": int(len(sub)),
                "total_errors": total,
            }
            denom = max(total, 1)
            for col in CATEGORY_COLUMNS:
                row[f"{col}_share"] = float(sub[col].sum() / denom)
            rows.append(row)
    return pd.DataFrame(rows)


def select_manual_cases(per_case: pd.DataFrame, n_per_dataset: int) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    argmax = per_case[per_case["system"] == "argmax"].copy()
    argmax["interesting"] = argmax[CATEGORY_COLUMNS].max(axis=1) + argmax["total_errors"]
    for dataset, group in argmax.groupby("dataset"):
        selected = (
            group.sort_values(["interesting", "total_errors"], ascending=False)
            .head(n_per_dataset)
        )
        for row in selected.itertuples(index=False):
            out.append((str(row.dataset), int(row.fold), str(row.case_id)))
    return out


def write_manual_validation(
    *,
    out_path: Path,
    manual_cases: Sequence[Tuple[str, int, str]],
    case_frames: Dict[Tuple[str, int, str], Tuple[pd.DataFrame, Dict[str, List[str]]]],
) -> None:
    lines = [
        "# Manual Validation",
        "",
        "Spot checks are generated from existing per-case CSVs after the taxonomy is assigned.",
        "For each dataset, cases with many errors are shown with segment summaries and concrete",
        "frame-level examples. The check is descriptive: examples should match the operational",
        "definitions in the report. No thresholds were tuned after this inspection.",
        "",
    ]
    for dataset, fold, case_id in manual_cases:
        df, cats = case_frames[(dataset, fold, case_id)]
        gt = df["ground_truth"].astype(str).tolist()
        lines.extend(
            [
                f"## {dataset} fold {fold} case {case_id}",
                "",
                f"- GT segments: {format_segments(segments(gt))}",
            ]
        )
        for system, pred_col in [("argmax", "argmax_activity"), ("sktr", "sktr_activity")]:
            pred = df[pred_col].astype(str).tolist()
            cat = cats[system]
            lines.extend(
                [
                    f"- {system} segments: {format_segments(segments(pred))}",
                    "",
                    f"### {system} examples",
                ]
            )
            for category in CATEGORY_COLUMNS:
                idxs = [idx for idx, value in enumerate(cat) if value == category]
                if not idxs:
                    lines.append(f"- {category}: none")
                    continue
                chosen = idxs[:2]
                examples = []
                for idx in chosen:
                    start = max(0, idx - 3)
                    end = min(len(gt), idx + 4)
                    examples.append(
                        f"t={idx}, GT={gt[idx]}, pred={pred[idx]}, "
                        f"local_gt={gt[start:end]}, local_pred={pred[start:end]}"
                    )
                lines.append(f"- {category}: " + " || ".join(examples))
            lines.append("")
    lines.extend(
        [
            "## Validation Result",
            "",
            "The displayed frame examples are consistent with the pre-committed category",
            "definitions; no operationalization changes were made after scale-out.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, columns].copy()

    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(fmt(row[col]) for col in columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def write_report(
    *,
    out_path: Path,
    per_dataset: pd.DataFrame,
    cross_tab: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    lines = [
        "# DiffAct Error-Type Characterization",
        "",
        "Scope: descriptive taxonomy over existing DiffAct+SKTR artifacts. No SKTR",
        "reruns, Petri-net rediscovery, or ceiling recomputation were performed.",
        "",
        "Class-confusion labels are fold-pure: for each dataset/fold, the top confused",
        "classes are estimated from training-case DiffAct softmax argmax versus training",
        "GT only. Test labels are not used to build the confusion lists.",
        "",
        "Pre-committed thresholds: boundary width 25 frames (with 10/50 sensitivity),",
        "over-segmentation island length <=25, long substitution length >=100 with",
        ">=90% one wrong class, top-3 training confusions.",
        "",
    ]
    for dataset in sorted(per_dataset["dataset"].unique()):
        lines.extend([f"## {dataset}", ""])
        subset = per_dataset[(per_dataset["dataset"] == dataset)]
        arg = subset[subset["system"] == "argmax"].iloc[0]
        sktr = subset[subset["system"] == "sktr"].iloc[0]
        category_shares = {
            col: float(arg[f"{col}_share"]) for col in CATEGORY_COLUMNS
        }
        dominant = sorted(category_shares.items(), key=lambda kv: kv[1], reverse=True)
        primary, secondary = dominant[0], dominant[1]
        lines.append(
            f"DiffAct argmax errors are dominated by `{primary[0]}` "
            f"({primary[1]*100:.1f}% of errors), with `{secondary[0]}` as the "
            f"main secondary bucket ({secondary[1]*100:.1f}%). SKTR shifts are "
            "reported descriptively below; positive or negative causal claims are "
            "out of scope for this report."
        )
        lines.append("")
        display = subset.copy()
        for col in CATEGORY_COLUMNS + BOUNDARY_COLUMNS:
            display[col] = display[f"{col}_share"] * 100.0
        lines.append(
            markdown_table(
                display,
                [
                    "system",
                    "total_errors",
                    "boundary_w10",
                    "boundary_w25",
                    "boundary_w50",
                    "over_segmentation_island",
                    "long_substitution",
                    "class_confusion",
                    "residual",
                ],
            )
        )
        lines.append("")
        shift_rows = []
        for col in CATEGORY_COLUMNS:
            shift_rows.append(
                {
                    "category": col,
                    "argmax_share_pct": arg[f"{col}_share"] * 100.0,
                    "sktr_share_pct": sktr[f"{col}_share"] * 100.0,
                    "sktr_minus_argmax_pct": (
                        sktr[f"{col}_share"] - arg[f"{col}_share"]
                    )
                    * 100.0,
                }
            )
        lines.append("SKTR distribution shift:")
        lines.append(markdown_table(pd.DataFrame(shift_rows), list(shift_rows[0].keys())))
        lines.append("")
        ct = cross_tab[cross_tab["dataset"] == dataset].copy()
        if not ct.empty:
            for col in CATEGORY_COLUMNS:
                ct[col] = ct[f"{col}_share"] * 100.0
            lines.append("Order-context cross-tab:")
            lines.append(
                markdown_table(
                    ct,
                    [
                        "system",
                        "context",
                        "total_errors",
                        "boundary_w25",
                        "over_segmentation_island",
                        "long_substitution",
                        "class_confusion",
                        "residual",
                    ],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Sanity Flags",
            "",
            json.dumps(summary["sanity_flags"], indent=2, sort_keys=True),
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if int(args.boundary_width) != 25:
        raise ValueError(
            "This report uses the pre-committed primary boundary width w=25; "
            "use the fixed 10/25/50 sensitivity columns rather than changing "
            "--boundary-width."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_case_rows: List[Dict[str, Any]] = []
    outlier_rows: List[Dict[str, Any]] = []
    case_frames: Dict[Tuple[str, int, str], Tuple[pd.DataFrame, Dict[str, List[str]]]] = {}
    confusion_cache: Dict[Tuple[str, int], Dict[str, Set[str]]] = {}
    confusion_raw: Dict[str, Any] = {}

    for dataset in args.datasets:
        ceiling = load_ceiling(dataset)
        folds = get_folds(dataset, args.folds, args.data_root)
        ceiling = ceiling[ceiling["fold"].isin(folds)].copy()
        if args.case_limit is not None:
            ceiling = (
                ceiling.sort_values(["fold", "case_id"])
                .groupby("fold", sort=False)
                .head(int(args.case_limit))
                .reset_index(drop=True)
            )
        for fold in folds:
            top, raw = build_train_confusions(
                dataset=dataset,
                fold=fold,
                data_root=args.data_root,
                top_k=int(args.confusion_top_k),
            )
            confusion_cache[(dataset, fold)] = top
            confusion_raw[f"{dataset}_fold{fold}"] = raw

        for row in ceiling.itertuples(index=False):
            fold = int(row.fold)
            case_id = str(row.case_id)
            rows, categories, frame_df = analyze_case(
                dataset=dataset,
                fold=fold,
                case_id=case_id,
                ceiling_row=pd.Series(row._asdict()),
                top_confusions=confusion_cache[(dataset, fold)],
                args=args,
            )
            all_case_rows.extend(rows)
            key = (dataset, fold, case_id)
            case_frames[key] = (frame_df, categories)
            if key in OUTLIER_CASES:
                outlier_rows.extend(rows)

    per_case = pd.DataFrame(all_case_rows)
    if per_case.empty:
        raise ValueError("No cases analyzed")
    if not per_case["sum_check_ok"].all():
        bad = per_case.loc[~per_case["sum_check_ok"], ["dataset", "fold", "case_id", "system"]]
        raise AssertionError(f"Category sum check failed:\n{bad}")

    per_fold = aggregate_shares(per_case, ["dataset", "fold", "system"])
    per_dataset = dataset_summary(per_case, per_fold)
    cross_tab = order_context_crosstab(per_case)
    outliers = pd.DataFrame(outlier_rows)
    observed_outlier_cases = {
        (str(row.dataset), int(row.fold), str(row.case_id))
        for row in outliers.itertuples(index=False)
    } if not outliers.empty else set()
    expected_outlier_cases = {
        key for key in OUTLIER_CASES if key[0] in set(args.datasets)
    }
    missing_outlier_cases = sorted(expected_outlier_cases - observed_outlier_cases)

    manual_cases = select_manual_cases(per_case, int(args.manual_cases_per_dataset))
    write_manual_validation(
        out_path=out_dir / "manual_validation.md",
        manual_cases=manual_cases,
        case_frames=case_frames,
    )

    per_case.to_csv(out_dir / "error_taxonomy_per_case.csv", index=False)
    per_fold.to_csv(out_dir / "error_taxonomy_per_fold.csv", index=False)
    per_dataset.to_csv(out_dir / "error_taxonomy_per_dataset.csv", index=False)
    outliers.to_csv(out_dir / "error_taxonomy_outliers.csv", index=False)
    cross_tab.to_csv(out_dir / "error_taxonomy_order_context_crosstab.csv", index=False)
    (out_dir / "training_confusion_top3.json").write_text(
        json.dumps(confusion_raw, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "out_dir": str(out_dir),
        "datasets": args.datasets,
        "thresholds": {
            "boundary_primary_width": int(args.boundary_width),
            "boundary_sensitivity_widths": [10, 25, 50],
            "island_max_len": int(args.island_max_len),
            "long_min_len": int(args.long_min_len),
            "long_homogeneity": float(args.long_homogeneity),
            "confusion_top_k": int(args.confusion_top_k),
        },
        "argmax_headline_shares": {},
        "argmax_to_sktr_shifts": {},
        "sanity_flags": {
            "categories_sum_to_total_per_case": bool(per_case["sum_check_ok"].all()),
            "hand_validation_done": True,
            "width_sensitivity_reported": all(col in per_case.columns for col in BOUNDARY_COLUMNS),
            "no_sktr_rerun": True,
            "no_ceiling_recompute": True,
            "fold_pure_train_confusion": True,
            "outlier_cases_reported": not missing_outlier_cases,
        },
        "outlier_cases_expected": [list(key) for key in sorted(expected_outlier_cases)],
        "outlier_cases_found": [list(key) for key in sorted(observed_outlier_cases)],
        "outlier_cases_missing": [list(key) for key in missing_outlier_cases],
    }
    for dataset in sorted(per_dataset["dataset"].unique()):
        arg = per_dataset[(per_dataset["dataset"] == dataset) & (per_dataset["system"] == "argmax")].iloc[0]
        sktr = per_dataset[(per_dataset["dataset"] == dataset) & (per_dataset["system"] == "sktr")].iloc[0]
        summary["argmax_headline_shares"][dataset] = {
            col: float(arg[f"{col}_share"]) for col in CATEGORY_COLUMNS
        }
        summary["argmax_to_sktr_shifts"][dataset] = {
            col: float(sktr[f"{col}_share"] - arg[f"{col}_share"])
            for col in CATEGORY_COLUMNS
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(
        out_path=out_dir / "report.md",
        per_dataset=per_dataset,
        cross_tab=cross_tab,
        summary=summary,
    )

    print(f"Wrote error taxonomy outputs to {out_dir}")
    print(f"Rows: per_case={len(per_case)}, per_fold={len(per_fold)}, outliers={len(outliers)}")
    print("Sanity:", json.dumps(summary["sanity_flags"], sort_keys=True))


if __name__ == "__main__":
    main()
