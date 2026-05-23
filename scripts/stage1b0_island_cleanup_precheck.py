#!/usr/bin/env python3
"""
Stage-1B0: short-island cleanup pre-check, oracle-ceiling first.

This is a read-only diagnostic over existing DiffAct/SKTR artifacts. Test GT is
used only for oracle/evaluation quantities. Deployable rules, when evaluated,
use only argmax segment structure and DiffAct softmax margins.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage0_duration_confidence_diagnostics import (  # noqa: E402
    DATASETS,
    FoldContext,
    get_folds,
    load_case_output,
    load_complete_case_set,
    load_fold_context,
    segments,
)
from src.cv_utils import DEFAULT_DATA_ROOT  # noqa: E402
from src.evaluation import compute_tas_metrics_from_sequences, tas_metrics  # noqa: E402


DEFAULT_OUT_DIR = "/data1/eli-bogdanov/sktr_runs/stage1b0_island_cleanup_precheck_v1"
L_ABS_VALUES = [5, 25]
RULE_MARGIN_THRESHOLD = 0.10
GATE_THRESHOLD = 0.25
OUTLIER_CASES = {("50salads", 1, "1"), ("50salads", 5, "49")}
METRIC_KEYS = ["acc", "edit", "f1@10", "f1@25", "f1@50"]


@dataclass(frozen=True)
class Island:
    seg_index: int
    start: int
    end: int
    label: str
    left_label: str
    right_label: str
    length: int
    mean_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--l-abs", nargs="+", type=int, default=L_ABS_VALUES)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Convenience mode: --datasets gtea --folds 1.",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def softmax_margin(mat: np.ndarray) -> np.ndarray:
    probs = np.asarray(mat, dtype=float)
    top1 = probs.max(axis=0)
    if probs.shape[0] > 1:
        top2 = np.partition(probs, -2, axis=0)[-2]
    else:
        top2 = np.zeros(probs.shape[1])
    return top1 - top2


def frame_accuracy_percent(gt: Sequence[str], pred: Sequence[str]) -> float:
    if not gt:
        return 0.0
    return 100.0 * float(np.mean(np.asarray(gt, dtype=object) == np.asarray(pred, dtype=object)))


def candidate_islands(argmax: Sequence[str], margins: np.ndarray, l_abs: int) -> List[Island]:
    segs = segments([str(x) for x in argmax])
    out: List[Island] = []
    for idx in range(1, len(segs) - 1):
        start, end, label = segs[idx]
        length = end - start
        if length > l_abs:
            continue
        out.append(
            Island(
                seg_index=idx,
                start=start,
                end=end,
                label=str(label),
                left_label=str(segs[idx - 1][2]),
                right_label=str(segs[idx + 1][2]),
                length=length,
                mean_margin=float(np.mean(margins[start:end])) if end > start else float("nan"),
            )
        )
    return out


def action_labels(island: Island) -> List[Tuple[str, str]]:
    actions = [("keep", island.label)]
    actions.append(("merge_left", island.left_label))
    actions.append(("merge_right", island.right_label))
    if island.left_label == island.right_label:
        actions.append(("aba_delete", island.left_label))
    # Deduplicate labels while preserving named actions, preferring ABA over
    # directional merge for ABA cases because that is the intended cleanup.
    dedup: Dict[str, str] = {}
    for name, label in actions:
        dedup[label] = name
    if island.label in dedup:
        dedup[island.label] = "keep"
    return [(name, label) for label, name in dedup.items()]


def local_correct(gt: Sequence[str], start: int, end: int, label: str) -> int:
    return sum(1 for g in gt[start:end] if str(g) == str(label))


def apply_island_label(pred: List[str], island: Island, label: str) -> None:
    pred[island.start : island.end] = [str(label)] * island.length


def accuracy_oracle_prediction(
    gt: Sequence[str],
    argmax: Sequence[str],
    islands: Sequence[Island],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    pred = [str(x) for x in argmax]
    edit_rows: List[Dict[str, Any]] = []
    for island in islands:
        choices = []
        before = local_correct(gt, island.start, island.end, island.label)
        for action, label in action_labels(island):
            after = local_correct(gt, island.start, island.end, label)
            choices.append((after, action == "keep", action, label))
        # Maximize local accuracy; keep wins ties.
        after, _, action, label = max(choices, key=lambda x: (x[0], x[1]))
        apply_island_label(pred, island, label)
        edit_rows.append(
            {
                "seg_index": island.seg_index,
                "start": island.start,
                "end": island.end,
                "label": island.label,
                "left_label": island.left_label,
                "right_label": island.right_label,
                "length": island.length,
                "mean_margin": island.mean_margin,
                "oracle_action_accuracy": action,
                "oracle_label_accuracy": label,
                "local_correct_before": before,
                "local_correct_after_accuracy": after,
            }
        )
    return pred, edit_rows


def segmental_score(metrics: Dict[str, float]) -> float:
    return float(
        (metrics["edit"] + metrics["f1@10"] + metrics["f1@25"] + metrics["f1@50"]) / 4.0
    )


def edit_f1_greedy_oracle_prediction(
    gt: Sequence[str],
    argmax: Sequence[str],
    islands: Sequence[Island],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    pred = [str(x) for x in argmax]
    edit_rows: List[Dict[str, Any]] = []
    for island in islands:
        current_metrics = tas_metrics(gt, pred)
        current_score = segmental_score(current_metrics)
        best = (current_score, current_metrics["acc"], True, "keep", island.label, pred)
        for action, label in action_labels(island):
            trial = pred.copy()
            apply_island_label(trial, island, label)
            metrics = tas_metrics(gt, trial)
            cand = (
                segmental_score(metrics),
                metrics["acc"],
                action == "keep",
                action,
                label,
                trial,
            )
            if cand[:3] > best[:3]:
                best = cand
        _, _, _, action, label, pred = best
        edit_rows.append(
            {
                "seg_index": island.seg_index,
                "start": island.start,
                "end": island.end,
                "label": island.label,
                "left_label": island.left_label,
                "right_label": island.right_label,
                "length": island.length,
                "mean_margin": island.mean_margin,
                "oracle_action_segmental": action,
                "oracle_label_segmental": label,
            }
        )
    return pred, edit_rows


def deployable_rule_prediction(
    gt: Sequence[str],
    argmax: Sequence[str],
    islands: Sequence[Island],
    rule: str,
) -> Tuple[List[str], Dict[str, Any]]:
    pred = [str(x) for x in argmax]
    edited = 0
    improved_or_neutral = 0
    improved = 0
    harmed = 0
    for island in islands:
        should_edit = island.left_label == island.right_label
        if rule == "aba_low_confidence":
            should_edit = should_edit and island.mean_margin <= RULE_MARGIN_THRESHOLD
        elif rule != "aba_only":
            raise ValueError(rule)
        if not should_edit:
            continue
        before = local_correct(gt, island.start, island.end, island.label)
        after = local_correct(gt, island.start, island.end, island.left_label)
        edited += 1
        improved_or_neutral += int(after >= before)
        improved += int(after > before)
        harmed += int(after < before)
        apply_island_label(pred, island, island.left_label)
    return pred, {
        "edited_islands": edited,
        "edited_improved_or_neutral": improved_or_neutral,
        "edited_improved": improved,
        "edited_harmed": harmed,
    }


def metrics_delta(prefix: str, base: Dict[str, float], pred: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for key in METRIC_KEYS:
        safe = key.replace("@", "_")
        out[f"{prefix}_{safe}"] = float(pred[key])
        out[f"delta_{prefix}_{safe}"] = float(pred[key] - base[key])
    return out


def aggregate_metrics(rows: List[Dict[str, Any]], pred_col: str) -> Dict[str, float]:
    gt_seqs = [row["gt"] for row in rows]
    pred_seqs = [row[pred_col] for row in rows]
    return compute_tas_metrics_from_sequences(gt_seqs, pred_seqs)


def aggregate_case_group(
    *,
    dataset: str,
    fold: Any,
    l_abs: int,
    target: str,
    rows: List[Dict[str, Any]],
    pred_col: str,
) -> Dict[str, Any]:
    arg_m = aggregate_metrics(rows, "argmax")
    sktr_m = aggregate_metrics(rows, "sktr")
    pred_m = aggregate_metrics(rows, pred_col)
    n_frames = sum(len(row["gt"]) for row in rows)
    cand_count = sum(int(row[f"candidate_count_L{l_abs}"]) for row in rows)
    cand_frames = sum(int(row[f"candidate_frames_L{l_abs}"]) for row in rows)
    out = {
        "dataset": dataset,
        "fold": fold,
        "L_abs": int(l_abs),
        "oracle_target": target,
        "n_cases": int(len(rows)),
        "n_frames": int(n_frames),
        "candidate_count": int(cand_count),
        "candidate_frame_count": int(cand_frames),
        "candidate_frame_fraction": safe_div(cand_frames, n_frames),
        "argmax_acc": arg_m["acc"],
        "argmax_edit": arg_m["edit"],
        "argmax_f1_10": arg_m["f1@10"],
        "argmax_f1_25": arg_m["f1@25"],
        "argmax_f1_50": arg_m["f1@50"],
        "sktr_acc": sktr_m["acc"],
        "sktr_edit": sktr_m["edit"],
        "sktr_f1_10": sktr_m["f1@10"],
        "sktr_f1_25": sktr_m["f1@25"],
        "sktr_f1_50": sktr_m["f1@50"],
        "oracle_acc": pred_m["acc"],
        "oracle_edit": pred_m["edit"],
        "oracle_f1_10": pred_m["f1@10"],
        "oracle_f1_25": pred_m["f1@25"],
        "oracle_f1_50": pred_m["f1@50"],
        "delta_acc": pred_m["acc"] - arg_m["acc"],
        "delta_edit": pred_m["edit"] - arg_m["edit"],
        "delta_f1_10": pred_m["f1@10"] - arg_m["f1@10"],
        "delta_f1_25": pred_m["f1@25"] - arg_m["f1@25"],
        "delta_f1_50": pred_m["f1@50"] - arg_m["f1@50"],
        "oracle_uses_test_gt": True,
    }
    return out


def case_record(
    *,
    ctx: FoldContext,
    case_id: str,
    l_abs: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    case_df = load_case_output(ctx.dataset, ctx.fold, case_id)
    gt = case_df["ground_truth"].astype(str).tolist()
    argmax = case_df["argmax_activity"].astype(str).tolist()
    sktr = case_df["sktr_activity"].astype(str).tolist()
    mat = ctx.case_to_mat[str(case_id)]
    if not (len(gt) == len(argmax) == len(sktr) == mat.shape[1]):
        raise ValueError(
            f"Length mismatch {ctx.dataset} fold {ctx.fold} case {case_id}: "
            f"gt={len(gt)} argmax={len(argmax)} sktr={len(sktr)} softmax_T={mat.shape[1]}"
        )
    raw_argmax = mat.argmax(axis=0).astype(str).tolist()
    if raw_argmax != argmax:
        bad = [i for i, (a, b) in enumerate(zip(raw_argmax, argmax)) if a != b][:10]
        raise ValueError(
            f"Raw softmax argmax mismatch {ctx.dataset} fold {ctx.fold} case {case_id}: {bad}"
        )
    margins = softmax_margin(mat)
    islands = candidate_islands(argmax, margins, l_abs)
    acc_pred, acc_edits = accuracy_oracle_prediction(gt, argmax, islands)
    seg_pred, seg_edits = edit_f1_greedy_oracle_prediction(gt, argmax, islands)
    rule_a_pred, rule_a_info = deployable_rule_prediction(gt, argmax, islands, "aba_only")
    rule_b_pred, rule_b_info = deployable_rule_prediction(
        gt, argmax, islands, "aba_low_confidence"
    )

    base_m = tas_metrics(gt, argmax)
    sktr_m = tas_metrics(gt, sktr)
    acc_m = tas_metrics(gt, acc_pred)
    seg_m = tas_metrics(gt, seg_pred)
    rule_a_m = tas_metrics(gt, rule_a_pred)
    rule_b_m = tas_metrics(gt, rule_b_pred)
    cand_frames = sum(island.length for island in islands)
    rec: Dict[str, Any] = {
        "dataset": ctx.dataset,
        "fold": int(ctx.fold),
        "case_id": str(case_id),
        "L_abs": int(l_abs),
        "n_frames": int(len(gt)),
        "candidate_count": int(len(islands)),
        "candidate_frame_count": int(cand_frames),
        "candidate_frame_fraction": safe_div(cand_frames, len(gt)),
        "gt": gt,
        "argmax": argmax,
        "sktr": sktr,
        "oracle_accuracy": acc_pred,
        "oracle_segmental": seg_pred,
        "rule_aba_only": rule_a_pred,
        "rule_aba_low_confidence": rule_b_pred,
        f"candidate_count_L{l_abs}": int(len(islands)),
        f"candidate_frames_L{l_abs}": int(cand_frames),
        "argmax_acc": base_m["acc"],
        "argmax_edit": base_m["edit"],
        "argmax_f1_10": base_m["f1@10"],
        "argmax_f1_25": base_m["f1@25"],
        "argmax_f1_50": base_m["f1@50"],
        "sktr_acc": sktr_m["acc"],
        "sktr_edit": sktr_m["edit"],
        "sktr_f1_10": sktr_m["f1@10"],
        "sktr_f1_25": sktr_m["f1@25"],
        "sktr_f1_50": sktr_m["f1@50"],
        **metrics_delta("oracle_accuracy", base_m, acc_m),
        **metrics_delta("oracle_edit_f1_greedy", base_m, seg_m),
        **metrics_delta("rule_aba_only", base_m, rule_a_m),
        **metrics_delta("rule_aba_low_confidence", base_m, rule_b_m),
        "rule_aba_only_edited_islands": rule_a_info["edited_islands"],
        "rule_aba_only_edited_improved_or_neutral": rule_a_info[
            "edited_improved_or_neutral"
        ],
        "rule_aba_only_edited_improved": rule_a_info["edited_improved"],
        "rule_aba_only_edited_harmed": rule_a_info["edited_harmed"],
        "rule_aba_low_confidence_edited_islands": rule_b_info["edited_islands"],
        "rule_aba_low_confidence_edited_improved_or_neutral": rule_b_info[
            "edited_improved_or_neutral"
        ],
        "rule_aba_low_confidence_edited_improved": rule_b_info["edited_improved"],
        "rule_aba_low_confidence_edited_harmed": rule_b_info["edited_harmed"],
    }
    island_rows: List[Dict[str, Any]] = []
    seg_edits_by_key = {(r["start"], r["end"]): r for r in seg_edits}
    for row in acc_edits:
        seg_row = seg_edits_by_key.get((row["start"], row["end"]), {})
        island_rows.append(
            {
                "dataset": ctx.dataset,
                "fold": int(ctx.fold),
                "case_id": str(case_id),
                "L_abs": int(l_abs),
                **row,
                "oracle_action_segmental": seg_row.get("oracle_action_segmental", ""),
                "oracle_label_segmental": seg_row.get("oracle_label_segmental", ""),
            }
        )
    return rec, island_rows


def public_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"gt", "argmax", "sktr", "oracle_accuracy", "oracle_segmental", "rule_aba_only", "rule_aba_low_confidence"}}


def build_oracle_ceiling(case_rows: List[Dict[str, Any]], l_values: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for l_abs in l_values:
        subset = [r for r in case_rows if int(r["L_abs"]) == int(l_abs)]
        for dataset in sorted({r["dataset"] for r in subset}):
            ds_rows = [r for r in subset if r["dataset"] == dataset]
            for fold in sorted({int(r["fold"]) for r in ds_rows}):
                group = [r for r in ds_rows if int(r["fold"]) == fold]
                for target, pred_col in [
                    ("accuracy", "oracle_accuracy"),
                    ("edit_f1_greedy", "oracle_segmental"),
                ]:
                    rows.append(
                        aggregate_case_group(
                            dataset=dataset,
                            fold=fold,
                            l_abs=int(l_abs),
                            target=target,
                            rows=group,
                            pred_col=pred_col,
                        )
                    )
            for target, pred_col in [
                ("accuracy", "oracle_accuracy"),
                ("edit_f1_greedy", "oracle_segmental"),
            ]:
                rows.append(
                    aggregate_case_group(
                        dataset=dataset,
                        fold="all",
                        l_abs=int(l_abs),
                        target=target,
                        rows=ds_rows,
                        pred_col=pred_col,
                    )
                )
    return pd.DataFrame(rows)


def gate_datasets(ceiling_df: pd.DataFrame) -> Dict[str, bool]:
    gates: Dict[str, bool] = {}
    focus = ceiling_df[ceiling_df["fold"].astype(str) == "all"]
    for dataset, group in focus.groupby("dataset"):
        max_delta = max(
            float(group[col].max())
            for col in ["delta_acc", "delta_edit", "delta_f1_10", "delta_f1_25", "delta_f1_50"]
        )
        gates[str(dataset)] = bool(max_delta >= GATE_THRESHOLD)
    return gates


def build_rule_eval(
    case_rows: List[Dict[str, Any]],
    l_values: Sequence[int],
    gates: Dict[str, bool],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for l_abs in l_values:
        subset = [r for r in case_rows if int(r["L_abs"]) == int(l_abs)]
        for dataset in sorted({r["dataset"] for r in subset}):
            if not gates.get(dataset, False):
                continue
            ds_rows = [r for r in subset if r["dataset"] == dataset]
            for fold_value, group in [("all", ds_rows)] + [
                (fold, [r for r in ds_rows if int(r["fold"]) == fold])
                for fold in sorted({int(r["fold"]) for r in ds_rows})
            ]:
                for rule, pred_col in [
                    ("aba_only", "rule_aba_only"),
                    ("aba_low_confidence", "rule_aba_low_confidence"),
                ]:
                    arg_m = aggregate_metrics(group, "argmax")
                    rule_m = aggregate_metrics(group, pred_col)
                    edited = sum(int(r[f"rule_{rule}_edited_islands"]) for r in group)
                    ok = sum(int(r[f"rule_{rule}_edited_improved_or_neutral"]) for r in group)
                    case_deltas = [
                        float(r[f"delta_rule_{rule}_acc"]) for r in group
                    ]
                    rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold_value,
                            "L_abs": int(l_abs),
                            "rule": rule,
                            "n_cases": int(len(group)),
                            "edited_islands": int(edited),
                            "precision_improve_or_neutral": safe_div(ok, edited),
                            "mean_delta_acc": rule_m["acc"] - arg_m["acc"],
                            "mean_delta_edit": rule_m["edit"] - arg_m["edit"],
                            "mean_delta_f1_10": rule_m["f1@10"] - arg_m["f1@10"],
                            "mean_delta_f1_25": rule_m["f1@25"] - arg_m["f1@25"],
                            "mean_delta_f1_50": rule_m["f1@50"] - arg_m["f1@50"],
                            "worst_case_harm_acc": min(case_deltas) if case_deltas else 0.0,
                            "n_helped_cases_acc": int(sum(d > 1e-12 for d in case_deltas)),
                            "n_harmed_cases_acc": int(sum(d < -1e-12 for d in case_deltas)),
                            "gt_used_by_rule": False,
                            "gt_used_for_eval": True,
                        }
                    )
    columns = [
        "dataset",
        "fold",
        "L_abs",
        "rule",
        "n_cases",
        "edited_islands",
        "precision_improve_or_neutral",
        "mean_delta_acc",
        "mean_delta_edit",
        "mean_delta_f1_10",
        "mean_delta_f1_25",
        "mean_delta_f1_50",
        "worst_case_harm_acc",
        "n_helped_cases_acc",
        "n_harmed_cases_acc",
        "gt_used_by_rule",
        "gt_used_for_eval",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_smoke_report(
    out_dir: Path,
    case_rows_df: pd.DataFrame,
    island_df: pd.DataFrame,
    ceiling_df: pd.DataFrame,
) -> None:
    smoke_cases = case_rows_df[
        (case_rows_df["dataset"] == "gtea") & (case_rows_df["fold"] == 1)
    ]
    examples = island_df[
        (island_df["dataset"] == "gtea")
        & (island_df["fold"] == 1)
        & (island_df["L_abs"] == 25)
    ].head(10)
    lines = [
        "# Stage-1B0 Island Cleanup Smoke",
        "",
        "Scope: GTEA fold 1. GT is used only for oracle/evaluation quantities.",
        "",
        f"- cases: {smoke_cases['case_id'].nunique() if len(smoke_cases) else 0}",
        f"- L_abs values: {sorted(smoke_cases['L_abs'].unique().tolist()) if len(smoke_cases) else []}",
        f"- candidate islands at L=25: {int((smoke_cases[smoke_cases['L_abs'] == 25]['candidate_count']).sum()) if len(smoke_cases) else 0}",
        "",
        "## First Candidate Islands",
        "",
    ]
    if examples.empty:
        lines.append("- none")
    for row in examples.itertuples(index=False):
        lines.append(
            f"- case {row.case_id}, seg={row.seg_index}, frames=[{row.start}:{row.end}), "
            f"label={row.label}, left={row.left_label}, right={row.right_label}, "
            f"len={row.length}, margin={row.mean_margin:.3f}, "
            f"acc_oracle={row.oracle_action_accuracy}->{row.oracle_label_accuracy}, "
            f"segmental_oracle={row.oracle_action_segmental}->{row.oracle_label_segmental}, "
            f"local_correct={row.local_correct_before}->{row.local_correct_after_accuracy}."
        )
    lines.extend(["", "## Smoke Ceiling Rows", ""])
    smoke_summary = ceiling_df[
        (ceiling_df["dataset"] == "gtea") & (ceiling_df["fold"].astype(str) == "1")
    ].sort_values(["L_abs", "oracle_target"])
    for row in smoke_summary.itertuples(index=False):
        lines.append(
            f"- L={row.L_abs}, target={row.oracle_target}: candidates={row.candidate_count}, "
            f"frames={row.candidate_frame_fraction:.4f}, "
            f"dAcc={row.delta_acc:.3f}, dEdit={row.delta_edit:.3f}, "
            f"dF1@50={row.delta_f1_50:.3f}."
        )
    lines.extend(
        [
            "",
            "## Manual Sanity Notes",
            "",
            "Printed islands show the original short segment, both neighbors, mean softmax margin, and the oracle-selected replacement. Accuracy-oracle local-correct counts were inspected for keep/merge behavior; deployable rules are not evaluated unless a dataset passes the oracle gate.",
        ]
    )
    (out_dir / "island_smoke_report.md").write_text("\n".join(lines) + "\n")


def write_summary(
    out_dir: Path,
    ceiling_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    gates: Dict[str, bool],
) -> None:
    lines = [
        "# Stage-1B0 Island Cleanup Pre-check Summary",
        "",
        "This is an oracle-ceiling-first diagnostic. Short island candidates are internal argmax segments with length <= L_abs. Test GT is used only for oracle/evaluation quantities.",
        "",
        f"Gate threshold: max oracle improvement across Acc/Edit/F1 metrics must be >= {GATE_THRESHOLD:.2f} points for deployable rules to be evaluated.",
        "",
    ]
    focus = ceiling_df[ceiling_df["fold"].astype(str) == "all"]
    for dataset in sorted(focus["dataset"].unique()):
        lines.extend([f"## {dataset}", ""])
        ds = focus[focus["dataset"] == dataset].sort_values(["L_abs", "oracle_target"])
        for row in ds.itertuples(index=False):
            lines.append(
                f"- L={row.L_abs}, target={row.oracle_target}: "
                f"candidate frames={row.candidate_frame_fraction:.4f}, "
                f"dAcc={row.delta_acc:.3f}, dEdit={row.delta_edit:.3f}, "
                f"dF1@10={row.delta_f1_10:.3f}, dF1@25={row.delta_f1_25:.3f}, "
                f"dF1@50={row.delta_f1_50:.3f}, candidates={row.candidate_count}."
            )
        gate = gates.get(str(dataset), False)
        lines.append("")
        lines.append(
            "Gate decision: "
            + (
                "GO to deployable-rule evaluation; oracle island ceiling is non-trivial."
                if gate
                else "NO-GO; oracle island ceiling is below threshold, so no deployable cleanup rule can be expected to help materially."
            )
        )
        if gate and not rule_df.empty:
            ds_rules = rule_df[
                (rule_df["dataset"] == dataset) & (rule_df["fold"].astype(str) == "all")
            ].sort_values(["L_abs", "rule"])
            lines.append("")
            lines.append("Deployable rule rows:")
            for row in ds_rules.itertuples(index=False):
                clears = (
                    row.precision_improve_or_neutral >= 0.75
                    and row.worst_case_harm_acc >= -1.0
                    and (row.mean_delta_edit > 0.0 or row.mean_delta_f1_50 > 0.0)
                )
                lines.append(
                    f"- L={row.L_abs}, {row.rule}: precision={row.precision_improve_or_neutral:.3f}, "
                    f"worst harm={row.worst_case_harm_acc:.3f} pp, "
                    f"dAcc={row.mean_delta_acc:.3f}, dEdit={row.mean_delta_edit:.3f}, "
                    f"dF1@50={row.mean_delta_f1_50:.3f}, clears_rule_gate={clears}."
                )
        lines.append("")
    if not rule_df.empty:
        lines.append("## Deployable Rule Acceptance")
        lines.append("")
        any_clear = False
        for row in rule_df[rule_df["fold"].astype(str) == "all"].itertuples(index=False):
            clears = (
                row.precision_improve_or_neutral >= 0.75
                and row.worst_case_harm_acc >= -1.0
                and (row.mean_delta_edit > 0.0 or row.mean_delta_f1_50 > 0.0)
            )
            any_clear = any_clear or clears
            lines.append(
                f"- {row.dataset} L={row.L_abs} {row.rule}: clears={clears}, "
                f"precision={row.precision_improve_or_neutral:.3f}, worst_harm={row.worst_case_harm_acc:.3f}."
            )
        if not any_clear:
            lines.append("")
            lines.append("No deployable rule clears precision >= 0.75, worst harm <= 1 pp, and positive Edit/F1@50.")
    else:
        lines.append("No dataset passed the oracle gate, so deployable rules were skipped.")
    (out_dir / "island_precheck_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets = ["gtea"]
        args.folds = [1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    complete_cases = load_complete_case_set()
    case_rows: List[Dict[str, Any]] = []
    island_rows: List[Dict[str, Any]] = []
    loaded_cases = 0
    skipped_incomplete = 0
    total_frames = 0
    total_case_l_rows = 0

    for dataset in args.datasets:
        for fold in get_folds(dataset, args.folds, args.data_root):
            ctx = load_fold_context(dataset, fold, args.data_root)
            test_cases = ctx.test_cases[: args.case_limit] if args.case_limit else ctx.test_cases
            for case_id in test_cases:
                key = (dataset, fold, str(case_id))
                if key not in complete_cases:
                    skipped_incomplete += 1
                    continue
                loaded_cases += 1
                first_l = True
                for l_abs in [int(x) for x in args.l_abs]:
                    row, islands = case_record(ctx=ctx, case_id=str(case_id), l_abs=l_abs)
                    if first_l:
                        total_frames += int(row["n_frames"])
                        first_l = False
                    total_case_l_rows += 1
                    case_rows.append(row)
                    island_rows.extend(islands)

    if not case_rows:
        raise ValueError("No complete cases loaded")
    ceiling_df = build_oracle_ceiling(case_rows, [int(x) for x in args.l_abs])
    gates = gate_datasets(ceiling_df)
    rule_df = build_rule_eval(case_rows, [int(x) for x in args.l_abs], gates)
    public_case_df = pd.DataFrame([public_case_row(r) for r in case_rows])
    island_df = pd.DataFrame(island_rows)
    outlier_df = public_case_df[
        public_case_df.apply(
            lambda r: (str(r["dataset"]), int(r["fold"]), str(r["case_id"])) in OUTLIER_CASES,
            axis=1,
        )
    ].copy()

    ceiling_df.to_csv(out_dir / "island_oracle_ceiling.csv", index=False)
    rule_df.to_csv(out_dir / "island_rule_eval.csv", index=False)
    public_case_df.to_csv(out_dir / "island_case_details.csv", index=False)
    island_df.to_csv(out_dir / "island_candidate_details.csv", index=False)
    outlier_df.to_csv(out_dir / "island_outlier_cases.csv", index=False)
    write_smoke_report(out_dir, public_case_df, island_df, ceiling_df)
    write_summary(out_dir, ceiling_df, rule_df, gates)

    payload = {
        "out_dir": str(out_dir),
        "datasets": args.datasets,
        "folds": args.folds,
        "L_abs": [int(x) for x in args.l_abs],
        "gate_threshold_points": GATE_THRESHOLD,
        "rule_margin_threshold": RULE_MARGIN_THRESHOLD,
        "parity_flags": {
            "raw_softmax_argmax_matches_case_csv": True,
            "frame_counts_match_gt_argmax_sktr_softmax": True,
            "incomplete_cases_excluded": True,
            "loaded_cases": loaded_cases,
            "skipped_incomplete_cases": skipped_incomplete,
            "total_frames": total_frames,
            "total_case_l_rows": total_case_l_rows,
            "oracle_uses_test_gt": True,
            "deployable_rules_use_gt": False,
            "gt_used_for_rule_eval": True,
        },
        "dataset_gates": gates,
        "outputs": [
            "island_oracle_ceiling.csv",
            "island_rule_eval.csv",
            "island_precheck_summary.md",
            "island_smoke_report.md",
            "island_case_details.csv",
            "island_candidate_details.csv",
            "island_outlier_cases.csv",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Stage-1B0 island cleanup pre-check to {out_dir}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
