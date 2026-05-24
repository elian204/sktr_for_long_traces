#!/usr/bin/env python3
"""
Stage-3C0: DiffAct post-process reconciliation and adaptive-postprocess probes.

This stage re-anchors every comparison to DiffAct's official fold-local
prediction path, then caches the pre/post-process substrate on native grids.
GT is used only for evaluation/oracle probes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DIFFACT_ROOT = REPO_ROOT / "baselines" / "DiffAct"
if str(DIFFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFACT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import (  # noqa: E402
    VideoFeatureDataset,
    get_data_dict,
    restore_full_sequence,
)
from main import Trainer  # noqa: E402
from stage3b_diffusion_ensemble_feasibility import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_SOFTMAX_ROOT,
    PUBLISHED,
    load_labels,
    load_test_videos,
    parse_video_index_map,
)
from stage3b_gate05_diffusion_ensemble_probe import (  # noqa: E402
    config_name,
    event_list_for,
    load_gt,
)
from utils import get_labels_start_end_time, load_config_file, mode_filter  # noqa: E402
from src.evaluation import _edit_score_asformer, _segmental_f1_counts_asformer  # noqa: E402


DEFAULT_OUT_DIR = Path("/data1/eli-bogdanov/sktr_runs/stage3c0_postprocess_reconciliation_v1")
WINDOWS = [0, 5, 10, 15, 30, 45, 60]
PURGE_THRESHOLDS = [0, 1, 3, 5, 10, 25]
BOUNDARY_W = 25
LONG_WRONG_MIN_LEN = 100
LONG_WRONG_PURITY = 0.90
METRIC_KEYS = ["acc", "edit", "f1@10", "f1@25", "f1@50"]


@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    fold: int
    case_id: str
    video: str
    fold_local_idx: int


def case_key(case: CaseSpec) -> Tuple[str, int, str]:
    return (case.dataset, case.fold, case.case_id)


def rel_case_path(case: CaseSpec, suffix: str) -> Path:
    return Path(case.dataset) / f"fold{case.fold}" / f"{case.case_id}{suffix}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, arr: np.ndarray) -> str:
    ensure_parent(path)
    np.save(path, arr)
    return str(path)


def maybe_symlink(src: Path, dst: Path) -> str:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def metrics(data_root: Path, dataset: str, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return aggregate_metric_components([metric_components(gt, pred, background_id_for_dataset(data_root, dataset))])


def background_id_for_dataset(data_root: Path, dataset: str) -> Optional[int]:
    labels = event_list_for(data_root, dataset)
    for idx, name in enumerate(labels):
        if str(name).lower() == "background":
            return idx
    return None


def metric_components(gt: np.ndarray, pred: np.ndarray, background_id: Optional[int] = None) -> Dict[str, Any]:
    y_true = [int(x) for x in gt]
    y_pred = [int(x) for x in pred]
    total = len(y_true)
    if total == 0:
        return {
            "n_frames": 0,
            "correct": 0,
            "edit": 0.0,
            "tp": np.zeros(3, dtype=np.float64),
            "fp": np.zeros(3, dtype=np.float64),
            "fn": np.zeros(3, dtype=np.float64),
        }
    background = int(background_id) if background_id is not None else None
    bg_class = [background] if background is not None else ["background"]
    tp_arr = np.zeros(3, dtype=np.float64)
    fp_arr = np.zeros(3, dtype=np.float64)
    fn_arr = np.zeros(3, dtype=np.float64)
    for idx, thresh in enumerate([0.10, 0.25, 0.50]):
        tp, fp, fn = _segmental_f1_counts_asformer(y_true, y_pred, thresh, background)
        tp_arr[idx] = tp
        fp_arr[idx] = fp
        fn_arr[idx] = fn
    return {
        "n_frames": int(total),
        "correct": int((gt == pred).sum()),
        "edit": float(_edit_score_asformer(y_true, y_pred, bg_class)),
        "tp": tp_arr,
        "fp": fp_arr,
        "fn": fn_arr,
    }


def aggregate_metric_components(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not items:
        return {"acc": 0.0, "edit": 0.0, "f1@10": 0.0, "f1@25": 0.0, "f1@50": 0.0, "acc_micro": 0.0}
    total = sum(int(x["n_frames"]) for x in items)
    correct = sum(int(x["correct"]) for x in items)
    edit_scores = [float(x["edit"]) for x in items]
    tp_total = np.sum([x["tp"] for x in items], axis=0)
    fp_total = np.sum([x["fp"] for x in items], axis=0)
    fn_total = np.sum([x["fn"] for x in items], axis=0)
    acc = 100.0 * correct / total if total else 0.0
    out = {
        "acc": float(acc),
        "edit": float(np.mean(edit_scores)) if edit_scores else 0.0,
    }
    for idx, name in enumerate(["f1@10", "f1@25", "f1@50"]):
        precision = tp_total[idx] / (tp_total[idx] + fp_total[idx]) if tp_total[idx] + fp_total[idx] > 0 else 0.0
        recall = tp_total[idx] / (tp_total[idx] + fn_total[idx]) if tp_total[idx] + fn_total[idx] > 0 else 0.0
        out[name] = float(2.0 * precision * recall / (precision + recall) * 100.0) if precision + recall > 0 else 0.0
    out["acc_micro"] = out["acc"]
    return out


def dataset_metrics(data_root: Path, cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], pred_map: Dict[Tuple[str, int, str], np.ndarray]) -> Dict[str, float]:
    items = []
    background_cache: Dict[str, Optional[int]] = {}
    for case in cases:
        if case.dataset not in background_cache:
            background_cache[case.dataset] = background_id_for_dataset(data_root, case.dataset)
        key = case_key(case)
        gt = gt_map[key]
        pred = pred_map[key]
        if len(gt) != len(pred):
            raise ValueError(f"{key} length mismatch: gt={len(gt)} pred={len(pred)}")
        items.append(metric_components(gt, pred, background_cache[case.dataset]))
    return aggregate_metric_components(items)


def metric_delta(row: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {f"delta_{k}": float(row[k] - base[k]) for k in METRIC_KEYS}


def segments(x: np.ndarray) -> List[Tuple[int, int, int]]:
    if len(x) == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(x[0])
    for idx in range(1, len(x)):
        if int(x[idx]) != cur:
            out.append((start, idx, cur))
            start = idx
            cur = int(x[idx])
    out.append((start, len(x), cur))
    return out


def boundary_mask(labels: np.ndarray, width: int = BOUNDARY_W) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=bool)
    for b in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        lo = max(0, int(b) - width)
        hi = min(len(labels), int(b) + width + 1)
        mask[lo:hi] = True
    return mask


def long_wrong_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros(len(gt), dtype=bool)
    for start, end, label in segments(pred):
        seg_wrong = pred[start:end] != gt[start:end]
        homogeneous = float((pred[start:end] == label).mean())
        if end - start >= LONG_WRONG_MIN_LEN and homogeneous >= LONG_WRONG_PURITY and seg_wrong.mean() >= LONG_WRONG_PURITY:
            out[start:end] = seg_wrong
    return out


def short_segments_removed(before: np.ndarray, after: np.ndarray, max_len: int) -> int:
    removed = 0
    after_labels = set(int(x) for x in np.unique(after))
    for start, end, label in segments(before):
        if end - start <= max_len and int(label) not in after_labels:
            removed += 1
    return removed


def postprocess_full_pred(pred: np.ndarray, postprocess: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    pred = pred.astype(np.int32, copy=True)
    purged: List[Dict[str, Any]] = []
    post_type = postprocess["type"]
    if post_type == "mode":
        return mode_filter(pred, postprocess["value"]).astype(np.int32, copy=False), purged
    if post_type != "purge":
        return pred, purged
    trans, starts, ends = get_labels_start_end_time(pred)
    if len(trans) <= 1:
        return pred, purged
    for idx in range(len(trans)):
        duration = ends[idx] - starts[idx]
        if duration <= postprocess["value"]:
            old_label = int(trans[idx])
            if idx == 0:
                pred[starts[idx] : ends[idx]] = trans[idx + 1]
                new_label = int(trans[idx + 1])
            elif idx == len(trans) - 1:
                pred[starts[idx] : ends[idx]] = trans[idx - 1]
                new_label = int(trans[idx - 1])
            else:
                mid = starts[idx] + duration // 2
                pred[starts[idx] : mid] = trans[idx - 1]
                pred[mid : ends[idx]] = trans[idx + 1]
                new_label = -1
            purged.append(
                {
                    "start": int(starts[idx]),
                    "end": int(ends[idx]),
                    "duration": int(duration),
                    "old_label": old_label,
                    "new_label": new_label,
                }
            )
    return pred.astype(np.int32, copy=False), purged


def median_and_renorm(probs: np.ndarray, window: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    if window <= 0:
        out = probs.copy()
    else:
        out = np.zeros_like(probs)
        for c in range(probs.shape[0]):
            out[c] = median_filter(probs[c], size=window)
    out = out / np.maximum(out.sum(axis=0, keepdims=True), 1e-12)
    return out.astype(np.float32, copy=False)


def restore_probs(probs: np.ndarray, full_len: int, left_offset: int, right_offset: int, sample_rate: int) -> np.ndarray:
    out = np.zeros((probs.shape[0], full_len), dtype=np.float32)
    for c in range(probs.shape[0]):
        out[c] = restore_full_sequence(probs[c], full_len, left_offset, right_offset, sample_rate)
    out = out / np.maximum(out.sum(axis=0, keepdims=True), 1e-12)
    return out


def restore_pred_from_native(probs: np.ndarray, full_len: int, sample_rate: int, *, left_offset: Optional[int] = None, right_offset: Optional[int] = None) -> np.ndarray:
    if left_offset is None:
        left_offset = sample_rate // 2
    if right_offset is None:
        right_offset = (sample_rate - 1) // 2
    pred_sub = probs.argmax(axis=0)
    return restore_full_sequence(pred_sub, full_len, left_offset, right_offset, sample_rate).astype(np.int32)


def build_cases(data_root: Path, softmax_root: Path, smoke: bool, datasets: Sequence[str]) -> List[CaseSpec]:
    out: List[CaseSpec] = []
    use_datasets = ["gtea"] if smoke else list(datasets)
    for dataset in use_datasets:
        n_folds = int(PUBLISHED[dataset]["folds"])
        for fold in range(1, n_folds + 1):
            if smoke and fold != 1:
                continue
            fold_dir = softmax_root / dataset / f"softmax_fold{fold}"
            id_to_video = parse_video_index_map(fold_dir / "video_index_map.txt")
            video_to_id = {video: case_id for case_id, video in id_to_video.items()}
            test_videos = load_test_videos(data_root, dataset, fold)
            for idx, video in enumerate(test_videos):
                out.append(CaseSpec(dataset, fold, video_to_id[video], video, idx))
    return out


class FoldExtractor:
    def __init__(self, data_root: Path, dataset: str, fold: int, videos: Sequence[str], device: Any) -> None:
        self.data_root = data_root
        self.dataset = dataset
        self.fold = fold
        self.device = device
        self.cfg = load_config_file(str(DIFFACT_ROOT / "configs" / config_name(dataset, fold)))
        self.cfg["root_data_dir"] = str(data_root)
        self.event_list = event_list_for(data_root, dataset)
        data_dict = get_data_dict(
            feature_dir=str(data_root / dataset / "features"),
            label_dir=str(data_root / dataset / "groundTruth"),
            video_list=list(videos),
            event_list=self.event_list,
            sample_rate=self.cfg["sample_rate"],
            temporal_aug=self.cfg["temporal_aug"],
            boundary_smooth=self.cfg["boundary_smooth"],
        )
        self.dataset_obj = VideoFeatureDataset(data_dict, len(self.event_list), mode="test")
        self.video_to_idx = {video: idx for idx, video in enumerate(self.dataset_obj.video_list)}
        self.trainer = Trainer(
            dict(self.cfg["encoder_params"]),
            dict(self.cfg["decoder_params"]),
            dict(self.cfg["diffusion_params"]),
            self.event_list,
            self.cfg["sample_rate"],
            self.cfg["temporal_aug"],
            self.cfg["set_sampling_seed"],
            self.cfg["postprocess"],
            device=device,
        )
        import torch

        state = torch.load(
            str(DIFFACT_ROOT / "trained_models" / self.cfg["naming"] / "release.model"),
            map_location=device,
        )
        self.trainer.model.load_state_dict(state)
        self.trainer.model.eval().to(device)

    def extract(self, case: CaseSpec) -> Dict[str, Any]:
        import torch

        idx = self.video_to_idx[case.video]
        feature, label, _, video = self.dataset_obj[idx]
        if video != case.video:
            raise RuntimeError(f"Video mismatch: {video} != {case.video}")
        seed = case.fold_local_idx if self.trainer.set_sampling_seed else None
        with torch.no_grad():
            offset_probs = [
                self.trainer.model.ddim_sample(f.to(self.device), seed).squeeze(0).cpu().numpy().astype(np.float32)
                for f in feature
            ]
        min_len = min(p.shape[1] for p in offset_probs)
        truncated = [p[:, :min_len] for p in offset_probs]
        mean_pre = np.stack(truncated, axis=0).mean(axis=0).astype(np.float32)
        post_type = self.trainer.postprocess["type"]
        post_native = median_and_renorm(mean_pre, self.trainer.postprocess["value"]) if post_type == "median" else mean_pre.copy()
        full_len = int(label.shape[-1])
        sample_rate = int(self.trainer.sample_rate)
        pre_full = restore_pred_from_native(mean_pre, full_len, sample_rate)
        post_native_full = restore_pred_from_native(post_native, full_len, sample_rate)
        official, purged = postprocess_full_pred(post_native_full, self.trainer.postprocess)
        offset_preds = []
        for offset, probs in enumerate(offset_probs):
            # For diagnostic offset predictions, restore each offset on its own
            # actual temporal grid rather than pretending it is the aggregated grid.
            right = full_len - (offset + sample_rate * (probs.shape[1] - 1)) - 1
            if right < 0:
                raise RuntimeError(
                    f"Invalid offset restore geometry: full_len={full_len}, "
                    f"offset={offset}, sample_rate={sample_rate}, T={probs.shape[1]}"
                )
            offset_preds.append(
                restore_pred_from_native(
                    probs,
                    full_len,
                    sample_rate,
                    left_offset=offset,
                    right_offset=right,
                )
            )
        return {
            "offset_probs": offset_probs,
            "mean_pre": mean_pre,
            "post_native": post_native,
            "pre_full": pre_full,
            "post_native_full": post_native_full,
            "official": official,
            "purged_segments": purged,
            "offset_preds": offset_preds,
            "full_len": full_len,
            "sample_rate": sample_rate,
            "n_classes": mean_pre.shape[0],
        }


def compare_pair(data_root: Path, case: CaseSpec, gt: np.ndarray, left_name: str, left: np.ndarray, right_name: str, right: np.ndarray) -> Dict[str, Any]:
    diff = left != right
    boundary = boundary_mask(gt)
    left_m = metrics(data_root, case.dataset, gt, left)
    right_m = metrics(data_root, case.dataset, gt, right)
    return {
        "dataset": case.dataset,
        "fold": case.fold,
        "case_id": case.case_id,
        "video": case.video,
        "pair": f"{left_name}_vs_{right_name}",
        "left_prediction": left_name,
        "right_prediction": right_name,
        "n_frames": int(len(gt)),
        "differing_frames": int(diff.sum()),
        "frame_disagreement_rate": float(diff.mean()),
        "differing_boundary_w25_frames": int((diff & boundary).sum()),
        "differing_interior_frames": int((diff & ~boundary).sum()),
        "boundary_diff_share": float(((diff & boundary).sum() / diff.sum()) if diff.sum() else 0.0),
        "left_segment_count": len(segments(left)),
        "right_segment_count": len(segments(right)),
        "segment_count_diff_left_minus_right": len(segments(left)) - len(segments(right)),
        "left_acc": left_m["acc"],
        "left_edit": left_m["edit"],
        "left_f1@10": left_m["f1@10"],
        "left_f1@25": left_m["f1@25"],
        "left_f1@50": left_m["f1@50"],
        "right_acc": right_m["acc"],
        "right_edit": right_m["edit"],
        "right_f1@10": right_m["f1@10"],
        "right_f1@25": right_m["f1@25"],
        "right_f1@50": right_m["f1@50"],
        "short_segments_removed_left_to_right_leq3": short_segments_removed(left, right, 3),
        "short_segments_removed_left_to_right_leq25": short_segments_removed(left, right, 25),
    }


def apply_purge_threshold(pred: np.ndarray, threshold: int) -> np.ndarray:
    if threshold <= 0:
        return pred.astype(np.int32, copy=True)
    out, _ = postprocess_full_pred(pred, {"type": "purge", "value": threshold})
    return out


def prediction_for_window(cache: Dict[Tuple[str, int, str], Dict[str, Any]], case: CaseSpec, window: int, variant: str) -> np.ndarray:
    item = cache[case_key(case)]
    pred_cache = item.setdefault("window_predictions", {})
    pred_key = (variant, int(window))
    if pred_key in pred_cache:
        return pred_cache[pred_key]

    sample_rate = item["sample_rate"]
    full_len = item["full_len"]
    if case.dataset == "gtea":
        pred = apply_purge_threshold(item["pre_full"], window)
    elif variant == "mean_then_median":
        native = median_and_renorm(item["mean_pre"], window)
        pred = restore_pred_from_native(native, full_len, sample_rate)
    elif variant == "median_each_offset_then_mean":
        filtered = [median_and_renorm(p[:, : item["mean_pre"].shape[1]], window) for p in item["offset_probs"]]
        native = np.stack(filtered, axis=0).mean(axis=0)
        native = native / np.maximum(native.sum(axis=0, keepdims=True), 1e-12)
        pred = restore_pred_from_native(native, full_len, sample_rate)
    else:
        raise ValueError(variant)
    pred_cache[pred_key] = pred.astype(np.int32, copy=False)
    return pred_cache[pred_key]


def aggregate_metrics_for_cases(data_root: Path, cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], pred_map: Dict[Tuple[str, int, str], np.ndarray]) -> Dict[str, float]:
    return dataset_metrics(data_root, cases, gt_map, pred_map)


def sweep_windows(data_root: Path, cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], official_map: Dict[Tuple[str, int, str], np.ndarray], cache: Dict[Tuple[str, int, str], Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    heldout_rows: List[Dict[str, Any]] = []
    for dataset in sorted({c.dataset for c in cases}):
        print(f"Probe 1 sweep: {dataset}", flush=True)
        ds_cases = [c for c in cases if c.dataset == dataset]
        variants = ["purge_threshold"] if dataset == "gtea" else ["mean_then_median"] + (["median_each_offset_then_mean"] if dataset == "50salads" else [])
        params = PURGE_THRESHOLDS if dataset == "gtea" else WINDOWS
        background_id = background_id_for_dataset(data_root, dataset)
        base_components = {
            case_key(c): metric_components(gt_map[case_key(c)], official_map[case_key(c)], background_id)
            for c in ds_cases
        }
        for variant in variants:
            eval_cache: Dict[Tuple[Tuple[str, int, str], int], Dict[str, Any]] = {}

            def eval_for(case: CaseSpec, param: int) -> Dict[str, Any]:
                key = case_key(case)
                cache_key = (key, int(param))
                if cache_key not in eval_cache:
                    pred = prediction_for_window(cache, case, param, variant)
                    eval_cache[cache_key] = metric_components(gt_map[key], pred, background_id)
                return eval_cache[cache_key]

            for fold in sorted({c.fold for c in ds_cases}):
                fold_cases = [c for c in ds_cases if c.fold == fold]
                base = aggregate_metric_components([base_components[case_key(c)] for c in fold_cases])
                for param in params:
                    m = aggregate_metric_components([eval_for(c, param) for c in fold_cases])
                    row = {"dataset": dataset, "fold": fold, "variant": variant, "param": param, **m}
                    row.update(metric_delta(m, base))
                    rows.append(row)

            folds = sorted({c.fold for c in ds_cases})
            if len(folds) > 1:
                for heldout in folds:
                    train_cases = [c for c in ds_cases if c.fold != heldout]
                    held_cases = [c for c in ds_cases if c.fold == heldout]
                    scored = []
                    for param in params:
                        m = aggregate_metric_components([eval_for(c, param) for c in train_cases])
                        scored.append((m["f1@50"], m["edit"], m["acc"], param))
                    selected = max(scored)[3]
                    held_base = aggregate_metric_components([base_components[case_key(c)] for c in held_cases])
                    m = aggregate_metric_components([eval_for(c, selected) for c in held_cases])
                    row = {"dataset": dataset, "heldout_fold": heldout, "variant": variant, "selected_param": selected, **m}
                    row.update(metric_delta(m, held_base))
                    heldout_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(heldout_rows)


def oracle_by_case(data_root: Path, cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], official_map: Dict[Tuple[str, int, str], np.ndarray], cache: Dict[Tuple[str, int, str], Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        base = metrics(data_root, case.dataset, gt_map[case_key(case)], official_map[case_key(case)])
        if case.dataset == "gtea":
            variants = ["purge_threshold"]
            params = PURGE_THRESHOLDS
        else:
            variants = ["mean_then_median"] + (["median_each_offset_then_mean"] if case.dataset == "50salads" else [])
            params = WINDOWS
        best = None
        for variant in variants:
            for param in params:
                pred = prediction_for_window(cache, case, param, variant)
                m = metrics(data_root, case.dataset, gt_map[case_key(case)], pred)
                score = (m["f1@50"], m["edit"], m["acc"])
                if best is None or score > best[0]:
                    best = (score, variant, param, m)
        assert best is not None
        row = {
            "dataset": case.dataset,
            "fold": case.fold,
            "case_id": case.case_id,
            "video": case.video,
            "oracle_variant": best[1],
            "oracle_param": best[2],
            **best[3],
        }
        row.update(metric_delta(best[3], base))
        rows.append(row)
    return pd.DataFrame(rows)


def offset_oracle_50salads(data_root: Path, cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], official_map: Dict[Tuple[str, int, str], np.ndarray], cache: Dict[Tuple[str, int, str], Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fs_cases = [c for c in cases if c.dataset == "50salads"]
    by_fold = defaultdict(list)
    by_fold["all"] = fs_cases
    for c in fs_cases:
        by_fold[c.fold].append(c)
    for scope, scope_cases in by_fold.items():
        if not scope_cases:
            continue
        base = aggregate_metrics_for_cases(data_root, scope_cases, gt_map, official_map)
        best_case_map: Dict[Tuple[str, int, str], np.ndarray] = {}
        best_frame_map: Dict[Tuple[str, int, str], np.ndarray] = {}
        disagreement_vals = []
        any_correct_vals = []
        for case in scope_cases:
            key = case_key(case)
            gt = gt_map[key]
            offsets = cache[key]["offset_preds"]
            best_score = None
            best_pred = offsets[0]
            for pred in offsets:
                m = metrics(data_root, "50salads", gt, pred)
                score = (m["f1@50"], m["edit"], m["acc"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_pred = pred
            best_case_map[key] = best_pred
            stacked = np.stack(offsets, axis=0)
            frame_oracle = offsets[0].copy()
            any_correct = (stacked == gt[None, :]).any(axis=0)
            frame_oracle[any_correct] = gt[any_correct]
            best_frame_map[key] = frame_oracle
            long_mask = long_wrong_mask(gt, official_map[key])
            if long_mask.any():
                pairs = []
                for i in range(stacked.shape[0]):
                    for j in range(i + 1, stacked.shape[0]):
                        pairs.append(float((stacked[i, long_mask] != stacked[j, long_mask]).mean()))
                disagreement_vals.append(float(np.mean(pairs)))
                any_correct_vals.append(float(any_correct[long_mask].mean()))
        for method, pred_map in [("oracle_best_offset_per_case", best_case_map), ("oracle_best_offset_per_frame", best_frame_map)]:
            m = aggregate_metrics_for_cases(data_root, scope_cases, gt_map, pred_map)
            row = {
                "dataset": "50salads",
                "scope": scope,
                "method": method,
                "n_cases": len(scope_cases),
                **m,
                "long_substitution_offset_disagreement_mean": float(np.nanmean(disagreement_vals)) if disagreement_vals else float("nan"),
                "long_substitution_any_offset_correct_mean": float(np.nanmean(any_correct_vals)) if any_correct_vals else float("nan"),
            }
            row.update(metric_delta(m, base))
            rows.append(row)
    return pd.DataFrame(rows)


def boundary_oracle(pred: np.ndarray, gt: np.ndarray, window: int = BOUNDARY_W) -> np.ndarray:
    """Cheap local boundary-shift oracle preserving segment labels/order.

    Each boundary is shifted within +/- window while neighboring boundaries are
    fixed at their current locations. This is a diagnostic ceiling for whether
    recoverable boundary information exists; it is intentionally not a global
    dynamic program so it remains tractable for Breakfast-scale runs.
    """
    segs = segments(pred)
    if len(segs) <= 1:
        return pred.copy()
    bounds = [end for _, end, _ in segs[:-1]]
    new_bounds = bounds.copy()
    prefix: Dict[int, np.ndarray] = {}

    def prefix_for(label: int) -> np.ndarray:
        label = int(label)
        if label not in prefix:
            prefix[label] = np.concatenate([[0], np.cumsum(gt == label)]).astype(np.int32)
        return prefix[label]

    for idx, b in enumerate(bounds):
        left_start = 0 if idx == 0 else bounds[idx - 1]
        right_end = len(pred) if idx == len(bounds) - 1 else bounds[idx + 1]
        left_label = segs[idx][2]
        right_label = segs[idx + 1][2]
        lo = max(left_start + 1, b - window)
        hi = min(right_end - 1, b + window)
        if hi < lo:
            continue
        cand = np.arange(lo, hi + 1, dtype=np.int32)
        left_prefix = prefix_for(left_label)
        right_prefix = prefix_for(right_label)
        scores = (
            left_prefix[cand] - left_prefix[left_start]
            + right_prefix[right_end] - right_prefix[cand]
        )
        best_b = int(cand[int(np.argmax(scores))])
        new_bounds[idx] = best_b
    # Ensure monotonicity after independent shifts.
    for idx in range(1, len(new_bounds)):
        if new_bounds[idx] <= new_bounds[idx - 1]:
            new_bounds[idx] = new_bounds[idx - 1] + 1
    for idx in range(len(new_bounds) - 2, -1, -1):
        if new_bounds[idx] >= new_bounds[idx + 1]:
            new_bounds[idx] = new_bounds[idx + 1] - 1
    if new_bounds[0] <= 0 or new_bounds[-1] >= len(pred):
        return pred.copy()
    out = np.empty_like(pred)
    start = 0
    for (seg_start, seg_end, label), b in zip(segs[:-1], new_bounds):
        out[start:b] = label
        start = b
    out[start:] = segs[-1][2]
    return out.astype(np.int32, copy=False)


def boundary_headroom(data_root: Path, cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], official_map: Dict[Tuple[str, int, str], np.ndarray], cache: Dict[Tuple[str, int, str], Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in sorted({c.dataset for c in cases}):
        ds_cases = [c for c in cases if c.dataset == dataset]
        pred_types = ["official", "pre_postprocess", "post_native_argmax"]
        for pred_type in pred_types:
            pred_map = {}
            oracle_map = {}
            for case in ds_cases:
                key = case_key(case)
                if pred_type == "official":
                    pred = official_map[key]
                elif pred_type == "pre_postprocess":
                    pred = cache[key]["pre_full"]
                else:
                    pred = cache[key]["post_native_full"]
                pred_map[key] = pred
                oracle_map[key] = boundary_oracle(pred, gt_map[key])
            base = aggregate_metrics_for_cases(data_root, ds_cases, gt_map, pred_map)
            oracle = aggregate_metrics_for_cases(data_root, ds_cases, gt_map, oracle_map)
            row = {"dataset": dataset, "prediction_type": pred_type, **oracle}
            row.update(metric_delta(oracle, base))
            rows.append(row)
    return pd.DataFrame(rows)


def official_error_mass(cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], official_map: Dict[Tuple[str, int, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    for dataset in sorted({c.dataset for c in cases}):
        total_errors = boundary_errors = island_errors = long_errors = 0
        addressable_errors = 0
        for case in [c for c in cases if c.dataset == dataset]:
            key = case_key(case)
            gt = gt_map[key]
            pred = official_map[key]
            err = pred != gt
            boundary = boundary_mask(gt)
            long_mask = long_wrong_mask(gt, pred)
            short_mask = np.zeros(len(gt), dtype=bool)
            total_errors += int(err.sum())
            boundary_errors += int((err & boundary).sum())
            long_errors += int(long_mask.sum())
            # operational short island proxy
            for start, end, _ in segments(pred):
                if end - start <= 25:
                    short_mask[start:end] = True
            island_errors += int((err & short_mask).sum())
            addressable_errors += int((err & (boundary | short_mask)).sum())
        rows.append(
            {
                "dataset": dataset,
                "official_total_error_frames": total_errors,
                "boundary_w25_error_share": boundary_errors / total_errors if total_errors else float("nan"),
                "short_segment_error_share": island_errors / total_errors if total_errors else float("nan"),
                "long_substitution_error_share": long_errors / total_errors if total_errors else float("nan"),
                "postprocessor_addressable_proxy_share": addressable_errors / total_errors if total_errors else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def official_baseline_vs_paper(data_root: Path, cases: Sequence[CaseSpec], gt_map: Dict[Tuple[str, int, str], np.ndarray], official_map: Dict[Tuple[str, int, str], np.ndarray]) -> pd.DataFrame:
    metric_to_paper = {
        "acc": "Acc",
        "edit": "Edit",
        "f1@10": "F1@10",
        "f1@25": "F1@25",
        "f1@50": "F1@50",
    }
    rows = []
    for dataset in sorted({c.dataset for c in cases}):
        ds_cases = [c for c in cases if c.dataset == dataset]
        official = aggregate_metrics_for_cases(data_root, ds_cases, gt_map, official_map)
        row: Dict[str, Any] = {"dataset": dataset, "n_cases": len(ds_cases)}
        for metric, paper_key in metric_to_paper.items():
            paper = float(PUBLISHED[dataset][paper_key])
            row[f"official_{metric}"] = official[metric]
            row[f"paper_{metric}"] = paper
            row[f"official_minus_paper_{metric}"] = official[metric] - paper
        rows.append(row)
    return pd.DataFrame(rows)


def write_metadata(out_dir: Path) -> None:
    meta = {}
    for dataset in ["gtea", "50salads", "breakfast"]:
        cfg = load_config_file(str(DIFFACT_ROOT / "configs" / config_name(dataset, 1)))
        meta[dataset] = {
            "sample_rate": cfg["sample_rate"],
            "temporal_aug": cfg["temporal_aug"],
            "n_offsets": cfg["sample_rate"] if cfg["temporal_aug"] else 1,
            "median_window": cfg["postprocess"]["value"] if cfg["postprocess"]["type"] == "median" else None,
            "purge_threshold": cfg["postprocess"]["value"] if cfg["postprocess"]["type"] == "purge" else None,
            "postprocess_type": cfg["postprocess"]["type"],
            "operation_order": "ddim_sample -> offset mean -> median+renorm if configured -> argmax -> restore_full_sequence -> purge if configured",
            "decoder_logits": "not captured",
            "encoder_softmax_stream": "not captured",
        }
    (out_dir / "postprocess_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def df_to_markdown(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """Small markdown table renderer without pandas' optional tabulate dependency."""
    if df.empty:
        return "(empty)"
    view = df.head(max_rows).copy() if max_rows else df.copy()
    cols = list(view.columns)

    def fmt(val: Any) -> str:
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                return ""
            return f"{float(val):.4f}"
        return str(val)

    rows = [[fmt(row[col]) for col in cols] for _, row in view.iterrows()]
    widths = [len(str(c)) for c in cols]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |" for row in rows]
    if max_rows and len(df) > max_rows:
        body.append(f"| ... {len(df) - max_rows} more rows |" + " |".join("" for _ in cols[1:]) + " |")
    return "\n".join([header, sep, *body])


def write_reconciliation_summary(out_dir: Path, case_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Postprocess Reconciliation Summary",
        "",
        "Official baseline for all future comparisons: **fold-local DiffAct `test_single_video` prediction**, not raw argmax of a restored softmax matrix.",
        "",
        "## Key Pair Summary",
        "",
        df_to_markdown(summary_df),
        "",
        "## Direct Answers",
        "",
    ]
    arg = summary_df[summary_df["pair"] == "official_vs_argmax_probs_full"]
    if not arg.empty:
        max_dis = float(arg["frame_disagreement_rate_mean"].max())
        lines.append(f"- `argmax(probs_full)` is not assumed identical to official. Maximum mean disagreement among datasets in this run: `{max_dis:.4f}`.")
    lines.extend(
        [
            "- Prior SKTR/error-taxonomy analyses used the exported softmax/case-output argmax stream. For GTEA this is especially important because purge is an official final step and the earlier all-video exports used global case-index seeds, not fold-local test indices.",
            "- GTEA purge directly targets only segments of length <=3 frames; the near-zero short-island finding is therefore partly expected after official purge.",
            "- 50Salads/Breakfast median smoothing happens before restoration. Any boundary/crossing analysis on restored post-median probabilities is already downstream of a strong fixed smoother.",
            "- Raw decoder logits and encoder streams were not captured in this stage.",
        ]
    )
    (out_dir / "postprocess_reconciliation_summary.md").write_text("\n".join(lines) + "\n")


def write_probe_summary(out_dir: Path, sweep: pd.DataFrame, held: pd.DataFrame, oracle: pd.DataFrame, offset: pd.DataFrame, boundary: pd.DataFrame, error_mass: pd.DataFrame, baseline_paper: pd.DataFrame) -> None:
    lines = [
        "# Adaptive Postprocess Probe Summary",
        "",
        "All deltas are versus official fold-local DiffAct `pred_full`. Test-swept, fold-held-out, and oracle numbers are separated.",
        "",
        "## Official Baseline Vs Paper",
        "",
        df_to_markdown(
            baseline_paper[
                [
                    "dataset",
                    "n_cases",
                    "official_acc",
                    "paper_acc",
                    "official_minus_paper_acc",
                    "official_edit",
                    "paper_edit",
                    "official_minus_paper_edit",
                    "official_f1@50",
                    "paper_f1@50",
                    "official_minus_paper_f1@50",
                ]
            ]
        ),
        "",
        "Metric-win constraint: GTEA claims should use F1@50/Edit only because Acc remains below the paper baseline; 50Salads and Breakfast can report all metrics.",
        "",
        "## Fold-Held-Out Best Rows",
        "",
        df_to_markdown(held.sort_values(["dataset", "variant", "heldout_fold"])) if not held.empty else "(no held-out rows)",
        "",
        "## Per-Case Oracle Window/Purge Aggregate",
        "",
    ]
    oracle_agg = oracle.groupby("dataset")[["delta_f1@50", "delta_edit", "delta_acc"]].mean().reset_index() if not oracle.empty else pd.DataFrame()
    lines.append(df_to_markdown(oracle_agg) if not oracle_agg.empty else "(empty)")
    lines.extend(["", "## 50Salads Offset Oracle", ""])
    lines.append(df_to_markdown(offset) if not offset.empty else "(empty)")
    lines.extend(["", "## Boundary Headroom Pre Vs Post", ""])
    lines.append(df_to_markdown(boundary) if not boundary.empty else "(empty)")
    lines.extend(["", "## Structural Ceiling Reconciliation", ""])
    lines.append(df_to_markdown(error_mass))
    lines.extend(
        [
            "",
            "Interpretation: median/purge postprocessing can only address boundary/short-island style errors. Long substitutions remain a poor target for adaptive smoothing unless offset disagreement shows recoverable alternatives inside those spans.",
            "",
            "## GO / NO-GO",
        ]
    )
    go_reasons = []
    for _, row in oracle_agg.iterrows():
        if max(row["delta_f1@50"], row["delta_edit"]) >= 1.0:
            go_reasons.append(f"{row['dataset']}: per-case oracle window/purge >= 1 point")
    if not boundary.empty:
        for dataset, g in boundary.groupby("dataset"):
            pre = g[g["prediction_type"] == "pre_postprocess"]
            post = g[g["prediction_type"] == "post_native_argmax"]
            if not pre.empty and not post.empty and float(pre["delta_f1@50"].iloc[0] - post["delta_f1@50"].iloc[0]) >= 1.0:
                go_reasons.append(f"{dataset}: pre-postprocess boundary headroom exceeds post by >=1 F1@50")
    if not offset.empty and {"dataset", "scope", "method"}.issubset(offset.columns):
        off_all = offset[
            (offset["dataset"] == "50salads")
            & (offset["scope"].astype(str) == "all")
            & (offset["method"] == "oracle_best_offset_per_case")
        ]
        if not off_all.empty and float(off_all["delta_f1@50"].iloc[0]) >= 1.0 and float(off_all["long_substitution_offset_disagreement_mean"].iloc[0]) > 0.01:
            go_reasons.append("50salads: offset oracle >=1 F1@50 with nonzero long-span offset disagreement")
    if go_reasons:
        lines.append("GO signal for learned/adaptive postprocessor, but only for the following diagnostic reasons:")
        lines.extend([f"- {r}" for r in go_reasons])
    else:
        lines.append("NO-GO: no probe clears the requested GO criteria.")
    (out_dir / "adaptive_postprocess_probe_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--softmax-root", type=Path, default=DEFAULT_SOFTMAX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=["gtea", "50salads", "breakfast"], choices=["gtea", "50salads", "breakfast"])
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import torch

    device = torch.device("cuda" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args.out_dir)

    cases = build_cases(args.data_root, args.softmax_root, args.smoke, args.datasets)
    grouped: Dict[Tuple[str, int], List[CaseSpec]] = defaultdict(list)
    for case in cases:
        grouped[(case.dataset, case.fold)].append(case)

    cache: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    gt_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    official_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    current_argmax_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    manifest_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []

    for (dataset, fold), fold_cases in sorted(grouped.items()):
        print(f"Extracting {dataset} fold {fold}: {len(fold_cases)} cases", flush=True)
        extractor = FoldExtractor(args.data_root, dataset, fold, [c.video for c in fold_cases], device)
        for case in fold_cases:
            key = case_key(case)
            row: Dict[str, Any] = {
                "dataset": dataset,
                "fold": fold,
                "case_id": case.case_id,
                "video": case.video,
                "video_idx_fold_local": case.fold_local_idx,
                "complete_flag": False,
                "error_message": "",
            }
            try:
                item = extractor.extract(case)
                cache[key] = item
                event_list = extractor.event_list
                gt = load_gt(args.data_root, dataset, case.video, event_list)
                official = item["official"]
                current_probs_path = args.softmax_root / dataset / f"softmax_fold{fold}" / f"{case.case_id}.npy"
                current_probs = np.load(current_probs_path)
                current_argmax = current_probs.argmax(axis=0).astype(np.int32)
                if len(official) != len(gt) or len(current_argmax) != len(gt):
                    raise RuntimeError("Length mismatch among official/current/GT")
                gt_map[key] = gt
                official_map[key] = official
                current_argmax_map[key] = current_argmax

                official_path = args.out_dir / "official_predictions" / rel_case_path(case, ".npy")
                current_link = args.out_dir / "current_exported_probs_full" / rel_case_path(case, ".npy")
                pre_mean_path = args.out_dir / "pre_median_mean_softmax" / rel_case_path(case, ".npy")
                post_path = args.out_dir / "post_median_softmax" / rel_case_path(case, ".npy")
                save_npy(official_path, official.astype(np.int16))
                maybe_symlink(current_probs_path, current_link)
                save_npy(pre_mean_path, item["mean_pre"].astype(np.float16))
                if dataset != "gtea":
                    save_npy(post_path, item["post_native"].astype(np.float16))
                for off, probs in enumerate(item["offset_probs"]):
                    save_npy(args.out_dir / "pre_median_offset_softmax" / case.dataset / f"fold{case.fold}" / f"{case.case_id}_offset{off}.npy", probs.astype(np.float16))
                if dataset == "gtea":
                    pre_path = args.out_dir / "GTEA_pre_purge_predictions" / rel_case_path(case, "_pre.npy")
                    postp_path = args.out_dir / "GTEA_pre_purge_predictions" / rel_case_path(case, "_post.npy")
                    purged_path = args.out_dir / "GTEA_pre_purge_predictions" / rel_case_path(case, "_purged_segments.json")
                    save_npy(pre_path, item["pre_full"].astype(np.int16))
                    save_npy(postp_path, official.astype(np.int16))
                    ensure_parent(purged_path)
                    purged_path.write_text(json.dumps(item["purged_segments"], indent=2) + "\n")

                compare_rows.append(compare_pair(args.data_root, case, gt, "official", official, "argmax_probs_full", current_argmax))
                if dataset == "gtea":
                    compare_rows.append(compare_pair(args.data_root, case, gt, "official", official, "pre_purge", item["pre_full"]))
                else:
                    compare_rows.append(compare_pair(args.data_root, case, gt, "official", official, "pre_median", item["pre_full"]))
                if dataset == "50salads":
                    compare_rows.append(compare_pair(args.data_root, case, gt, "official", official, "mean_pre_median", item["pre_full"]))
                    for off, pred in enumerate(item["offset_preds"]):
                        compare_rows.append(compare_pair(args.data_root, case, gt, "official", official, f"offset{off}_pre_mean", pred))

                row.update(
                    {
                        "n_frames_full": item["full_len"],
                        "n_classes": item["n_classes"],
                        "sample_rate": item["sample_rate"],
                        "n_offsets": len(item["offset_probs"]),
                        "median_kernel": extractor.cfg["postprocess"]["value"] if extractor.cfg["postprocess"]["type"] == "median" else "",
                        "purge_threshold": extractor.cfg["postprocess"]["value"] if extractor.cfg["postprocess"]["type"] == "purge" else "",
                        "official_prediction_path": str(official_path),
                        "current_exported_probs_full_path": str(current_link),
                        "pre_median_mean_softmax_path": str(pre_mean_path),
                        "post_median_softmax_path": str(post_path) if dataset != "gtea" else "",
                        "complete_flag": True,
                    }
                )
            except Exception as exc:  # fail case-level, not whole run
                row["error_message"] = repr(exc)
                print(f"ERROR {dataset} fold {fold} case {case.case_id}: {exc}", flush=True)
            manifest_rows.append(row)
        del extractor
        torch.cuda.empty_cache()

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.out_dir / "cache_manifest.csv", index=False)
    complete_cases = [c for c in cases if case_key(c) in official_map]
    case_df = pd.DataFrame(compare_rows)
    case_df.to_csv(args.out_dir / "postprocess_reconciliation_by_case.csv", index=False)
    summary_df = (
        case_df.groupby(["dataset", "pair"])
        .agg(
            n_cases=("case_id", "count"),
            frame_disagreement_rate_mean=("frame_disagreement_rate", "mean"),
            frame_disagreement_rate_max=("frame_disagreement_rate", "max"),
            differing_frames_sum=("differing_frames", "sum"),
            segment_count_diff_mean=("segment_count_diff_left_minus_right", "mean"),
            short_segments_removed_leq3_sum=("short_segments_removed_left_to_right_leq3", "sum"),
            short_segments_removed_leq25_sum=("short_segments_removed_left_to_right_leq25", "sum"),
        )
        .reset_index()
    )
    summary_df.to_csv(args.out_dir / "postprocess_reconciliation_summary.csv", index=False)
    write_reconciliation_summary(args.out_dir, case_df, summary_df)

    sweep, held = sweep_windows(args.data_root, complete_cases, gt_map, official_map, cache)
    sweep.to_csv(args.out_dir / "adaptive_postprocess_window_sweep.csv", index=False)
    held.to_csv(args.out_dir / "adaptive_postprocess_foldheldout.csv", index=False)
    oracle = oracle_by_case(args.data_root, complete_cases, gt_map, official_map, cache)
    oracle.to_csv(args.out_dir / "adaptive_postprocess_oracle_by_case.csv", index=False)
    offset = offset_oracle_50salads(args.data_root, complete_cases, gt_map, official_map, cache)
    offset.to_csv(args.out_dir / "offset_oracle_50salads.csv", index=False)
    boundary = boundary_headroom(args.data_root, complete_cases, gt_map, official_map, cache)
    boundary.to_csv(args.out_dir / "boundary_headroom_pre_vs_post.csv", index=False)
    err_mass = official_error_mass(complete_cases, gt_map, official_map)
    err_mass.to_csv(args.out_dir / "official_error_mass_postprocess_route.csv", index=False)
    baseline_paper = official_baseline_vs_paper(args.data_root, complete_cases, gt_map, official_map)
    baseline_paper.to_csv(args.out_dir / "official_baseline_vs_paper.csv", index=False)
    write_probe_summary(args.out_dir, sweep, held, oracle, offset, boundary, err_mass, baseline_paper)

    if args.smoke:
        smoke = [
            "# Stage-3C0 Smoke Report",
            "",
            f"Cases: {len(complete_cases)}",
            "",
            "Reconciliation summary:",
            "",
            df_to_markdown(summary_df),
            "",
            "Probe summary path: `adaptive_postprocess_probe_summary.md`",
        ]
        (args.out_dir / "stage3c0_smoke_report.md").write_text("\n".join(smoke) + "\n")

    final = {
        "out_dir": str(args.out_dir),
        "smoke": args.smoke,
        "n_requested_cases": len(cases),
        "n_complete_cases": len(complete_cases),
        "n_manifest_rows": len(manifest),
        "complete_flag_all": bool(manifest["complete_flag"].all()) if len(manifest) else False,
        "decoder_logits": "not captured",
        "encoder_softmax_stream": "not captured",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Stage-3C0 outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
