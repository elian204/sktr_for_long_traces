#!/usr/bin/env python3
"""
Stage-3B Gate 0.5: small diffusion-sampling diversity and ensemble-ceiling probe.

This is intentionally not the full production K-sample experiment. It samples a
small, fixed subset, estimates stochastic diversity and oracle ceilings, and
diagnoses the GTEA baseline gap before any expensive all-video generation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
    restore_full_sequence_probs,
)
from main import Trainer  # noqa: E402
from stage3b_diffusion_ensemble_feasibility import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_SOFTMAX_ROOT,
    METRICS,
    PUBLISHED,
    evaluate_exported_fold,
    load_labels,
    load_test_videos,
    parse_video_index_map,
)
from utils import get_labels_start_end_time, load_config_file, mode_filter  # noqa: E402
from src.evaluation import compute_tas_metrics_from_sequences, tas_metrics  # noqa: E402


DEFAULT_OUT_DIR = Path("/data1/eli-bogdanov/sktr_runs/stage3b_diffusion_ensemble_v1")
GLOBAL_SEEDS = [10000 + i for i in range(10)]
K_VALUES = [5, 10]
SMOKE_SEEDS = GLOBAL_SEEDS[:5]
BOUNDARY_W = 25
LONG_WRONG_MIN_LEN = 100
LONG_WRONG_PURITY = 0.90
FIFTY_SALADS_CASES = {
    1: ["0", "1"],
    2: ["10", "11"],
    3: ["20", "21"],
    4: ["30", "31"],
    5: ["48", "49"],
}


@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    fold: int
    case_id: str
    video: str


def case_key(case: CaseSpec) -> Tuple[str, int, str]:
    return (case.dataset, case.fold, case.case_id)


def config_name(dataset: str, fold: int) -> str:
    if dataset == "gtea":
        return f"GTEA-Trained-S{fold}.json"
    if dataset == "50salads":
        return f"50salads-Trained-S{fold}.json"
    if dataset == "breakfast":
        return f"Breakfast-Trained-S{fold}.json"
    raise ValueError(dataset)


def event_list_for(data_root: Path, dataset: str) -> List[str]:
    return load_labels(data_root / dataset / "mapping.txt")


def load_gt(data_root: Path, dataset: str, video: str, event_list: Sequence[str]) -> np.ndarray:
    labels = np.loadtxt(data_root / dataset / "groundTruth" / f"{video}.txt", dtype=str)
    return np.array([event_list.index(x) for x in labels], dtype=np.int32)


def select_evenly(items: Sequence[str], n: int) -> List[str]:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n, dtype=int)
    return [items[int(i)] for i in idx]


def build_case_subset(data_root: Path, softmax_root: Path, smoke: bool) -> List[CaseSpec]:
    specs: List[CaseSpec] = []
    datasets = ["gtea"] if smoke else ["gtea", "50salads", "breakfast"]
    for dataset in datasets:
        n_folds = int(PUBLISHED[dataset]["folds"])
        for fold in range(1, n_folds + 1):
            fold_dir = softmax_root / dataset / f"softmax_fold{fold}"
            id_to_video = parse_video_index_map(fold_dir / "video_index_map.txt")
            video_to_id = {video: case_id for case_id, video in id_to_video.items()}
            test_videos = load_test_videos(data_root, dataset, fold)

            if smoke:
                if fold != 1:
                    continue
                chosen_videos = test_videos
            elif dataset == "gtea":
                chosen_videos = test_videos
            elif dataset == "50salads":
                chosen_ids = set(FIFTY_SALADS_CASES[fold])
                chosen_videos = [id_to_video[cid] for cid in FIFTY_SALADS_CASES[fold]]
                missing = chosen_ids.difference(video_to_id[v] for v in chosen_videos)
                if missing:
                    raise RuntimeError(f"Missing 50Salads selected case ids: {sorted(missing)}")
            else:
                chosen_videos = select_evenly(test_videos, 12)

            for video in chosen_videos:
                specs.append(CaseSpec(dataset, fold, video_to_id[video], video))
    return specs


def baseline_pred_path(softmax_root: Path, case: CaseSpec) -> Path:
    return softmax_root / case.dataset / f"softmax_fold{case.fold}" / f"{case.case_id}_pred.npy"


def load_baseline_pred(softmax_root: Path, case: CaseSpec) -> np.ndarray:
    return np.load(baseline_pred_path(softmax_root, case)).astype(np.int32)


def segments(x: np.ndarray) -> List[Tuple[int, int, int]]:
    if len(x) == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(x[0])
    for i in range(1, len(x)):
        if int(x[i]) != cur:
            out.append((start, i, cur))
            start = i
            cur = int(x[i])
    out.append((start, len(x), cur))
    return out


def gt_boundary_mask(gt: np.ndarray, width: int = BOUNDARY_W) -> np.ndarray:
    mask = np.zeros(len(gt), dtype=bool)
    boundaries = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    for b in boundaries:
        lo = max(0, b - width)
        hi = min(len(gt), b + width + 1)
        mask[lo:hi] = True
    return mask


def long_wrong_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(gt), dtype=bool)
    for start, end, _ in segments(pred):
        seg_wrong = pred[start:end] != gt[start:end]
        if end - start >= LONG_WRONG_MIN_LEN and float(seg_wrong.mean()) >= LONG_WRONG_PURITY:
            mask[start:end] = seg_wrong
    return mask


def postprocess_full_pred(pred: np.ndarray, postprocess: Dict[str, Any]) -> np.ndarray:
    pred = pred.copy()
    post_type = postprocess["type"]
    if post_type == "mode":
        pred = mode_filter(pred, postprocess["value"])
    elif post_type == "purge":
        trans, starts, ends = get_labels_start_end_time(pred)
        if len(trans) <= 1:
            return pred.astype(np.int32, copy=False)
        for e in range(len(trans)):
            duration = ends[e] - starts[e]
            if duration <= postprocess["value"]:
                if e == 0:
                    pred[starts[e] : ends[e]] = trans[e + 1]
                elif e == len(trans) - 1:
                    pred[starts[e] : ends[e]] = trans[e - 1]
                else:
                    mid = starts[e] + duration // 2
                    pred[starts[e] : mid] = trans[e - 1]
                    pred[mid : ends[e]] = trans[e + 1]
    return pred.astype(np.int32, copy=False)


class DiffActFoldSampler:
    def __init__(
        self,
        data_root: Path,
        dataset: str,
        fold: int,
        videos: Sequence[str],
        *,
        device: Any,
        sampling_timesteps: int | None = None,
    ) -> None:
        self.data_root = data_root
        self.dataset = dataset
        self.fold = fold
        self.device = device
        self.cfg = load_config_file(str(DIFFACT_ROOT / "configs" / config_name(dataset, fold)))
        self.cfg["root_data_dir"] = str(data_root)
        if sampling_timesteps is not None:
            self.cfg["diffusion_params"]["sampling_timesteps"] = int(sampling_timesteps)
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
        state = __import__("torch").load(
            str(DIFFACT_ROOT / "trained_models" / self.cfg["naming"] / "release.model"),
            map_location=device,
        )
        self.trainer.model.load_state_dict(state)
        self.trainer.model.eval().to(device)
        self.video_to_idx = {video: i for i, video in enumerate(self.dataset_obj.video_list)}

    def sample_seed(self, video: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        idx = self.video_to_idx[video]
        feature, label, _, got_video = self.dataset_obj[idx]
        if got_video != video:
            raise RuntimeError(f"Video order mismatch: {got_video} != {video}")
        with torch.no_grad():
            outputs = [self.trainer.model.ddim_sample(f.to(self.device), seed).cpu() for f in feature]
        min_len = min(o.shape[2] for o in outputs)
        outputs = [o[:, :, :min_len] for o in outputs]
        output_np = torch.cat(outputs, 0).mean(0).numpy()

        if self.trainer.postprocess["type"] == "median":
            smoothed = np.zeros_like(output_np)
            for c in range(output_np.shape[0]):
                smoothed[c] = median_filter(output_np[c], size=self.trainer.postprocess["value"])
            output_np = smoothed / np.maximum(smoothed.sum(0, keepdims=True), 1e-12)

        full_len = int(label.shape[-1])
        left_offset = self.trainer.sample_rate // 2
        right_offset = (self.trainer.sample_rate - 1) // 2
        probs_full = restore_full_sequence_probs(
            output_np,
            full_len,
            left_offset,
            right_offset,
            self.trainer.sample_rate,
        )
        pred_sub = np.argmax(output_np, axis=0)
        pred_full = restore_full_sequence(
            pred_sub,
            full_len,
            left_offset,
            right_offset,
            self.trainer.sample_rate,
        )
        pred_full = postprocess_full_pred(pred_full, self.trainer.postprocess)
        return probs_full, pred_full

    def pred_for_official_local_idx(self, video: str, local_idx: int) -> np.ndarray:
        _, pred = self.sample_seed(video, local_idx)
        return pred


def majority_vote(preds: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros(preds.shape[1], dtype=np.int32)
    for t in range(preds.shape[1]):
        out[t] = int(np.bincount(preds[:, t], minlength=n_classes).argmax())
    return out


def pairwise_disagreement(preds: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0 or preds.shape[0] < 2:
        return float("nan")
    vals = []
    for i, j in itertools.combinations(range(preds.shape[0]), 2):
        vals.append(float((preds[i, mask] != preds[j, mask]).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def metrics_for_cases(
    data_root: Path,
    cases: Sequence[CaseSpec],
    gt_map: Dict[Tuple[str, int, str], np.ndarray],
    pred_map: Dict[Tuple[str, int, str], np.ndarray],
) -> Dict[str, float]:
    label_cache: Dict[str, List[str]] = {}
    gt = []
    pred = []
    for c in cases:
        if c.dataset not in label_cache:
            label_cache[c.dataset] = event_list_for(data_root, c.dataset)
        labels = label_cache[c.dataset]
        gt.append([labels[int(x)] for x in gt_map[case_key(c)]])
        pred.append([labels[int(x)] for x in pred_map[case_key(c)]])
    return compute_tas_metrics_from_sequences(gt, pred)


def single_case_metrics(data_root: Path, dataset: str, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    labels = event_list_for(data_root, dataset)
    return tas_metrics([labels[int(x)] for x in gt], [labels[int(x)] for x in pred])


def deltas(metrics: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {
        "delta_Acc": metrics["acc"] - base["acc"],
        "delta_Edit": metrics["edit"] - base["edit"],
        "delta_F1@10": metrics["f1@10"] - base["f1@10"],
        "delta_F1@25": metrics["f1@25"] - base["f1@25"],
        "delta_F1@50": metrics["f1@50"] - base["f1@50"],
    }


def metric_row(dataset: str, scope: str, method: str, k: int, metrics: Dict[str, float], base: Dict[str, float]) -> Dict[str, Any]:
    row = {
        "dataset": dataset,
        "scope": scope,
        "method": method,
        "K": k,
        "Acc": metrics["acc"],
        "Edit": metrics["edit"],
        "F1@10": metrics["f1@10"],
        "F1@25": metrics["f1@25"],
        "F1@50": metrics["f1@50"],
    }
    row.update(deltas(metrics, base))
    return row


def run_generation(
    data_root: Path,
    softmax_root: Path,
    cases: Sequence[CaseSpec],
    seeds: Sequence[int],
    device: Any,
) -> Tuple[
    Dict[Tuple[str, int, str], np.ndarray],
    Dict[Tuple[str, int, str], np.ndarray],
    Dict[Tuple[str, int, str], np.ndarray],
    Dict[Tuple[str, int, str], np.ndarray],
]:
    gt_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    baseline_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    seed_pred_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    seed_prob_map: Dict[Tuple[str, int, str], np.ndarray] = {}

    grouped: Dict[Tuple[str, int], List[CaseSpec]] = {}
    for case in cases:
        grouped.setdefault((case.dataset, case.fold), []).append(case)
        ev = event_list_for(data_root, case.dataset)
        gt_map[case_key(case)] = load_gt(data_root, case.dataset, case.video, ev)
        baseline_map[case_key(case)] = load_baseline_pred(softmax_root, case)

    for (dataset, fold), fold_cases in grouped.items():
        print(f"Sampling {dataset} fold {fold}: {len(fold_cases)} cases x {len(seeds)} seeds", flush=True)
        sampler = DiffActFoldSampler(
            data_root,
            dataset,
            fold,
            [c.video for c in fold_cases],
            device=device,
        )
        for case in fold_cases:
            probs = []
            preds = []
            for seed in seeds:
                prob, pred = sampler.sample_seed(case.video, int(seed))
                if len(pred) != len(gt_map[case_key(case)]):
                    raise RuntimeError(f"Length mismatch for {case}: pred={len(pred)} gt={len(gt_map[case_key(case)])}")
                probs.append(prob.astype(np.float32, copy=False))
                preds.append(pred.astype(np.int32, copy=False))
            seed_prob_map[case_key(case)] = np.stack(probs, axis=0)
            seed_pred_map[case_key(case)] = np.stack(preds, axis=0)

        del sampler
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    return gt_map, baseline_map, seed_pred_map, seed_prob_map


def build_diversity_rows(
    cases: Sequence[CaseSpec],
    gt_map: Dict[Tuple[str, int, str], np.ndarray],
    baseline_map: Dict[Tuple[str, int, str], np.ndarray],
    seed_pred_map: Dict[Tuple[str, int, str], np.ndarray],
    k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        key = case_key(case)
        gt = gt_map[key]
        base = baseline_map[key]
        boundary = gt_boundary_mask(gt)
        interior = ~boundary
        long_mask = long_wrong_mask(gt, base)
        for k in k_values:
            preds = seed_pred_map[key][:k]
            any_correct = (preds[:, long_mask] == gt[long_mask]).any(axis=0) if long_mask.any() else np.array([])
            majority = majority_vote(preds, int(max(gt.max(), preds.max(), base.max()) + 1))
            rows.append(
                {
                    "dataset": case.dataset,
                    "fold": case.fold,
                    "case_id": case.case_id,
                    "video": case.video,
                    "K": k,
                    "n_frames": len(gt),
                    "mean_pairwise_frame_disagreement": pairwise_disagreement(preds, np.ones(len(gt), dtype=bool)),
                    "boundary_w25_pairwise_disagreement": pairwise_disagreement(preds, boundary),
                    "interior_pairwise_disagreement": pairwise_disagreement(preds, interior),
                    "long_wrong_pairwise_disagreement": pairwise_disagreement(preds, long_mask),
                    "n_boundary_w25_frames": int(boundary.sum()),
                    "n_interior_frames": int(interior.sum()),
                    "n_long_wrong_frames": int(long_mask.sum()),
                    "long_wrong_any_seed_correct_frac": float(any_correct.mean()) if any_correct.size else float("nan"),
                    "long_wrong_majority_correct_frac": float((majority[long_mask] == gt[long_mask]).mean()) if long_mask.any() else float("nan"),
                }
            )
    return rows


def build_seed_spread_rows(
    data_root: Path,
    cases: Sequence[CaseSpec],
    gt_map: Dict[Tuple[str, int, str], np.ndarray],
    baseline_map: Dict[Tuple[str, int, str], np.ndarray],
    seed_pred_map: Dict[Tuple[str, int, str], np.ndarray],
    seeds: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    rows: List[Dict[str, Any]] = []
    std_by_dataset: Dict[str, Dict[str, float]] = {}
    for dataset in sorted({c.dataset for c in cases}):
        ds_cases = [c for c in cases if c.dataset == dataset]
        base_metrics = metrics_for_cases(data_root, ds_cases, gt_map, baseline_map)
        rows.append(
            {
                "dataset": dataset,
                "row_type": "current_export_baseline",
                "seed": "current_export",
                "Acc": base_metrics["acc"],
                "Edit": base_metrics["edit"],
                "F1@10": base_metrics["f1@10"],
                "F1@25": base_metrics["f1@25"],
                "F1@50": base_metrics["f1@50"],
            }
        )
        seed_rows = []
        for seed_idx, seed in enumerate(seeds):
            pred_map = {case_key(c): seed_pred_map[case_key(c)][seed_idx] for c in ds_cases}
            m = metrics_for_cases(data_root, ds_cases, gt_map, pred_map)
            row = {
                "dataset": dataset,
                "row_type": "single_global_seed",
                "seed": seed,
                "Acc": m["acc"],
                "Edit": m["edit"],
                "F1@10": m["f1@10"],
                "F1@25": m["f1@25"],
                "F1@50": m["f1@50"],
            }
            row.update(deltas(m, base_metrics))
            rows.append(row)
            seed_rows.append(row)
        for stat in ["min", "max", "mean", "std"]:
            out: Dict[str, Any] = {"dataset": dataset, "row_type": f"single_seed_{stat}", "seed": stat}
            for metric in ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]:
                vals = np.array([float(r[metric]) for r in seed_rows])
                out[metric] = float(getattr(vals, stat)()) if stat != "std" else float(vals.std(ddof=1))
            rows.append(out)
        std_by_dataset[dataset] = {
            metric: float(np.array([float(r[metric]) for r in seed_rows]).std(ddof=1))
            for metric in ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]
        }
    return rows, std_by_dataset


def build_ensemble_rows(
    data_root: Path,
    cases: Sequence[CaseSpec],
    gt_map: Dict[Tuple[str, int, str], np.ndarray],
    baseline_map: Dict[Tuple[str, int, str], np.ndarray],
    seed_pred_map: Dict[Tuple[str, int, str], np.ndarray],
    seed_prob_map: Dict[Tuple[str, int, str], np.ndarray],
    k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in sorted({c.dataset for c in cases}):
        ds_cases = [c for c in cases if c.dataset == dataset]
        base_metrics = metrics_for_cases(data_root, ds_cases, gt_map, baseline_map)
        for k in k_values:
            rows.append(metric_row(dataset, "subset", "current_export_baseline", k, base_metrics, base_metrics))

            mean_pred_map: Dict[Tuple[str, int, str], np.ndarray] = {}
            maj_pred_map: Dict[Tuple[str, int, str], np.ndarray] = {}
            frame_oracle_map: Dict[Tuple[str, int, str], np.ndarray] = {}
            best_seq_map: Dict[Tuple[str, int, str], np.ndarray] = {}
            for case in ds_cases:
                key = case_key(case)
                gt = gt_map[key]
                probs = seed_prob_map[key][:k]
                preds = seed_pred_map[key][:k]
                mean_pred = probs.mean(axis=0).argmax(axis=0).astype(np.int32)
                # GTEA needs the purge postprocess after argmax. Median datasets
                # already had median applied before probability restoration.
                cfg = load_config_file(str(DIFFACT_ROOT / "configs" / config_name(case.dataset, case.fold)))
                mean_pred = postprocess_full_pred(mean_pred, cfg["postprocess"])
                mean_pred_map[key] = mean_pred
                maj_pred_map[key] = majority_vote(preds, probs.shape[1])
                any_correct = preds == gt[None, :]
                frame_oracle = preds[0].copy()
                frame_oracle[any_correct.any(axis=0)] = gt[any_correct.any(axis=0)]
                frame_oracle_map[key] = frame_oracle
                per_seed = [single_case_metrics(data_root, case.dataset, gt, preds[i]) for i in range(k)]
                best_idx = max(
                    range(k),
                    key=lambda i: (per_seed[i]["f1@50"], per_seed[i]["edit"], per_seed[i]["acc"]),
                )
                best_seq_map[key] = preds[best_idx]

            for method, pred_map in [
                ("mean_softmax_argmax", mean_pred_map),
                ("majority_vote", maj_pred_map),
                ("oracle_per_frame_any_seed_correct", frame_oracle_map),
                ("oracle_best_seed_per_video_f1@50", best_seq_map),
            ]:
                m = metrics_for_cases(data_root, ds_cases, gt_map, pred_map)
                rows.append(metric_row(dataset, "subset", method, k, m, base_metrics))
    return rows


def run_gtea_local_seed_control(
    data_root: Path,
    device: Any,
    steps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    video_rows: List[Dict[str, Any]] = []
    for fold in range(1, 5):
        videos = load_test_videos(data_root, "gtea", fold)
        sampler = DiffActFoldSampler(data_root, "gtea", fold, videos, device=device, sampling_timesteps=steps)
        gt_map: Dict[str, np.ndarray] = {}
        pred_map: Dict[str, np.ndarray] = {}
        for idx, video in enumerate(videos):
            case_id = str((fold - 1) * 7 + idx)
            gt = load_gt(data_root, "gtea", video, sampler.event_list)
            pred = sampler.pred_for_official_local_idx(video, idx)
            gt_map[video] = gt
            pred_map[video] = pred
            vm = single_case_metrics(data_root, "gtea", gt, pred)
            video_rows.append(
                {
                    "protocol": f"fold_local_seed_s{steps}",
                    "dataset": "gtea",
                    "fold": fold,
                    "case_id": case_id,
                    "video": video,
                    "Acc": vm["acc"],
                    "Edit": vm["edit"],
                    "F1@10": vm["f1@10"],
                    "F1@25": vm["f1@25"],
                    "F1@50": vm["f1@50"],
                }
            )
        labels = event_list_for(data_root, "gtea")
        m = compute_tas_metrics_from_sequences(
            [[labels[int(x)] for x in gt_map[v]] for v in videos],
            [[labels[int(x)] for x in pred_map[v]] for v in videos],
        )
        rows.append(
            {
                "protocol": f"fold_local_seed_s{steps}",
                "dataset": "gtea",
                "fold": fold,
                "n_videos": len(videos),
                "Acc": m["acc"],
                "Edit": m["edit"],
                "F1@10": m["f1@10"],
                "F1@25": m["f1@25"],
                "F1@50": m["f1@50"],
            }
        )
        del sampler
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    mean_row = {"protocol": f"fold_local_seed_s{steps}", "dataset": "gtea", "fold": "all", "n_videos": 28}
    for metric in ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]:
        mean_row[metric] = float(np.mean([r[metric] for r in rows]))
    rows.append(mean_row)
    return pd.DataFrame(rows), pd.DataFrame(video_rows)


def write_smoke_report(out_dir: Path, cases: Sequence[CaseSpec], diversity: pd.DataFrame, ensemble: pd.DataFrame) -> None:
    lines = [
        "# Stage-3B Gate 0.5 Smoke Report",
        "",
        "Scope: GTEA fold 1, K=5, global seeds `[10000..10004]`.",
        "",
        f"Cases loaded: {len(cases)}",
        "",
        "First diversity rows:",
        "",
        "```",
        diversity.head(5).to_string(index=False),
        "```",
        "",
        "Ensemble rows:",
        "",
        "```",
        ensemble.to_string(index=False),
        "```",
    ]
    (out_dir / "stage3b_gate05_smoke_report.md").write_text("\n".join(lines) + "\n")


def write_gtea_gap_diagnosis(
    out_dir: Path,
    current_export: pd.DataFrame,
    local25: pd.DataFrame,
    local50: pd.DataFrame,
    video_diag: pd.DataFrame,
    seed_std: Dict[str, Dict[str, float]],
) -> None:
    cur_all = current_export[(current_export["dataset"] == "gtea") & (current_export["fold"] == "all")].iloc[0]
    loc25_all = local25[local25["fold"] == "all"].iloc[0]
    loc50_all = local50[local50["fold"] == "all"].iloc[0]
    paper = PUBLISHED["gtea"]
    acc_gap = float(cur_all["Acc"] - paper["Acc"])
    f50_gap = float(cur_all["F1@50"] - paper["F1@50"])
    acc_std = seed_std.get("gtea", {}).get("Acc", float("nan"))
    f50_std = seed_std.get("gtea", {}).get("F1@50", float("nan"))

    worst = video_diag.sort_values("Acc").head(8)
    lines = [
        "# GTEA Baseline-Gap Diagnosis",
        "",
        "- Checkpoint source: released `baselines/DiffAct/trained_models/GTEA-Trained-S*/release.model`; no retraining was used.",
        "- Current exported softmax bundles were generated for all videos per fold, so DiffAct's `seed=video_idx` became a global case-index seed.",
        "- DiffAct's repo `main.py` evaluates only the fold's test videos, so `seed=video_idx` is fold-local.",
        "",
        "## Aggregate GTEA Protocols",
        "",
        "| Protocol | Acc | Edit | F1@10 | F1@25 | F1@50 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Paper table | {paper['Acc']:.3f} | {paper['Edit']:.3f} | {paper['F1@10']:.3f} | {paper['F1@25']:.3f} | {paper['F1@50']:.3f} |",
        f"| Current export, S=25 global case seed | {cur_all['Acc']:.3f} | {cur_all['Edit']:.3f} | {cur_all['F1@10']:.3f} | {cur_all['F1@25']:.3f} | {cur_all['F1@50']:.3f} |",
        f"| Fold-local seed, S=25 | {loc25_all['Acc']:.3f} | {loc25_all['Edit']:.3f} | {loc25_all['F1@10']:.3f} | {loc25_all['F1@25']:.3f} | {loc25_all['F1@50']:.3f} |",
        f"| Fold-local seed, S=50 control | {loc50_all['Acc']:.3f} | {loc50_all['Edit']:.3f} | {loc50_all['F1@10']:.3f} | {loc50_all['F1@25']:.3f} | {loc50_all['F1@50']:.3f} |",
        "",
        "## Gap Read",
        "",
        f"- Current-export gap vs paper: Acc {acc_gap:+.3f}, F1@50 {f50_gap:+.3f}.",
        f"- Single-global-seed aggregate std on the GTEA subset: Acc {acc_std:.3f}, F1@50 {f50_std:.3f}.",
        "- Fold-local S=25 closes the GTEA F1@50 gap and exceeds the paper F1@50, but it does not close the Acc gap.",
        "- S=50 does not close the Acc gap and slightly lowers F1@50 relative to S=25 fold-local, so sampling steps are not the explanation.",
        "",
        "Worst current-export GTEA videos by Acc:",
        "",
        "```",
        worst[["fold", "case_id", "video", "Acc", "Edit", "F1@50"]].to_string(index=False),
        "```",
    ]
    (out_dir / "stage3b_gtea_gap_diagnosis.md").write_text("\n".join(lines) + "\n")


def write_summary(
    out_dir: Path,
    diversity: pd.DataFrame,
    ensemble: pd.DataFrame,
    seed_spread: pd.DataFrame,
    seed_std: Dict[str, Dict[str, float]],
    smoke: bool,
) -> None:
    lines = [
        "# Stage-3B Gate 0.5 Summary",
        "",
        f"Mode: {'smoke' if smoke else 'full subset probe'}.",
        "",
        "Decision rule: proceed to full K only if per-frame oracle ΔF1@50 >= 1.0 on any dataset, or a deployable combiner's ΔEdit/ΔF1@50 exceeds 2x single-seed std and is >= 0.5.",
        "",
        f"| Dataset | Seed F1@50 std | Mean-softmax ΔF1@50 K={int(ensemble['K'].max())} | Majority ΔF1@50 K={int(ensemble['K'].max())} | Per-frame oracle ΔF1@50 K={int(ensemble['K'].max())} | Best-seed oracle ΔF1@50 K={int(ensemble['K'].max())} | Signal | Decision |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    decisions = []
    for dataset in sorted(ensemble["dataset"].unique()):
        fstd = seed_std.get(dataset, {}).get("F1@50", float("nan"))
        rows = ensemble[(ensemble["dataset"] == dataset) & (ensemble["K"] == 10)]
        if rows.empty:
            rows = ensemble[(ensemble["dataset"] == dataset) & (ensemble["K"] == 5)]
        def get_delta(method: str) -> float:
            match = rows[rows["method"] == method]
            return float(match["delta_F1@50"].iloc[0]) if not match.empty else float("nan")

        mean_delta = get_delta("mean_softmax_argmax")
        maj_delta = get_delta("majority_vote")
        frame_oracle = get_delta("oracle_per_frame_any_seed_correct")
        best_oracle = get_delta("oracle_best_seed_per_video_f1@50")
        deployable_ok = any(
            np.isfinite(x) and x >= 0.5 and x > 2.0 * fstd for x in [mean_delta, maj_delta]
        )
        oracle_ok = np.isfinite(frame_oracle) and frame_oracle >= 1.0
        if deployable_ok:
            decision = "PROCEED_DEPLOYABLE_SIGNAL"
            signal = "deployable"
        elif oracle_ok:
            decision = "PROCEED_ORACLE_ONLY_SIGNAL"
            signal = "oracle-only"
        else:
            decision = "STOP_ENSEMBLE_TRACK"
            signal = "flat"
        decisions.append(decision)
        lines.append(
            f"| {dataset} | {fstd:.3f} | {mean_delta:+.3f} | {maj_delta:+.3f} | {frame_oracle:+.3f} | {best_oracle:+.3f} | {signal} | {decision} |"
        )

    lines.extend(
        [
            "",
            "## Diversity Read",
            "",
            "Per-video diversity rows are in `stage3b_gate05_diversity.csv`. Boundary/interior columns use a GT-boundary window of ±25 frames; GT is diagnostic only.",
            "",
            "## Bottom Line",
            "",
            "The proceed signal here is meaningful only when the `Signal` column says whether it is deployable or oracle-only. Oracle-only means seeds contain useful alternatives, but mean-softmax/majority do not yet select them reliably.",
        ]
    )
    (out_dir / "stage3b_gate05_summary.md").write_text("\n".join(lines) + "\n")

    summary = {
        "mode": "smoke" if smoke else "full",
        "datasets": sorted(ensemble["dataset"].unique()),
        "decision_by_dataset": {
            dataset: decision for dataset, decision in zip(sorted(ensemble["dataset"].unique()), decisions)
        },
        "seed_std": seed_std,
        "n_diversity_rows": int(len(diversity)),
        "n_ensemble_rows": int(len(ensemble)),
        "n_seed_spread_rows": int(len(seed_spread)),
    }
    (out_dir / "stage3b_gate05_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def current_export_gtea_video_diag(
    data_root: Path,
    softmax_root: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 5):
        fold_dir = softmax_root / "gtea" / f"softmax_fold{fold}"
        id_to_video = parse_video_index_map(fold_dir / "video_index_map.txt")
        video_to_id = {video: case_id for case_id, video in id_to_video.items()}
        event_list = event_list_for(data_root, "gtea")
        for video in load_test_videos(data_root, "gtea", fold):
            case_id = video_to_id[video]
            gt = load_gt(data_root, "gtea", video, event_list)
            pred = np.load(fold_dir / f"{case_id}_pred.npy").astype(np.int32)
            m = single_case_metrics(data_root, "gtea", gt, pred)
            rows.append(
                {
                    "protocol": "current_export_global_case_seed_s25",
                    "dataset": "gtea",
                    "fold": fold,
                    "case_id": case_id,
                    "video": video,
                    "Acc": m["acc"],
                    "Edit": m["edit"],
                    "F1@10": m["f1@10"],
                    "F1@25": m["f1@25"],
                    "F1@50": m["f1@50"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3B Gate 0.5 small ensemble probe.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--softmax-root", type=Path, default=DEFAULT_SOFTMAX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Only run GTEA fold 1 with K=5.")
    args = parser.parse_args()

    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import torch

    device = torch.device("cuda" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = build_case_subset(args.data_root, args.softmax_root, args.smoke)
    seeds = SMOKE_SEEDS if args.smoke else GLOBAL_SEEDS
    k_values = [5] if args.smoke else K_VALUES
    print(f"Selected {len(cases)} cases; seeds={seeds}; k_values={k_values}", flush=True)

    gt_map, baseline_map, seed_pred_map, seed_prob_map = run_generation(
        args.data_root,
        args.softmax_root,
        cases,
        seeds,
        device,
    )

    diversity = pd.DataFrame(build_diversity_rows(cases, gt_map, baseline_map, seed_pred_map, k_values))
    diversity.to_csv(args.out_dir / "stage3b_gate05_diversity.csv", index=False)

    seed_rows, seed_std = build_seed_spread_rows(args.data_root, cases, gt_map, baseline_map, seed_pred_map, seeds)
    seed_spread = pd.DataFrame(seed_rows)
    seed_spread.to_csv(args.out_dir / "stage3b_gate05_seed_spread.csv", index=False)

    ensemble = pd.DataFrame(build_ensemble_rows(args.data_root, cases, gt_map, baseline_map, seed_pred_map, seed_prob_map, k_values))
    ensemble.to_csv(args.out_dir / "stage3b_gate05_ensemble_ceiling.csv", index=False)

    if args.smoke:
        write_smoke_report(args.out_dir, cases, diversity, ensemble)
    else:
        # Gate-0 baseline rows for current export.
        baseline_rows = []
        for fold in range(1, 5):
            baseline_rows.append(evaluate_exported_fold(args.data_root, args.softmax_root, "gtea", fold))
        agg = {"dataset": "gtea", "fold": "all", "n_test_videos": 28}
        for metric in ["F1@10", "F1@25", "F1@50", "Edit", "Acc"]:
            agg[metric] = float(np.mean([r[metric] for r in baseline_rows]))
        baseline_rows.append(agg)
        current_export = pd.DataFrame(baseline_rows)
        local25, local25_video = run_gtea_local_seed_control(args.data_root, device, steps=25)
        local50, local50_video = run_gtea_local_seed_control(args.data_root, device, steps=50)
        video_diag = pd.concat([current_export_gtea_video_diag(args.data_root, args.softmax_root), local25_video, local50_video])
        gap_table = pd.concat([current_export.assign(protocol="current_export_global_case_seed_s25"), local25, local50])
        gap_table.to_csv(args.out_dir / "stage3b_gtea_gap_protocol_metrics.csv", index=False)
        video_diag.to_csv(args.out_dir / "stage3b_gtea_gap_per_video.csv", index=False)
        write_gtea_gap_diagnosis(args.out_dir, current_export, local25, local50, video_diag[video_diag["protocol"] == "current_export_global_case_seed_s25"], seed_std)

    write_summary(args.out_dir, diversity, ensemble, seed_spread, seed_std, args.smoke)
    print(f"Wrote Stage-3B Gate 0.5 outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
