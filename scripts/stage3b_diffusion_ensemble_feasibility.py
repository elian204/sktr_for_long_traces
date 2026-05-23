#!/usr/bin/env python3
"""
Stage-3B Gate-0 for DiffAct diffusion ensembling.

This script deliberately stops at baseline honesty. It evaluates the existing
exported DiffAct predictions with the official DiffAct metric code, records the
released-code/paper protocol, and optionally checks that different diffusion
seeds change one GTEA prediction. It does not generate K-sample ensembles when
the reproduced baseline is below the paper baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DIFFACT_ROOT = REPO_ROOT / "baselines" / "DiffAct"
if str(DIFFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFACT_ROOT))

from utils import func_eval, load_config_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")
DEFAULT_SOFTMAX_ROOT = Path(
    "/data1/eli-bogdanov/sktr_runs/moved_from_home/"
    "sktr_for_long_traces/baselines/DiffAct/results"
)
DEFAULT_OUT_DIR = Path("/data1/eli-bogdanov/sktr_runs/stage3b_diffusion_ensemble_v1")

METRICS = ["F1@10", "F1@25", "F1@50", "Edit", "Acc"]
PUBLISHED = {
    "gtea": {
        "folds": 4,
        "F1@10": 92.5,
        "F1@25": 91.5,
        "F1@50": 84.7,
        "Edit": 89.6,
        "Acc": 82.2,
    },
    "50salads": {
        "folds": 5,
        "F1@10": 90.1,
        "F1@25": 89.2,
        "F1@50": 83.7,
        "Edit": 85.0,
        "Acc": 88.9,
    },
    "breakfast": {
        "folds": 4,
        "F1@10": 80.3,
        "F1@25": 75.9,
        "F1@50": 64.6,
        "Edit": 78.4,
        "Acc": 76.4,
    },
}


def parse_video_index_map(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case_id, video = line.split()[:2]
            out[case_id] = video
    return out


def load_labels(mapping_file: Path) -> List[str]:
    rows = np.loadtxt(mapping_file, dtype=str)
    return [row[1] for row in rows]


def load_test_videos(data_root: Path, dataset: str, fold: int) -> List[str]:
    split_file = data_root / dataset / "splits" / f"test.split{fold}.bundle"
    videos: List[str] = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                videos.append(line.split(".")[0])
    return videos


def evaluate_exported_fold(
    data_root: Path,
    softmax_root: Path,
    dataset: str,
    fold: int,
) -> Dict[str, Any]:
    """Evaluate saved {case_id}_pred.npy using DiffAct's own func_eval."""
    labels = load_labels(data_root / dataset / "mapping.txt")
    fold_dir = softmax_root / dataset / f"softmax_fold{fold}"
    id_to_video = parse_video_index_map(fold_dir / "video_index_map.txt")
    video_to_id = {video: case_id for case_id, video in id_to_video.items()}
    test_videos = load_test_videos(data_root, dataset, fold)

    tmp = Path(tempfile.mkdtemp(prefix=f"stage3b_{dataset}_f{fold}_"))
    pred_dir = tmp / "prediction"
    pred_dir.mkdir(parents=True)
    try:
        for video in test_videos:
            case_id = video_to_id[video]
            pred_path = fold_dir / f"{case_id}_pred.npy"
            pred = np.load(pred_path)
            pred_labels = [labels[int(x)] for x in pred]
            with open(pred_dir / f"{video}.txt", "w") as f:
                f.write("### Frame level recognition: ###\n")
                f.write(" ".join(pred_labels))

        # DiffAct's released metric code still uses np.float. Keep behavior but
        # make it runnable on modern NumPy.
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
        acc, edit, f1s = func_eval(
            str(data_root / dataset / "groundTruth"),
            str(pred_dir),
            test_videos,
        )
    finally:
        shutil.rmtree(tmp)

    return {
        "dataset": dataset,
        "fold": fold,
        "n_test_videos": len(test_videos),
        "F1@10": float(f1s[0]),
        "F1@25": float(f1s[1]),
        "F1@50": float(f1s[2]),
        "Edit": float(edit),
        "Acc": float(acc),
    }


def build_protocol_rows(data_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    config_name = {
        "gtea": "GTEA-Trained-S{fold}.json",
        "50salads": "50salads-Trained-S{fold}.json",
        "breakfast": "Breakfast-Trained-S{fold}.json",
    }
    for dataset, meta in PUBLISHED.items():
        for fold in range(1, int(meta["folds"]) + 1):
            cfg = load_config_file(str(DIFFACT_ROOT / "configs" / config_name[dataset].format(fold=fold)))
            sample_rate = int(cfg["sample_rate"])
            temporal_offsets = sample_rate if cfg["temporal_aug"] else 1
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "mode": "decoder-agg",
                    "sampling_timesteps": cfg["diffusion_params"]["sampling_timesteps"],
                    "ddim_sampling_eta": cfg["diffusion_params"]["ddim_sampling_eta"],
                    "set_sampling_seed": cfg["set_sampling_seed"],
                    "seed_rule": "video_idx" if cfg["set_sampling_seed"] else "unseeded",
                    "independent_noise_samples_per_offset": 1,
                    "temporal_offsets_averaged": temporal_offsets,
                    "sample_rate": sample_rate,
                    "temporal_aug": cfg["temporal_aug"],
                    "postprocess_type": cfg["postprocess"]["type"],
                    "postprocess_value": cfg["postprocess"]["value"],
                    "current_softmax_K_independent_diffusion_samples": 1,
                    "current_softmax_notes": (
                        "Mean over temporal offsets for decoder-agg; no independent "
                        "multi-seed sample ensemble in the released eval/export path."
                    ),
                }
            )
    return rows


def run_stochasticity_smoke(data_root: Path, device: int) -> Dict[str, Any]:
    """Run one GTEA video with two diffusion seeds and compare decoder outputs."""
    if device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    import torch  # noqa: WPS433
    from dataset import VideoFeatureDataset, get_data_dict  # noqa: WPS433
    from main import Trainer  # noqa: WPS433

    cfg = load_config_file(str(DIFFACT_ROOT / "configs" / "GTEA-Trained-S1.json"))
    cfg["root_data_dir"] = str(data_root)
    dataset = cfg["dataset_name"]
    event_rows = np.loadtxt(data_root / dataset / "mapping.txt", dtype=str)
    event_list = [row[1] for row in event_rows]
    test_videos = load_test_videos(data_root, dataset, 1)[:1]
    data_dict = get_data_dict(
        feature_dir=str(data_root / dataset / "features"),
        label_dir=str(data_root / dataset / "groundTruth"),
        video_list=test_videos,
        event_list=event_list,
        sample_rate=cfg["sample_rate"],
        temporal_aug=cfg["temporal_aug"],
        boundary_smooth=cfg["boundary_smooth"],
    )
    test_ds = VideoFeatureDataset(data_dict, len(event_list), mode="test")
    torch_device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        dict(cfg["encoder_params"]),
        dict(cfg["decoder_params"]),
        dict(cfg["diffusion_params"]),
        event_list,
        cfg["sample_rate"],
        cfg["temporal_aug"],
        cfg["set_sampling_seed"],
        cfg["postprocess"],
        device=torch_device,
    )
    state = torch.load(
        str(DIFFACT_ROOT / "trained_models" / cfg["naming"] / "release.model"),
        map_location=torch_device,
    )
    trainer.model.load_state_dict(state)
    trainer.model.eval().to(torch_device)

    feature, _, _, video = test_ds[0]
    with torch.no_grad():
        out_a = trainer.model.ddim_sample(feature[0].to(torch_device), seed=0).cpu().numpy()
        out_b = trainer.model.ddim_sample(feature[0].to(torch_device), seed=999).cpu().numpy()

    abs_diff = np.abs(out_a - out_b)
    return {
        "dataset": "gtea",
        "fold": 1,
        "video": video,
        "seed_a": 0,
        "seed_b": 999,
        "shape": list(out_a.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "argmax_diff_fraction": float((out_a.argmax(axis=1) != out_b.argmax(axis=1)).mean()),
        "stochastic_outputs_differ": bool(abs_diff.max() > 0),
    }


def write_summary(
    out_dir: Path,
    baseline: pd.DataFrame,
    protocol: pd.DataFrame,
    smoke: Dict[str, Any] | None,
    stop_gate0: bool,
    tolerance: float,
) -> None:
    aggregate = baseline[baseline["fold"] == "all"].copy()
    lines = [
        "# Stage-3B DiffAct Diffusion Ensemble Gate-0",
        "",
        "Status: **STOP at Gate 0**." if stop_gate0 else "Status: Gate 0 passed.",
        "",
        "No K-sample ensemble generation was launched. The reproduced baseline is below the paper baseline on at least one metric beyond the configured tolerance, so ensemble gains would not be claimable until the baseline gap is resolved.",
        "",
        "## Released-Code Protocol",
        "",
        "- Paper/repo inference steps: 25 DDIM sampling steps.",
        "- Sampling eta: 1.0.",
        "- Released eval mode: `decoder-agg`.",
        "- Seed handling: `set_sampling_seed=True`, seed = `video_idx`.",
        "- Independent stochastic samples averaged by the released path: 1.",
        "- Temporal-offset averaging is separate from stochastic ensembling: GTEA=1 offset, 50Salads=8 offsets, Breakfast=1 offset.",
        "- Current exported softmax follows the released `decoder-agg` path; it is not an independent multi-seed ensemble.",
        "",
        "## Baseline Reproduction vs ICCV Table",
        "",
        "| Dataset | F1@10 | F1@25 | F1@50 | Edit | Acc | Worst Gap | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            "| {dataset} | {f10:.3f} ({g10:+.3f}) | {f25:.3f} ({g25:+.3f}) | "
            "{f50:.3f} ({g50:+.3f}) | {edit:.3f} ({gedit:+.3f}) | "
            "{acc:.3f} ({gacc:+.3f}) | {worst:+.3f} | {gate} |".format(
                dataset=row["dataset"],
                f10=row["F1@10"],
                g10=row["gap_F1@10_vs_paper"],
                f25=row["F1@25"],
                g25=row["gap_F1@25_vs_paper"],
                f50=row["F1@50"],
                g50=row["gap_F1@50_vs_paper"],
                edit=row["Edit"],
                gedit=row["gap_Edit_vs_paper"],
                acc=row["Acc"],
                gacc=row["gap_Acc_vs_paper"],
                worst=row["worst_gap_vs_paper"],
                gate=row["gate0_status"],
            )
        )

    lines.extend(
        [
            "",
            f"Gate tolerance for rounded paper values: {tolerance:.3f} points. A negative gap below `-tolerance` triggers STOP.",
            "",
            "## Stochasticity Smoke",
            "",
        ]
    )
    if smoke is None:
        lines.append("Not run.")
    else:
        lines.append(
            "- {video}: seed {seed_a} vs {seed_b}, max abs diff {max_abs_diff:.6f}, "
            "mean abs diff {mean_abs_diff:.6f}, argmax diff fraction {argmax_diff_fraction:.6f}. "
            "Outputs differ: `{stochastic_outputs_differ}`.".format(**smoke)
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Stage-3B ensembling is not started from this artifact. The next action is to resolve/accept the Gate-0 baseline definition, especially GTEA F1@50/Acc and 50Salads Acc. If we decide that the released checkpoints plus official evaluator are the baseline despite the paper gaps, rerun this script's future ensemble step against that explicitly defined baseline.",
            "",
            "Generated files:",
            "",
            "- `stage3b_baseline_repro.csv`",
            "- `stage3b_protocol.csv`",
            "- `stage3b_stochasticity_smoke.json` (if smoke was run)",
            "- skipped placeholders for ensemble/rule/significance CSVs",
        ]
    )
    (out_dir / "stage3b_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3B Gate-0 baseline honesty report.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--softmax-root", type=Path, default=DEFAULT_SOFTMAX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gap-tolerance", type=float, default=0.20)
    parser.add_argument("--run-stochastic-smoke", action="store_true")
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    protocol = pd.DataFrame(build_protocol_rows(args.data_root))
    protocol.to_csv(args.out_dir / "stage3b_protocol.csv", index=False)

    rows: List[Dict[str, Any]] = []
    for dataset, meta in PUBLISHED.items():
        fold_rows = []
        for fold in range(1, int(meta["folds"]) + 1):
            row = evaluate_exported_fold(args.data_root, args.softmax_root, dataset, fold)
            fold_rows.append(row)
            rows.append(row)
        agg = {"dataset": dataset, "fold": "all", "n_test_videos": sum(r["n_test_videos"] for r in fold_rows)}
        for metric in METRICS:
            agg[metric] = float(np.mean([r[metric] for r in fold_rows]))
        rows.append(agg)

    baseline = pd.DataFrame(rows)
    for metric in METRICS:
        baseline[f"paper_{metric}"] = baseline["dataset"].map(lambda ds: PUBLISHED[ds][metric])
        baseline[f"gap_{metric}_vs_paper"] = baseline[metric] - baseline[f"paper_{metric}"]
    gap_cols = [f"gap_{metric}_vs_paper" for metric in METRICS]
    baseline["worst_gap_vs_paper"] = baseline[gap_cols].min(axis=1)
    baseline["below_paper_beyond_tolerance"] = baseline["worst_gap_vs_paper"] < -args.gap_tolerance
    baseline["gate0_status"] = np.where(
        baseline["below_paper_beyond_tolerance"],
        "FAIL_BASELINE_BELOW_PAPER",
        "PASS_OR_WITHIN_ROUNDING_TOLERANCE",
    )
    baseline.to_csv(args.out_dir / "stage3b_baseline_repro.csv", index=False)

    aggregate = baseline[baseline["fold"] == "all"]
    stop_gate0 = bool(aggregate["below_paper_beyond_tolerance"].any())

    smoke = run_stochasticity_smoke(args.data_root, args.device) if args.run_stochastic_smoke else None
    if smoke is not None:
        (args.out_dir / "stage3b_stochasticity_smoke.json").write_text(
            json.dumps(smoke, indent=2, sort_keys=True) + "\n"
        )

    skipped = pd.DataFrame(
        [
            {
                "status": "skipped_gate0_baseline_gap" if stop_gate0 else "not_run",
                "reason": "Baseline reproduction is below paper on at least one aggregate metric.",
            }
        ]
    )
    for name in [
        "stage3b_ensemble_by_K.csv",
        "stage3b_combination_rules.csv",
        "stage3b_significance.csv",
    ]:
        skipped.to_csv(args.out_dir / name, index=False)

    summary = {
        "out_dir": str(args.out_dir),
        "data_root": str(args.data_root),
        "softmax_root": str(args.softmax_root),
        "gate0_stop": stop_gate0,
        "gap_tolerance": args.gap_tolerance,
        "datasets_below_paper": aggregate.loc[
            aggregate["below_paper_beyond_tolerance"], "dataset"
        ].tolist(),
        "protocol_rows": len(protocol),
        "baseline_rows": len(baseline),
        "stochasticity_smoke": smoke,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_summary(args.out_dir, baseline, protocol, smoke, stop_gate0, args.gap_tolerance)

    print(f"Wrote Stage-3B Gate-0 report to {args.out_dir}")
    print(f"gate0_stop={stop_gate0}; datasets_below_paper={summary['datasets_below_paper']}")


if __name__ == "__main__":
    main()
