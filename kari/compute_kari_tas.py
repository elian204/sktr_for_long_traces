import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Ensure sktr_for_long_traces/src is on the path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SKTR_SRC = REPO_ROOT / "src"
if str(SKTR_SRC) not in sys.path:
    sys.path.insert(0, str(SKTR_SRC))

# Import prepare_df and TAS metrics from sktr
from utils import prepare_df
from evaluation import compute_tas_metrics_from_sequences


def load_ground_truth(dataset: str, df=None) -> Tuple[Dict[int, List[str]], List[int]]:
    """
    Load ground-truth sequences for the given dataset using prepare_df.

    Returns
    -------
    gt_sequences_by_case : dict[int, list[str]]
        Mapping from case id (case:concept:name) to the ground-truth label sequence.
    sorted_case_ids : list[int]
        Sorted list of case ids to preserve deterministic ordering.
    """
    if df is None:
        df, _, _ = prepare_df(dataset, return_mapping=True)
    gt_sequences_by_case: Dict[int, List[str]] = {}
    for case_id, group in df.groupby("case:concept:name"):
        gt_sequences_by_case[int(case_id)] = group["concept:name"].astype(str).tolist()
    sorted_case_ids = sorted(gt_sequences_by_case.keys())
    return gt_sequences_by_case, sorted_case_ids


def load_predictions(pred_pkl: Path) -> List[Dict]:
    """
    Load KARI predictions from a pickle file.

    Expected format: an iterable of dicts with keys:
      - 'video_id': case identifier (should correspond to case:concept:name)
      - 'labels': numpy array or list of predicted labels
    """
    with open(pred_pkl, "rb") as f:
        data = pickle.load(f)

    entries = list(data.values()) if isinstance(data, dict) else list(data)

    parsed_entries: List[Dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Prediction entry must be a dict with keys 'video_id' and 'labels'.")
        if "video_id" not in entry or "labels" not in entry:
            raise ValueError("Each prediction dict must contain 'video_id' and 'labels'.")

        labels = entry["labels"]
        labels_arr = np.asarray(labels)
        if labels_arr.ndim != 1:
            raise ValueError(
                f"Prediction labels must be 1D. Got shape {labels_arr.shape} for entry with video_id={entry.get('video_id')}"
            )

        parsed_entries.append(
            {
                "video_id": int(entry["video_id"]),
                "labels": labels_arr.astype(str).tolist(),
            }
        )

    return parsed_entries


def align_and_verify(
    gt_by_case: Dict[int, List[str]],
    pred_by_case: Dict[int, List[str]],
    case_order: Sequence[int],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Align predictions to ground truth by case id and verify length consistency.

    Raises
    ------
    ValueError
        If there are missing cases or length mismatches.
    """
    missing_in_pred = [cid for cid in case_order if cid not in pred_by_case]
    extra_in_pred = [cid for cid in pred_by_case if cid not in gt_by_case]
    errors = []
    if missing_in_pred:
        errors.append(f"Missing predictions for cases: {missing_in_pred[:5]} (total {len(missing_in_pred)})")
    if extra_in_pred:
        errors.append(f"Predictions contain unknown cases: {extra_in_pred[:5]} (total {len(extra_in_pred)})")
    if errors:
        raise ValueError("; ".join(errors))

    aligned_gt: List[List[str]] = []
    aligned_pred: List[List[str]] = []
    length_mismatches = []

    for cid in case_order:
        gt_seq = gt_by_case[cid]
        pred_seq = pred_by_case[cid]
        if len(gt_seq) != len(pred_seq):
            length_mismatches.append((cid, len(pred_seq), len(gt_seq)))
        aligned_gt.append(gt_seq)
        aligned_pred.append(pred_seq)

    if length_mismatches:
        msg_lines = ["Length mismatches detected (case, pred_len, gt_len):"]
        msg_lines += [f"  {cid}: pred={p}, gt={g}" for cid, p, g in length_mismatches[:10]]
        raise ValueError("\n".join(msg_lines))

    return aligned_gt, aligned_pred


def remap_with_test_matrices(
    pred_entries: List[Dict],
    gt_softmax_list: Sequence[np.ndarray],
    test_matrices_path: Path,
) -> Dict[int, List[str]]:
    """
    Remap prediction case ids by comparing test matrices to ground-truth softmax matrices.

    This is useful when the stored video_id ordering differs from the ground-truth ordering.
    """
    with open(test_matrices_path, "rb") as f:
        test_mats = pickle.load(f)

    length_to_gt = {}
    for idx, arr in enumerate(gt_softmax_list):
        length_to_gt.setdefault(arr.shape[1], []).append(idx)

    remapped: Dict[int, List[str]] = {}
    used_gt = set()

    for entry in pred_entries:
        pred_id = entry["video_id"]
        labels = entry["labels"]
        try:
            mat = test_mats[pred_id]
        except Exception:
            raise ValueError(f"Cannot access test matrix at index {pred_id} for remapping.")

        T = np.asarray(mat).T  # (C, L)
        candidates = length_to_gt.get(T.shape[1], [])
        match = None
        for gt_idx in candidates:
            if np.allclose(T, gt_softmax_list[gt_idx]):
                match = gt_idx
                break
        if match is None:
            raise ValueError(f"Could not remap prediction {pred_id}: no matching ground-truth softmax found.")
        if match in used_gt:
            raise ValueError(f"Non-unique remap detected: ground-truth case {match} already matched.")

        remapped[match] = labels
        used_gt.add(match)

    if len(remapped) != len(gt_softmax_list):
        raise ValueError(f"Remapping incomplete: mapped {len(remapped)} of {len(gt_softmax_list)} cases.")

    return remapped


def main():
    parser = argparse.ArgumentParser(description="Compute TAS metrics for KARI predictions on 50salads.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to KARI predictions pickle file (contains 'video_id' and 'labels').",
    )
    parser.add_argument(
        "--dataset",
        default="50salads",
        help="Dataset name to load with prepare_df (default: 50salads).",
    )
    parser.add_argument(
        "--test-matrices",
        type=Path,
        default=REPO_ROOT / "kari" / "test_matrices_50salads_complete.pkl",
        help="Path to the test matrices used to generate KARI results (for auto-remapping when IDs are permuted).",
    )
    parser.add_argument(
        "--background",
        default=None,
        help="Optional background label to ignore for Edit/F1 (default: auto-detect).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--allow-remap",
        action="store_true",
        help="If set, attempt to remap case IDs using test matrices when lengths mismatch. "
             "Otherwise, mismatches raise an error.",
    )
    args = parser.parse_args()

    metrics = compute_kari_tas(
        predictions_path=args.predictions,
        dataset=args.dataset,
        background=args.background,
        allow_remap=args.allow_remap,
        test_matrices_path=args.test_matrices,
        verbose=not args.json,
    )
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print("\nTAS metrics")
        print("-" * 40)
        for k, v in metrics.items():
            print(f"{k:10s}: {v:.2f}")


def compute_kari_tas(
    predictions_path: Path,
    *,
    dataset: str = "50salads",
    background=None,
    allow_remap: bool = False,
    test_matrices_path: Path = REPO_ROOT / "kari" / "test_matrices_50salads_complete.pkl",
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute TAS metrics for KARI predictions from a pickle file.

    Parameters
    ----------
    predictions_path : Path
        Path to pickle containing KARI outputs with keys 'video_id' and 'labels'.
    dataset : str
        Dataset name to load via prepare_df (default: '50salads').
    background : Any, optional
        Background label to ignore; None triggers auto-detection.
    allow_remap : bool
        If True, attempt remapping via test matrices when length mismatches occur.
    test_matrices_path : Path
        Path to the test matrices used to produce the predictions (needed for remap).
    verbose : bool
        If True, prints when remapping is attempted.
    """
    df, gt_softmax_list, _ = prepare_df(dataset, return_mapping=True)
    gt_by_case, case_order = load_ground_truth(dataset, df=df)
    pred_entries = load_predictions(predictions_path)
    pred_by_case = {e["video_id"]: e["labels"] for e in pred_entries}

    try:
        aligned_gt, aligned_pred = align_and_verify(gt_by_case, pred_by_case, case_order)
    except ValueError as exc:
        if allow_remap and test_matrices_path and Path(test_matrices_path).exists():
            if verbose:
                print("Direct alignment failed; attempting remap using test matrices...")
            remapped_pred = remap_with_test_matrices(pred_entries, gt_softmax_list, test_matrices_path)
            aligned_gt, aligned_pred = align_and_verify(gt_by_case, remapped_pred, case_order)
        else:
            raise exc

    return compute_tas_metrics_from_sequences(
        gt_sequences=aligned_gt,
        pred_sequences=aligned_pred,
        background=background,
    )


if __name__ == "__main__":
    main()
