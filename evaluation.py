from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from collections import defaultdict

# --- helpers -----------------------------------------------------------------
def _collapse_runs(seq: Sequence[Any], background: Optional[Any]) -> List[Any]:
    out: List[Any] = []
    last = object()
    for x in seq:
        if x != last:
            out.append(x); last = x
    if background is not None:
        out = [x for x in out if x != background]
    return out

def _levenshtein_norm(a: Sequence[Any], b: Sequence[Any]) -> float:
    m, n = len(a), len(b)
    if m == 0 and n == 0: return 1.0
    if m == 0 or n == 0:  return 0.0
    prev = np.arange(n + 1)
    cur = np.empty_like(prev)
    for i in range(1, m + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    dist = int(prev[n])
    return 1.0 - dist / max(m, n)

def _segments_from_frames(labels: np.ndarray, background: Optional[Any]) -> List[Tuple[int, int, Any]]:
    if labels.size == 0: return []
    segs: List[Tuple[int, int, Any]] = []
    s, cur = 0, labels[0]
    for t in range(1, labels.size):
        if labels[t] != cur:
            if background is None or cur != background:
                segs.append((s, t, cur))
            s, cur = t, labels[t]
    if background is None or cur != background:
        segs.append((s, labels.size, cur))
    return segs

def _iou_1d(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter / union if union > 0 else 0.0

def _auto_background(label_names: Optional[Sequence[str]]) -> Optional[str]:
    if not label_names: return None
    lower = [str(x).lower() for x in label_names]
    for cand in ("background", "bg", "sil", "silence"):
        if cand in lower:
            # return original-cased token at that index
            return label_names[lower.index(cand)]
    return None

# --- public API ---------------------------------------------------------------
def frame_accuracy(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    gt = np.asarray(list(y_true), dtype=object)
    pr = np.asarray(list(y_pred), dtype=object)
    if gt.ndim != 1 or pr.ndim != 1 or gt.shape != pr.shape:
        raise ValueError("y_true and y_pred must be 1D and equal length")
    return float((gt == pr).mean() * 100.0) if gt.size else 0.0

def edit_score(
    y_true: Iterable[Any], y_pred: Iterable[Any],
    background: Optional[Any] = None,
    label_names: Optional[Sequence[str]] = None,
) -> float:
    if background is None:
        background = _auto_background(label_names)
    a = _collapse_runs(list(y_true), background)
    b = _collapse_runs(list(y_pred), background)
    return _levenshtein_norm(a, b) * 100.0

def segmental_f1(
    y_true: Iterable[Any], y_pred: Iterable[Any], iou_threshold: float,
    background: Optional[Any] = None,
    label_names: Optional[Sequence[str]] = None,
) -> float:
    if background is None:
        background = _auto_background(label_names)
    gt = np.asarray(list(y_true), dtype=object)
    pr = np.asarray(list(y_pred), dtype=object)
    G = _segments_from_frames(gt, background)
    P = _segments_from_frames(pr, background)
    if not G and not P:  # standard TAS treatment: perfect agreement on "no segments"
        return 100.0

    g_by, p_by = defaultdict(list), defaultdict(list)
    for s in G: g_by[s[2]].append(s)
    for s in P: p_by[s[2]].append(s)
    TP = FP = FN = 0
    for lbl in (set(g_by) | set(p_by)):
        g, p = g_by.get(lbl, []), p_by.get(lbl, [])
        pairs = [(i, j, _iou_1d((gs, ge), (ps, pe)))
                 for i,(gs,ge,_) in enumerate(g) for j,(ps,pe,_) in enumerate(p)]
        mg, mp = set(), set()
        for i, j, v in sorted(pairs, key=lambda t: t[2], reverse=True):
            if v >= iou_threshold and i not in mg and j not in mp:
                mg.add(i); mp.add(j)
        TP += len(mg); FP += len(p) - len(mp); FN += len(g) - len(mg)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    return (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0) * 100.0

def tas_metrics(
    y_true: Iterable[Any], y_pred: Iterable[Any],
    background: Optional[Any] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if background is None:
        background = _auto_background(label_names)
    return {
        "acc":   frame_accuracy(y_true, y_pred),
        "edit":  edit_score(y_true, y_pred, background, label_names),
        "f1@10": segmental_f1(y_true, y_pred, 0.10, background, label_names),
        "f1@25": segmental_f1(y_true, y_pred, 0.25, background, label_names),
        "f1@50": segmental_f1(y_true, y_pred, 0.50, background, label_names),
    }

def compute_tas_metrics_macro(
    df: pd.DataFrame,
    pred_col: str,
    *,
    gt_col: str = "ground_truth",
    case_col: str = "case:concept:name",
    background: Optional[Any] = 0,
    label_names: Optional[Sequence[str]] = None,
    return_per_video: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
    """
    Compute dataset-level TAS metrics following the standard protocol:
      - Macro average for Edit and F1@{10,25,50} (computed per video/case).
      - Micro/global frame accuracy over all frames.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns `[case_col, gt_col, pred_col]`.
    pred_col : str
        Column with the per-frame predictions to evaluate.
    gt_col : str, default "ground_truth"
        Column with per-frame ground-truth labels.
    case_col : str, default "case:concept:name"
        Column identifying each video/sequence.
    background : Any, optional
        Label to treat as background (ignored in Edit/F1 segmentization).
        If None, `_auto_background` is used inside `tas_metrics`.
    label_names : Sequence[str], optional
        Optional label names (used for background auto-detection).
    return_per_video : bool, default False
        If True, also return the per-video metrics DataFrame.

    Returns
    -------
    dict
        {
          "acc_micro": <global frame accuracy>,
          "edit":  <macro mean>,
          "f1@10": <macro mean>,
          "f1@25": <macro mean>,
          "f1@50": <macro mean>,
        }
    (dict, DataFrame) if return_per_video=True
        The dict above and a per-video table with columns:
        ["video", "n_frames", "acc", "edit", "f1@10", "f1@25", "f1@50"].

    Notes
    -----
    - Edit/F1s are averaged across videos (macro), which is the standard in TAS.
    - Frame accuracy is computed once over all frames (micro/global).
    """
    required = {case_col, gt_col, pred_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    # --- per-video metrics (for macro Edit/F1) ---
    rows = []
    for vid, g in df.groupby(case_col, sort=False):
        y_true = g[gt_col].tolist()
        y_pred = g[pred_col].tolist()
        if not y_true:  # skip empty sequences just in case
            continue
        m = tas_metrics(y_true, y_pred, background=background, label_names=label_names)
        rows.append({"video": vid, "n_frames": len(g), **m})
    per_video_df = pd.DataFrame(rows)

    # Handle corner case: no videos (empty df)
    if per_video_df.empty:
        summary = {"acc_micro": 0.0, "edit": 0.0, "f1@10": 0.0, "f1@25": 0.0, "f1@50": 0.0}
        return (summary, per_video_df) if return_per_video else summary

    # --- macro averages for Edit/F1s ---
    macro = per_video_df[["edit", "f1@10", "f1@25", "f1@50"]].mean().to_dict()

    # --- micro/global frame accuracy ---
    acc_micro = frame_accuracy(df[gt_col].values, df[pred_col].values)

    summary = {"acc_micro": float(acc_micro), **{k: float(v) for k, v in macro.items()}}

    return (summary, per_video_df) if return_per_video else summary


def compute_tas_metrics_from_sequences(
    gt_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    background: Optional[Any] = 0,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute TAS metrics directly from ground truth and prediction sequences.

    This function takes lists of sequences (one per video/case) and computes
    aggregated metrics similar to compute_tas_metrics_macro.

    Parameters
    ----------
    gt_sequences : List[List[str]]
        List of ground truth sequences, one per video/case.
    pred_sequences : List[List[str]]
        List of prediction sequences, one per video/case.
        Must have same length as gt_sequences.
    background : Any, optional
        Label to treat as background (ignored in Edit/F1 segmentization).
        If None, `_auto_background` is used inside `tas_metrics`.
    label_names : Sequence[str], optional
        Optional label names (used for background auto-detection).

    Returns
    -------
    Dict[str, float]
        {
          "acc_micro": <global frame accuracy over all sequences>,
          "edit":  <macro mean edit score across sequences>,
          "f1@10": <macro mean F1@10 across sequences>,
          "f1@25": <macro mean F1@25 across sequences>,
          "f1@50": <macro mean F1@50 across sequences>,
        }

    Raises
    ------
    ValueError
        If gt_sequences and pred_sequences have different lengths,
        or if individual sequence pairs have different lengths.
    """
    if len(gt_sequences) != len(pred_sequences):
        raise ValueError(
            f"Number of sequences don't match: GT has {len(gt_sequences)}, "
            f"Pred has {len(pred_sequences)}"
        )

    # Compute per-sequence metrics
    per_sequence_metrics = []
    all_gt_frames = []
    all_pred_frames = []

    for i, (gt_seq, pred_seq) in enumerate(zip(gt_sequences, pred_sequences)):
        if len(gt_seq) != len(pred_seq):
            raise ValueError(
                f"Sequence {i} length mismatch: GT has {len(gt_seq)}, "
                f"Pred has {len(pred_seq)}"
            )

        # Get metrics for this sequence
        seq_metrics = tas_metrics(gt_seq, pred_seq, background=background, label_names=label_names)
        per_sequence_metrics.append(seq_metrics)

        # Collect all frames for micro accuracy
        all_gt_frames.extend(gt_seq)
        all_pred_frames.extend(pred_seq)

    # Compute macro averages for Edit/F1s
    macro_metrics = {}
    for key in ["edit", "f1@10", "f1@25", "f1@50"]:
        values = [m[key] for m in per_sequence_metrics]
        macro_metrics[key] = float(sum(values) / len(values))

    # Compute micro/global frame accuracy
    acc_micro = frame_accuracy(all_gt_frames, all_pred_frames)

    return {
        "acc_micro": float(acc_micro),
        **macro_metrics
    }