from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

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


def _normalize_background_value(background: Any) -> Any:
    if isinstance(background, (list, tuple, set, np.ndarray)):
        return [str(x) for x in background]
    return str(background)


def _read_background_from_mapping(mapping_path: Path) -> Optional[str]:
    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[1].lower() == "background":
                    return str(parts[0])
    except FileNotFoundError:
        return None
    return None


def _resolve_asformer_background(
    background: Optional[Any],
    *,
    label_names: Optional[Sequence[Any]] = None,
    sample_values: Optional[Iterable[Any]] = None,
    dataset_name: Optional[str] = None,
    mapping_path: Optional[Union[str, Path]] = None,
) -> Optional[Any]:
    if background is not None and str(background).lower() not in ("auto", "infer"):
        return _normalize_background_value(background)

    # Prefer explicit "background" label if present
    for seq in (label_names, sample_values):
        if not seq:
            continue
        for v in seq:
            if str(v).lower() == "background":
                return str(v)

    paths: List[Path] = []
    if mapping_path:
        paths.append(Path(mapping_path))
    if dataset_name:
        ds = str(dataset_name).lower()
        here = Path(__file__).resolve()
        paths.extend([
            Path.cwd() / "data" / "data" / ds / "mapping.txt",
            here.parents[2] / "data" / "data" / ds / "mapping.txt",
            Path.home() / "data" / "data" / ds / "mapping.txt",
        ])

    for p in paths:
        if p.exists():
            bg = _read_background_from_mapping(p)
            if bg is not None:
                return bg

    return None


def _asformer_bg_class(background: Optional[Any]) -> List[Any]:
    if background is None:
        return ["background"]
    if isinstance(background, (list, tuple, set, np.ndarray)):
        return list(background)
    return [background]


def _infer_dataset_name_from_path(path: str) -> Optional[str]:
    lower = str(path).lower()
    for name in ("50salads", "gtea", "breakfast"):
        if name in lower:
            return name
    return None


def _get_labels_start_end_time_asformer(
    frame_wise_labels: Sequence[Any],
    bg_class: Sequence[Any],
) -> Tuple[List[Any], List[int], List[int]]:
    labels: List[Any] = []
    starts: List[int] = []
    ends: List[int] = []
    if not frame_wise_labels:
        return labels, starts, ends
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def _edit_score_asformer(
    recognized: Iterable[Any],
    ground_truth: Iterable[Any],
    bg_class: Sequence[Any],
) -> float:
    p_label, _, _ = _get_labels_start_end_time_asformer(list(recognized), bg_class)
    y_label, _, _ = _get_labels_start_end_time_asformer(list(ground_truth), bg_class)
    return _levenshtein_norm(p_label, y_label) * 100.0


def _segmental_f1_counts_asformer(
    y_true: Iterable[Any], y_pred: Iterable[Any], iou_threshold: float,
    background: Optional[Any] = None,
) -> Tuple[int, int, int]:
    """
    Compute TP, FP, FN for segmental F1 using ASFormer/MS-TCN2's matching strategy.

    This uses per-prediction greedy matching: for each predicted segment, find the
    best matching GT segment (argmax IoU), and if it passes the threshold and hasn't
    been matched yet, count it as TP.

    This matches the implementation in ASFormer/MS-TCN2 eval.py exactly.

    Returns
    -------
    Tuple[int, int, int]
        (TP, FP, FN) counts for this sequence.
    """
    bg_class = _asformer_bg_class(background)
    y_label, y_start, y_end = _get_labels_start_end_time_asformer(list(y_true), bg_class)
    p_label, p_start, p_end = _get_labels_start_end_time_asformer(list(y_pred), bg_class)

    if not y_label and not p_label:
        return (0, 0, 0)

    y_start = np.array(y_start)
    y_end = np.array(y_end)
    p_start = np.array(p_start)
    p_end = np.array(p_end)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    # ASFormer-style per-prediction greedy matching
    for j in range(len(p_label)):
        if len(y_label) == 0:
            fp += 1
            continue

        # Compute IoU with all GT segments (ASFormer style)
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)

        # Mask by label match (ASFormer multiplies by label equality)
        label_match = np.array([p_label[j] == y_label[x] for x in range(len(y_label))], dtype=float)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            IoU = np.where(union > 0, (intersection / union) * label_match, 0.0)

        # Get the best scoring segment
        idx = np.argmax(IoU)

        if IoU[idx] >= iou_threshold and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(y_label) - int(sum(hits))
    return (tp, fp, fn)

def _auto_background(label_names: Optional[Sequence[str]]) -> Optional[str]:
    if not label_names: return None
    lower = [str(x).lower() for x in label_names]
    for cand in ("background", "bg", "sil", "silence"):
        if cand in lower:
            # return original-cased token at that index
            return label_names[lower.index(cand)]
    return None


def _infer_background_value(
    background: Optional[Any],
    label_names: Optional[Sequence[Any]],
    sample_values: Optional[Iterable[Any]] = None,
) -> Optional[Any]:
    """
    Resolve which label should be treated as background.

    Priority:
      1) Explicit `background` argument (if provided) - will be normalized to match
         the data type found in sample_values/label_names.
      2) Semantic match in `label_names` or `sample_values`
         for common background tokens: background/bg/sil/silence.
      3) Fallback: if labels look numeric and contain 0, treat 0 as background
         (common convention in TAS datasets, including GTEA).
    Returns the value from the supplied sequences to preserve type (str/int).
    """
    def _find(seq: Iterable[Any], target_str: Optional[str] = None) -> Optional[Any]:
        """Find background value in sequence, optionally matching a target string."""
        for v in seq:
            lv = str(v).lower()
            if target_str is not None:
                if lv == target_str.lower():
                    return v
            elif lv in ("background", "bg", "sil", "silence"):
                return v
        # Fallback to numeric 0
        if target_str is None or target_str == "0":
            for v in seq:
                if str(v) == "0":
                    return v
        return None

    # If explicit background provided, try to find matching value with same string repr
    # This handles type normalization (e.g., "0" -> 0 if data has integers)
    if background is not None:
        bg_str = str(background)
        # Try to find matching value in sample_values or label_names
        if sample_values is not None:
            found = _find(sample_values, bg_str)
            if found is not None:
                return found
        if label_names:
            found = _find(label_names, bg_str)
            if found is not None:
                return found
        # No match found, return original
        return background

    # Auto-detection
    if label_names:
        found = _find(label_names)
        if found is not None:
            return found

    if sample_values is not None:
        found = _find(sample_values)
        if found is not None:
            return found

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
    resolved_background = _infer_background_value(
        background,
        label_names,
        sample_values=list(y_true) + list(y_pred),
    )
    return {
        "acc":   frame_accuracy(y_true, y_pred),
        "edit":  edit_score(y_true, y_pred, resolved_background, label_names),
        "f1@10": segmental_f1(y_true, y_pred, 0.10, resolved_background, label_names),
        "f1@25": segmental_f1(y_true, y_pred, 0.25, resolved_background, label_names),
        "f1@50": segmental_f1(y_true, y_pred, 0.50, resolved_background, label_names),
    }

def compute_tas_metrics_macro(
    df: pd.DataFrame,
    pred_col: str,
    *,
    gt_col: str = "ground_truth",
    case_col: str = "case:concept:name",
    background: Optional[Any] = None,
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

    # Build label names if not provided (use original value types)
    if label_names is None:
        combined_vals = pd.concat([df[gt_col], df[pred_col]]).unique().tolist()
        label_names = combined_vals

    resolved_background = _infer_background_value(
        background,
        label_names,
        sample_values=pd.concat([df[gt_col], df[pred_col]]).tolist(),
    )

    # --- per-video metrics (for macro Edit/F1) ---
    rows = []
    for vid, g in df.groupby(case_col, sort=False):
        y_true = g[gt_col].tolist()
        y_pred = g[pred_col].tolist()
        if not y_true:  # skip empty sequences just in case
            continue
        m = tas_metrics(
            y_true,
            y_pred,
            background=resolved_background,
            label_names=label_names,
        )
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


def compute_tas_metrics_asformer(
    df: pd.DataFrame,
    pred_col: str,
    *,
    gt_col: str = "ground_truth",
    case_col: str = "case:concept:name",
    background: Optional[Any] = None,
    dataset_name: Optional[str] = None,
    mapping_path: Optional[Union[str, Path]] = None,
    label_names: Optional[Sequence[str]] = None,
    return_per_video: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
    """
    Compute dataset-level TAS metrics following ASFormer/MS-TCN2 protocol exactly.

    Key differences from compute_tas_metrics_macro:
      - F1 scores use MICRO-averaging: aggregate TP/FP/FN across all videos first,
        then compute F1 once (instead of averaging per-video F1 scores).
      - Segment matching uses per-prediction greedy (ASFormer style) instead of
        global best-first greedy.
      - Edit score is still macro-averaged (mean across videos) as in ASFormer.

    This function produces results identical to ASFormer/MS-TCN2's eval.py.

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
        Label(s) to treat as background. If None, defaults to the literal
        "background" (ASFormer/MS-TCN2 behavior). Pass a dataset-specific label
        (e.g., "10") when your labels are numeric IDs.
    dataset_name : str, optional
        Dataset name used to locate a mapping.txt for auto background resolution.
    mapping_path : str or Path, optional
        Explicit mapping.txt path for auto background resolution.
    label_names : Sequence[str], optional
        Optional label names used for background auto-resolution.
    return_per_video : bool, default False
        If True, also return the per-video metrics DataFrame.

    Returns
    -------
    dict
        {
          "acc": <global frame accuracy (micro)>,
          "edit": <macro mean across videos>,
          "f1@10": <micro F1 from aggregated TP/FP/FN>,
          "f1@25": <micro F1 from aggregated TP/FP/FN>,
          "f1@50": <micro F1 from aggregated TP/FP/FN>,
        }
    (dict, DataFrame) if return_per_video=True
    """
    required = {case_col, gt_col, pred_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df_local = df.copy()
    df_local[gt_col] = df_local[gt_col].astype(str)
    df_local[pred_col] = df_local[pred_col].astype(str)

    # Build label names if not provided (for background auto-resolution)
    if label_names is None:
        label_names = pd.concat([df_local[gt_col], df_local[pred_col]]).unique().tolist()

    resolved_background = _resolve_asformer_background(
        background,
        label_names=label_names,
        sample_values=pd.concat([df_local[gt_col], df_local[pred_col]]).tolist(),
        dataset_name=dataset_name,
        mapping_path=mapping_path,
    )
    bg_class = _asformer_bg_class(resolved_background)

    # Aggregated counts for micro F1
    overlap = [0.1, 0.25, 0.5]
    tp_total = np.zeros(3)
    fp_total = np.zeros(3)
    fn_total = np.zeros(3)

    # Per-video tracking
    correct = 0
    total = 0
    edit_scores = []
    per_video_rows = []

    for vid, g in df_local.groupby(case_col, sort=False):
        y_true = g[gt_col].tolist()
        y_pred = g[pred_col].tolist()
        if not y_true:
            continue

        # Frame accuracy (accumulate for micro)
        for i in range(len(y_true)):
            total += 1
            if y_true[i] == y_pred[i]:
                correct += 1

        # Edit score (per-video, will be macro-averaged)
        ed = _edit_score_asformer(y_true, y_pred, bg_class)
        edit_scores.append(ed)

        # F1 counts using ASFormer matching (aggregate for micro)
        video_f1s = []
        for s, thresh in enumerate(overlap):
            tp1, fp1, fn1 = _segmental_f1_counts_asformer(
                y_true, y_pred, thresh, bg_class
            )
            tp_total[s] += tp1
            fp_total[s] += fp1
            fn_total[s] += fn1

            # Compute per-video F1 for the per_video_df
            if tp1 + fp1 > 0:
                prec = tp1 / (tp1 + fp1)
            else:
                prec = 0.0
            if tp1 + fn1 > 0:
                rec = tp1 / (tp1 + fn1)
            else:
                rec = 0.0
            if prec + rec > 0:
                f1 = 2.0 * prec * rec / (prec + rec) * 100.0
            else:
                f1 = 0.0
            video_f1s.append(f1)

        # Per-video accuracy
        vid_correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        vid_acc = 100.0 * vid_correct / len(y_true) if y_true else 0.0

        per_video_rows.append({
            "video": vid,
            "n_frames": len(y_true),
            "acc": vid_acc,
            "edit": ed,
            "f1@10": video_f1s[0],
            "f1@25": video_f1s[1],
            "f1@50": video_f1s[2],
        })

    per_video_df = pd.DataFrame(per_video_rows)

    # Handle empty case
    if total == 0:
        summary = {"acc": 0.0, "edit": 0.0, "f1@10": 0.0, "f1@25": 0.0, "f1@50": 0.0}
        return (summary, per_video_df) if return_per_video else summary

    # Micro accuracy
    acc = 100.0 * correct / total

    # Macro edit score
    edit_macro = sum(edit_scores) / len(edit_scores) if edit_scores else 0.0

    # Micro F1 scores (from aggregated counts)
    f1_micro = {}
    for s, thresh in enumerate(overlap):
        if tp_total[s] + fp_total[s] > 0:
            precision = tp_total[s] / (tp_total[s] + fp_total[s])
        else:
            precision = 0.0
        if tp_total[s] + fn_total[s] > 0:
            recall = tp_total[s] / (tp_total[s] + fn_total[s])
        else:
            recall = 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall) * 100.0
        else:
            f1 = 0.0
        f1_micro[f"f1@{int(thresh*100)}"] = f1

    summary = {
        "acc": float(acc),
        "edit": float(edit_macro),
        **{k: float(v) for k, v in f1_micro.items()},
    }

    return (summary, per_video_df) if return_per_video else summary


def compute_tas_metrics_from_sequences(
    gt_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    background: Optional[Any] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute TAS metrics directly from ground truth and prediction sequences.

    This function takes lists of sequences (one per video/case) and computes
    aggregated metrics using ASFormer/MS-TCN2 compatible methodology.

    Parameters
    ----------
    gt_sequences : List[List[str]]
        List of ground truth sequences, one per video/case.
    pred_sequences : List[List[str]]
        List of prediction sequences, one per video/case.
        Must have same length as gt_sequences.
    background : Any, optional
        Label to treat as background (ignored in Edit/F1 segmentization).
        If None, defaults to "background" (ASFormer behavior).
    label_names : Sequence[str], optional
        Optional label names (unused, present for API compatibility).

    Returns
    -------
    Dict[str, float]
        {
          "acc_micro": <global frame accuracy (micro)>,
          "acc": <same as acc_micro>,
          "edit":  <macro mean edit score across sequences>,
          "f1@10": <micro F1@10 from aggregated TP/FP/FN>,
          "f1@25": <micro F1@25 from aggregated TP/FP/FN>,
          "f1@50": <micro F1@50 from aggregated TP/FP/FN>,
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

    # Validate sequence lengths
    for i, (gt_seq, pred_seq) in enumerate(zip(gt_sequences, pred_sequences)):
        if len(gt_seq) != len(pred_seq):
            raise ValueError(
                f"Sequence {i} length mismatch: GT has {len(gt_seq)}, "
                f"Pred has {len(pred_seq)}"
            )

    # Build DataFrame from sequences for ASFormer-compatible metrics
    rows = []
    for case_idx, (gt_seq, pred_seq) in enumerate(zip(gt_sequences, pred_sequences)):
        case_id = f"case_{case_idx}"
        for gt_label, pred_label in zip(gt_seq, pred_seq):
            rows.append({
                'case:concept:name': case_id,
                'ground_truth': str(gt_label),
                'prediction': str(pred_label),
            })
    metrics_df = pd.DataFrame(rows)

    # Compute metrics using ASFormer-compatible method
    metrics = compute_tas_metrics_asformer(
        df=metrics_df,
        pred_col='prediction',
        gt_col='ground_truth',
        case_col='case:concept:name',
        background=background,
    )

    # Add acc_micro alias for backwards compatibility
    if 'acc' in metrics:
        metrics['acc_micro'] = metrics['acc']

    return metrics


def compute_sktr_vs_argmax_metrics(
    csv_path: str,
    *,
    case_col: str = "case:concept:name",
    sktr_pred_col: str = "sktr_activity",
    argmax_pred_col: str = "argmax_activity",
    kari_pred_col: Optional[str] = "kari_activity",
    gt_col: str = "ground_truth",
    background: Optional[Any] = None,
    dataset_name: Optional[str] = None,
    mapping_path: Optional[Union[str, Path]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute TAS metrics for both SKTR and argmax approaches from a CSV file.
    Optionally computes KARI metrics if the specified KARI prediction column exists.

    Uses ASFormer/MS-TCN2 compatible metrics computation (micro-averaged F1 scores)
    to ensure fair comparison with published results.

    This function loads prediction results from a CSV file containing SKTR and argmax
    predictions alongside ground truth labels, and computes comprehensive TAS metrics
    for both approaches. If a KARI prediction column is specified and exists, KARI
    metrics are also computed.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing prediction results.
    case_col : str, default "case:concept:name"
        Column name identifying each case/video.
    sktr_pred_col : str, default "sktr_activity"
        Column name containing SKTR predictions.
    argmax_pred_col : str, default "argmax_activity"
        Column name containing argmax predictions.
    kari_pred_col : str, default "kari_activity"
        Optional column name containing KARI predictions. If None or the column
        doesn't exist, KARI metrics are not computed.
    gt_col : str, default "ground_truth"
        Column name containing ground truth labels.
    background : Any, optional
        Label to treat as background (ignored in Edit/F1 segmentation).
        If None, auto-detection is used.
    dataset_name : str, optional
        Dataset name used for background auto-resolution. If None, attempts to infer
        from the CSV path (50salads/gtea/breakfast).
    mapping_path : str or Path, optional
        Explicit mapping.txt path for background auto-resolution.
    label_names : Sequence[str], optional
        Optional label names for background auto-detection.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with keys 'sktr', 'argmax', and optionally 'kari', each containing
        a dictionary of TAS metrics:
        {
            'sktr': {
                'acc': float,
                'edit': float,
                'f1@10': float,
                'f1@25': float,
                'f1@50': float,
            },
            'argmax': {
                'acc': float,
                'edit': float,
                'f1@10': float,
                'f1@25': float,
                'f1@50': float,
            },
            'kari': { ... }  # if kari_pred_col exists
        }

    Raises
    ------
    ValueError
        If required columns are missing from the CSV file.
    FileNotFoundError
        If the CSV file doesn't exist.

    Examples
    --------
    >>> metrics = compute_sktr_vs_argmax_metrics('sktr_results_50salads_condprob.csv')
    >>> print(f"SKTR Edit: {metrics['sktr']['edit']:.2f}%")
    >>> print(f"Argmax Edit: {metrics['argmax']['edit']:.2f}%")

    For files with KARI results:
    >>> metrics = compute_sktr_vs_argmax_metrics('sktr_kari_argmax_50salads_results.csv')
    >>> if 'kari' in metrics:
    ...     print(f"KARI Edit: {metrics['kari']['edit']:.2f}%")
    """
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Check required columns (SKTR, argmax, GT, case always required)
    required_cols = {case_col, sktr_pred_col, argmax_pred_col, gt_col}
    # Add KARI if specified and not None
    if kari_pred_col is not None:
        if kari_pred_col not in df.columns:
            print(f"Warning: KARI column '{kari_pred_col}' not found. Skipping KARI metrics.")
            kari_pred_col = None
        else:
            required_cols.add(kari_pred_col)

    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if dataset_name is None:
        dataset_name = _infer_dataset_name_from_path(csv_path)

    # Normalize label columns to string for consistent matching
    for col in [sktr_pred_col, argmax_pred_col, gt_col]:
        df[col] = df[col].astype(str)
    if kari_pred_col is not None and kari_pred_col in df.columns:
        df[kari_pred_col] = df[kari_pred_col].astype(str)

    if background is not None:
        if isinstance(background, (list, tuple, set, np.ndarray)):
            background = [str(x) for x in background]
        else:
            background = str(background)

    # Compute metrics for SKTR (using ASFormer-compatible method)
    sktr_metrics = compute_tas_metrics_asformer(
        df,
        pred_col=sktr_pred_col,
        gt_col=gt_col,
        case_col=case_col,
        background=background,
        dataset_name=dataset_name,
        mapping_path=mapping_path,
        label_names=label_names,
    )

    # Compute metrics for argmax (using ASFormer-compatible method)
    argmax_metrics = compute_tas_metrics_asformer(
        df,
        pred_col=argmax_pred_col,
        gt_col=gt_col,
        case_col=case_col,
        background=background,
        dataset_name=dataset_name,
        mapping_path=mapping_path,
        label_names=label_names,
    )

    # Compute metrics for KARI if available (using ASFormer-compatible method)
    kari_metrics = None
    if kari_pred_col is not None:
        kari_metrics = compute_tas_metrics_asformer(
            df,
            pred_col=kari_pred_col,
            gt_col=gt_col,
            case_col=case_col,
            background=background,
            dataset_name=dataset_name,
            mapping_path=mapping_path,
            label_names=label_names,
        )

    return {
        'sktr': sktr_metrics,
        'argmax': argmax_metrics,
        **({'kari': kari_metrics} if kari_metrics is not None else {})
    }


def print_tas_metrics_from_csv(
    csv_path: str,
    *,
    case_col: str = "case:concept:name",
    sktr_pred_col: str = "sktr_activity",
    argmax_pred_col: str = "argmax_activity",
    kari_pred_col: Optional[str] = "kari_activity",
    gt_col: str = "ground_truth",
    background: Optional[Any] = None,
    dataset_name: Optional[str] = None,
    mapping_path: Optional[Union[str, Path]] = None,
    label_names: Optional[Sequence[str]] = None,
    precision: int = 2,
    return_tables: bool = False,
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load a CSV, compute TAS metrics (SKTR, Argmax, optionally KARI), and print a
    compact comparison table plus SKTR-ARGMAX (and SKTR-KARI if available) differences.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns `[case_col, gt_col, sktr_pred_col, argmax_pred_col]`.
        If ``kari_pred_col`` exists in the CSV, KARI metrics are included automatically.
    case_col, sktr_pred_col, argmax_pred_col, kari_pred_col, gt_col : str
        Column names; defaults match the repository's CSVs.
    background : Any, optional
        Background label for Edit/F1. Use None or "auto" for auto-detection.
    dataset_name : str, optional
        Dataset name used for background auto-resolution.
    mapping_path : str or Path, optional
        Explicit mapping.txt path for background auto-resolution.
    label_names : Sequence[str], optional
        Label names for background auto-detection.
    precision : int, default 2
        Rounding precision for printed tables.
    return_tables : bool, default False
        If True, return a dict with the pivot and difference tables instead of None.

    Returns
    -------
    Optional[Dict[str, pd.DataFrame]]
        When ``return_tables`` is True, returns {
          'pivot': DataFrame,
          'diff_sktr_argmax': DataFrame,
          'diff_sktr_kari': DataFrame (only if KARI present)
        }.
    """
    metrics = compute_sktr_vs_argmax_metrics(
        csv_path,
        case_col=case_col,
        sktr_pred_col=sktr_pred_col,
        argmax_pred_col=argmax_pred_col,
        kari_pred_col=kari_pred_col,
        gt_col=gt_col,
        background=background,
        dataset_name=dataset_name,
        mapping_path=mapping_path,
        label_names=label_names,
    )

    # Build tidy table: rows: approaches, columns: metrics
    approaches = ['sktr', 'argmax']
    if 'kari' in metrics:
        approaches.append('kari')
    approaches = sorted(approaches, key=lambda x: x.upper())  # ARGMAX, KARI, SKTR

    comparison_data = []
    for approach in approaches:
        for metric_name, value in metrics[approach].items():
            comparison_data.append({
                'Approach': approach.upper(),
                'Metric': metric_name,
                'Value': value,
            })

    comparison_df = pd.DataFrame(comparison_data)
    pivot_df = comparison_df.pivot(index='Approach', columns='Metric', values='Value').round(precision)

    print("\nTAS Metrics Comparison")
    print("=" * 50)
    print(pivot_df)

    # Differences: SKTR - ARGMAX
    print("\n" + "=" * 50)
    print("SKTR - Argmax Differences")
    argmax_row = pivot_df.loc['ARGMAX']
    sktr_row = pivot_df.loc['SKTR']
    diff_row = sktr_row - argmax_row
    diff_df = pd.DataFrame([diff_row], index=['SKTR-ARGMAX']).round(precision)
    print(diff_df)

    # Optional: SKTR - KARI
    diff_kari_df: Optional[pd.DataFrame] = None
    if 'KARI' in pivot_df.index:
        print("\n" + "=" * 50)
        print("SKTR - KARI Differences")
        kari_row = pivot_df.loc['KARI']
        sktr_kari_diff = sktr_row - kari_row
        diff_kari_df = pd.DataFrame([sktr_kari_diff], index=['SKTR-KARI']).round(precision)
        print(diff_kari_df)
    else:
        print("\nNo KARI data available for comparison.")

    if return_tables:
        out: Dict[str, pd.DataFrame] = {
            'pivot': pivot_df,
            'diff_sktr_argmax': diff_df,
        }
        if diff_kari_df is not None:
            out['diff_sktr_kari'] = diff_kari_df
        return out
    return None
