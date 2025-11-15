from __future__ import annotations

import ast
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

from src.data_processing import filter_indices, split_train_test
from src.petri_model import build_probability_dict


HistoryKey = Tuple[str, ...]
ProbabilityDict = Mapping[HistoryKey, Mapping[str, float]]


_HISTORY_NAME_MAP = {
    1: "unigram",
    2: "bigram",
    3: "trigram",
}


def fix_windowed_csv_steps(
    csv_path: str,
    output_path: str | None = None,
    case_col: str = "case:concept:name",
    step_col: str = "step",
) -> pd.DataFrame:
    """
    Fix step values in a CSV file that contains windowed data with step resets.

    Creates a cumulative step counter for each case, replacing the original
    cyclic step values with continuous 0-based indices.

    Parameters
    ----------
    csv_path
        Path to the input CSV file with windowed data.
    output_path
        Path to save the corrected CSV. If None, overwrites the input file.
    case_col
        Column name for case identifier. Default is "case:concept:name".
    step_col
        Column name for step/index. Default is "step".

    Returns
    -------
    pd.DataFrame
        The corrected DataFrame with cumulative step indices.

    Notes
    -----
    This function detects window boundaries (where step values reset) and
    replaces them with continuous 0-based counters for each case.
    """
    df = pd.read_csv(csv_path)

    if case_col not in df.columns:
        raise ValueError(f"Column '{case_col}' not found in CSV")
    if step_col not in df.columns:
        raise ValueError(f"Column '{step_col}' not found in CSV")

    corrected_dfs = []
    total_resets = 0
    cases_with_resets = 0

    for case_id, group in df.groupby(case_col, sort=False):
        # Preserve original row order
        group_sorted = group.copy()

        # Detect resets
        step_values = group_sorted[step_col].values
        has_resets = False
        for i in range(1, len(step_values)):
            if step_values[i] <= step_values[i - 1]:
                has_resets = True
                total_resets += 1

        if has_resets:
            cases_with_resets += 1

        # Replace step column with cumulative counter
        group_sorted[step_col] = range(len(group_sorted))
        corrected_dfs.append(group_sorted)

    # Concatenate all corrected groups
    df_corrected = pd.concat(corrected_dfs, ignore_index=True)

    # Report statistics
    if cases_with_resets > 0:
        print(f"✓ Fixed windowed data:")
        print(f"  - {cases_with_resets} case(s) had step resets")
        print(f"  - {total_resets} window boundaries corrected")
        print(f"  - Step values now continuous (0-based) for each case")
    else:
        print("No step resets detected. CSV already has continuous step values.")

    # Save to file
    if output_path is None:
        output_path = csv_path

    df_corrected.to_csv(output_path, index=False)
    print(f"✓ Saved corrected CSV to: {output_path}")

    return df_corrected


def _resolve_history_label(history_len: int) -> str:
    """
    Return a readable label for a given history length.
    """
    return _HISTORY_NAME_MAP.get(history_len, f"{history_len}-gram")


def build_probability_dicts_from_config(
    df: pd.DataFrame,
    *,
    n_train_traces: int,
    train_cases: Optional[List[Any]] = None,
    ensure_train_variant_diversity: bool = False,
    sequential_sampling: bool = True,
    n_indices: Optional[int] = None,
    n_per_run: Optional[int] = None,
    independent_sampling: bool = True,
    max_hist_len: int = 3,
    random_seed: int = 42,
) -> Tuple[ProbabilityDict, ProbabilityDict]:
    """
    Build the uncollapsed and collapsed probability dictionaries using the exact
    filtering and splitting logic from ``incremental_softmax_recovery``.

    This helper makes it easy to recreate ``prob_dict_uncollapsed`` and
    ``prob_dict_collapsed`` for post-hoc analysis without rerunning recovery.
    """
    required_cols = {"case:concept:name", "concept:name"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "DataFrame is missing required columns: "
            f"{sorted(missing_cols)}. Expected {sorted(required_cols)}"
        )

    unique_case_ids = df["case:concept:name"].drop_duplicates().tolist()
    try:
        numeric_case_ids = [int(cid) for cid in unique_case_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Case IDs must be numeric strings (e.g., '0', '1', ...)"
        ) from exc

    expected_ids = list(range(len(unique_case_ids)))
    if sorted(numeric_case_ids) != expected_ids:
        raise ValueError(
            "Case IDs must be sequential strings starting at '0'. "
            f"Expected {[str(i) for i in expected_ids]}, "
            f"found {unique_case_ids}"
        )

    df_sorted = df.copy()
    df_sorted["_case_id_numeric"] = pd.to_numeric(df_sorted["case:concept:name"])
    df_sorted["_event_order"] = df_sorted.groupby("case:concept:name").cumcount()
    df_sorted = df_sorted.sort_values(
        ["_case_id_numeric", "_event_order"], kind="stable"
    ).drop(columns=["_case_id_numeric", "_event_order"])
    df_sorted = df_sorted.reset_index(drop=True)

    filtered_log, _ = filter_indices(
        df_sorted,
        softmax_list=None,
        n_indices=n_indices,
        n_per_run=n_per_run,
        sequential_sampling=sequential_sampling,
        independent_sampling=independent_sampling,
        random_seed=random_seed,
    )

    train_df, _ = split_train_test(
        filtered_log,
        n_train_traces=n_train_traces,
        n_test_traces=None,
        train_cases=train_cases,
        test_cases=None,
        ensure_train_variant_diversity=ensure_train_variant_diversity,
        ensure_test_variant_diversity=False,
        allow_train_cases_in_test=False,
        random_seed=random_seed,
    )

    prob_dict_uncollapsed = build_probability_dict(
        train_df,
        max_hist_len=max_hist_len,
        use_collapsed=False,
    )
    prob_dict_collapsed = build_probability_dict(
        train_df,
        max_hist_len=max_hist_len,
        use_collapsed=True,
    )
    return prob_dict_uncollapsed, prob_dict_collapsed


def build_probability_dicts_from_config_dict(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[ProbabilityDict, ProbabilityDict]:
    """
    Convenience wrapper that extracts the relevant pieces from a config dict.
    """
    return build_probability_dicts_from_config(
        df=df,
        n_train_traces=config["n_train_traces"],
        train_cases=config.get("train_cases"),
        ensure_train_variant_diversity=config.get("ensure_train_variant_diversity", False),
        sequential_sampling=config.get("sequential_sampling", True),
        n_indices=config.get("n_indices"),
        n_per_run=config.get("n_per_run"),
        independent_sampling=config.get("independent_sampling", True),
        max_hist_len=config.get("max_hist_len", 3),
        random_seed=config.get("random_seed", 42),
    )


def _safe_literal_eval(value: Any) -> Optional[Any]:
    """
    Safely parse string representations of Python literals.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return value


def _extract_softmax_probs(
    row: Mapping[str, Any],
    *,
    pred_label: str,
    truth_label: str,
    all_probs_col: str,
    all_activities_col: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the softmax probabilities for the predicted and ground-truth labels.
    """
    all_probs = _safe_literal_eval(row.get(all_probs_col))
    all_activities = _safe_literal_eval(row.get(all_activities_col))
    if not all_probs or not all_activities:
        return None, None
    if len(all_probs) != len(all_activities):
        return None, None

    activity_to_prob = {
        str(activity): prob for activity, prob in zip(all_activities, all_probs)
    }
    return activity_to_prob.get(pred_label), activity_to_prob.get(truth_label)


def summarize_wrong_transitions(
    results_df: pd.DataFrame,
    prob_dict_collapsed: ProbabilityDict,
    *,
    case_col: str = "case:concept:name",
    step_col: str = "step",
    pred_col: str = "sktr_activity",
    truth_col: str = "ground_truth",
    all_probs_col: str = "all_probs",
    all_activities_col: str = "all_activities",
    history_lengths: Sequence[int] = (1,),
    min_correct_streak: int = 1,
    warn_on_resets: bool = True,
) -> pd.DataFrame:
    """
    Collect every incorrect prediction that occurred immediately after a streak
    of correct predictions and annotate it with transition and softmax
    probabilities.

    Parameters
    ----------
    results_df
        DataFrame produced by ``incremental_softmax_recovery``.
    prob_dict_collapsed
        Collapsed probability dictionary (same structure returned by SKTR).
    history_lengths
        Distinct history lengths to report (e.g., (1,) for unigram).
    min_correct_streak
        Minimum number of consecutive correct predictions required before a
        violation is recorded. Default is 1 (any single correct prediction).
    warn_on_resets
        If True, print a warning when step resets are detected (indicating
        windowed data). Default is True.

    Notes
    -----
    All rows for each case are treated as one continuous sequence, with violations
    analyzed across the entire trace. If the data contains windowed chunks (with
    step numbers resetting), windows are automatically stitched together for
    trace-level analysis by creating a cumulative step index.
    """
    if not history_lengths:
        raise ValueError("history_lengths must contain at least one value")
    normalized_lengths = tuple(sorted({int(h) for h in history_lengths if h >= 1}))
    if not normalized_lengths:
        raise ValueError("history_lengths must contain positive integers")
    max_history = max(normalized_lengths)

    required_cols = {
        case_col,
        step_col,
        pred_col,
        truth_col,
        all_probs_col,
        all_activities_col,
    }
    missing_cols = required_cols - set(results_df.columns)
    if missing_cols:
        raise ValueError(
            f"results_df is missing required columns: {sorted(missing_cols)}"
        )

    records: List[Dict[str, Any]] = []
    cases_with_resets = []

    for case_id, group in results_df.groupby(case_col, sort=False):
        # Preserve original row order from CSV
        sorted_group = group.reset_index(drop=True)

        # Detect if step values reset (indicating windowed/cyclic data)
        step_values = sorted_group[step_col].values
        has_resets = False
        num_resets = 0
        for i in range(1, len(step_values)):
            if step_values[i] <= step_values[i - 1]:
                has_resets = True
                num_resets += 1

        if has_resets:
            cases_with_resets.append((case_id, num_resets))

        # Create cumulative step index
        # If data has window resets, we create a continuous counter
        # If data is already continuous, we still create the index for consistency
        sorted_group = sorted_group.copy()
        sorted_group['_cumulative_step'] = range(len(sorted_group))

        # Treat entire case as one continuous sequence (stitching windows together)
        _process_single_sequence(
            sequence_group=sorted_group,
            case_id=case_id,
            case_col=case_col,
            step_col='_cumulative_step',
            pred_col=pred_col,
            truth_col=truth_col,
            all_probs_col=all_probs_col,
            all_activities_col=all_activities_col,
            prob_dict_collapsed=prob_dict_collapsed,
            normalized_lengths=normalized_lengths,
            max_history=max_history,
            min_correct_streak=min_correct_streak,
            records=records,
        )

    # Warn if windowed/cyclic data was detected
    if warn_on_resets and cases_with_resets:
        total_cases = len(cases_with_resets)
        total_resets = sum(num for _, num in cases_with_resets)
        print(f"⚠️  Detected windowed data: {total_cases} case(s) with step resets")
        print(f"   Total window boundaries detected: {total_resets}")
        print(f"   Automatically stitching windows together for trace-level analysis.")
        print(f"   Using cumulative step indexing (0-based continuous counter).")

    if not records:
        cols = [
            case_col,
            "prev_index",
            "violation_index",
            "streak_length",
            "prev_activity",
            "predicted_activity",
            "ground_truth",
            "predicted_prob_softmax",
            "unigram_argmax_prediction",
        ]
        for hist_len in normalized_lengths:
            hist_label = _resolve_history_label(hist_len)
            cols.append(f"{hist_label}_transition_prob")
        return pd.DataFrame(columns=cols)

    summary_df = pd.DataFrame.from_records(records)
    return summary_df


def _process_single_sequence(
    sequence_group: pd.DataFrame,
    case_id: Any,
    case_col: str,
    step_col: str,
    pred_col: str,
    truth_col: str,
    all_probs_col: str,
    all_activities_col: str,
    prob_dict_collapsed: ProbabilityDict,
    normalized_lengths: Tuple[int, ...],
    max_history: int,
    min_correct_streak: int,
    records: List[Dict[str, Any]],
) -> None:
    """
    Process a single sequence (either an entire case or one window).

    This processes predictions in order, tracking correct streaks and recording
    violations when an incorrect prediction follows a correct streak.
    """
    history = deque(maxlen=max_history)
    correct_streak = 0
    last_correct_step: Optional[int] = None

    for _, row in sequence_group.iterrows():
        predicted_label = str(row[pred_col])
        truth_label = str(row[truth_col])
        step_value = int(row[step_col])

        if predicted_label == truth_label:
            correct_streak += 1
            last_correct_step = step_value
            if not history or history[-1] != truth_label:
                history.append(truth_label)
            continue

        if correct_streak >= min_correct_streak and history:
            record: Dict[str, Any] = {
                case_col: case_id,
                "prev_index": last_correct_step,
                "violation_index": step_value,
                "streak_length": correct_streak,
                "prev_activity": history[-1],
                "predicted_activity": predicted_label,
                "ground_truth": truth_label,
            }

            pred_prob_softmax, _ = _extract_softmax_probs(
                row,
                pred_label=predicted_label,
                truth_label=truth_label,
                all_probs_col=all_probs_col,
                all_activities_col=all_activities_col,
            )
            record["predicted_prob_softmax"] = pred_prob_softmax

            history_list = list(history)

            # Compute unigram argmax prediction (what would unigram-only predict?)
            if len(history_list) >= 1:
                unigram_key = tuple(history_list[-1:])
                unigram_probs = prob_dict_collapsed.get(unigram_key, {})
                if unigram_probs:
                    unigram_argmax = max(unigram_probs.items(), key=lambda x: x[1])[0]
                    record["unigram_argmax_prediction"] = unigram_argmax
                else:
                    record["unigram_argmax_prediction"] = None
            else:
                record["unigram_argmax_prediction"] = None

            for hist_len in normalized_lengths:
                hist_label = _resolve_history_label(hist_len)
                pred_prob_col = f"{hist_label}_transition_prob"

                if len(history_list) >= hist_len:
                    history_key = tuple(history_list[-hist_len:])
                    history_probs = prob_dict_collapsed.get(history_key, {})
                    record[pred_prob_col] = history_probs.get(predicted_label)
                else:
                    record[pred_prob_col] = None

            records.append(record)

        correct_streak = 0
        last_correct_step = None
        history.clear()
