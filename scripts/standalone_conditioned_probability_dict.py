"""
Build conditional next-activity probability dictionaries from traces.

This file is a standalone extraction of the probability-dictionary builder used
in ``sktr_for_long_traces/src/petri_model.py``.  It intentionally has no SKTR,
Petri-net, or pm4py dependency.

The main output is a dictionary of the form::

    {
        history_tuple: {
            next_activity: probability,
            ...
        },
        ...
    }

For example, if ``('A', 'B')`` maps to ``{'C': 0.75, 'D': 0.25}``, then the
estimated probabilities are P(C | A, B) = 0.75 and P(D | A, B) = 0.25.

The empty history ``()`` is reserved for the first activity of each trace, so
``prob_dict[()]`` is the distribution of trace-start activities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, DefaultDict, Dict, Hashable, List, Tuple


Activity = Hashable
History = Tuple[Activity, ...]
ProbabilityDict = Dict[History, Dict[Activity, float]]


def collapse_runs(sequence: Sequence[Activity]) -> List[Activity]:
    """
    Collapse consecutive repeated activities into a single activity.

    Parameters
    ----------
    sequence:
        Activity sequence, possibly containing consecutive repeats.

    Returns
    -------
    list
        A new list with consecutive repeated labels removed.

    Examples
    --------
    >>> collapse_runs(["A", "A", "B", "B", "A"])
    ['A', 'B', 'A']
    """
    if not sequence:
        return []

    collapsed = [sequence[0]]
    for activity in sequence[1:]:
        if activity != collapsed[-1]:
            collapsed.append(activity)
    return collapsed


def history_activity_pairs(
    sequence: Sequence[Activity],
    max_history_length: int,
    *,
    collapse_consecutive_runs: bool = True,
) -> List[Tuple[History, Activity]]:
    """
    Generate all ``(history, next_activity)`` pairs up to a maximum history size.

    Histories are ordered chronologically, from oldest to newest.  For the
    sequence ``["A", "B", "C"]`` with ``max_history_length=2``, the generated
    pairs are::

        ((), "A")
        (("A",), "B")
        (("B",), "C")
        (("A", "B"), "C")

    Parameters
    ----------
    sequence:
        One activity trace.
    max_history_length:
        Longest history tuple to generate.  Use 1 for one previous activity,
        2 for two previous activities, and so on.  A value of 0 keeps only the
        empty-history trace-start pair.
    collapse_consecutive_runs:
        If True, consecutive repeats are collapsed before extracting histories.
        This matches the run-to-run transition dictionary used by
        ``sktr_for_long_traces``.  If False, repeated labels are kept and the
        dictionary captures within-run continuation probabilities.

    Returns
    -------
    list[tuple[tuple, activity]]
        All history/next-activity observations from the trace.
    """
    if max_history_length < 0:
        raise ValueError("max_history_length must be non-negative")

    processed_sequence = (
        collapse_runs(sequence) if collapse_consecutive_runs else list(sequence)
    )
    if not processed_sequence:
        return []

    pairs: List[Tuple[History, Activity]] = [((), processed_sequence[0])]

    for index in range(1, len(processed_sequence)):
        next_activity = processed_sequence[index]
        longest_available_history = min(index, max_history_length)

        for history_length in range(1, longest_available_history + 1):
            history = tuple(
                processed_sequence[index - history_length:index]
            )
            pairs.append((history, next_activity))

    return pairs


def build_conditioned_probability_dict(
    data: Any,
    *,
    max_history_length: int = 2,
    precision: int = 2,
    case_column: str = "case:concept:name",
    activity_column: str = "concept:name",
    collapse_consecutive_runs: bool = True,
) -> ProbabilityDict:
    """
    Build empirical conditional probabilities P(next_activity | history).

    Parameters
    ----------
    data:
        Either:

        - a pandas DataFrame-like event log with case and activity columns, or
        - an iterable of activity sequences, such as
          ``[["A", "B", "C"], ["A", "C"]]``.

        DataFrame rows are used in their existing order inside each case.  This
        matches the original SKTR implementation.
    max_history_length:
        Maximum history length to include in the dictionary.
    precision:
        Number of decimal places used when rounding probabilities.
    case_column:
        Case-id column name when ``data`` is a DataFrame.
    activity_column:
        Activity-label column name when ``data`` is a DataFrame.
    collapse_consecutive_runs:
        If True, collapse consecutive repeated activities before counting.  Use
        True for run-to-run transition probabilities and False for raw
        event-to-event continuation probabilities.

    Returns
    -------
    dict[tuple, dict[activity, float]]
        Mapping from history tuples to next-activity probability distributions.

    Notes
    -----
    This is a relative-frequency estimator with no smoothing.  Unseen histories
    and unseen next activities are absent from the dictionary and should be
    treated as probability 0 by the caller.

    Examples
    --------
    >>> traces = [["A", "A", "B"], ["A", "C"]]
    >>> build_conditioned_probability_dict(traces, max_history_length=1)
    {(): {'A': 1.0}, ('A',): {'B': 0.5, 'C': 0.5}}
    """
    if max_history_length < 0:
        raise ValueError("max_history_length must be non-negative")
    if precision < 0:
        raise ValueError("precision must be non-negative")

    sequences = _extract_activity_sequences(
        data,
        case_column=case_column,
        activity_column=activity_column,
    )

    history_counts: DefaultDict[History, DefaultDict[Activity, int]]
    history_counts = defaultdict(lambda: defaultdict(int))

    for sequence in sequences:
        for history, next_activity in history_activity_pairs(
            sequence,
            max_history_length,
            collapse_consecutive_runs=collapse_consecutive_runs,
        ):
            history_counts[history][next_activity] += 1

    probability_dict: ProbabilityDict = {}
    for history, next_activity_counts in history_counts.items():
        total = sum(next_activity_counts.values())
        if total <= 0:
            continue

        probability_dict[history] = {
            activity: round(count / total, precision)
            for activity, count in next_activity_counts.items()
        }

    return probability_dict


def build_probability_dict(
    train_df: Any,
    max_hist_len: int = 3,
    precision: int = 2,
    use_collapsed: bool = True,
) -> ProbabilityDict:
    """
    Compatibility wrapper matching ``sktr_for_long_traces`` naming.

    Parameters match ``sktr_for_long_traces/src/petri_model.py``:
    ``max_hist_len`` is the maximum history length and ``use_collapsed`` controls
    whether consecutive repeated labels are collapsed before counting.
    """
    return build_conditioned_probability_dict(
        train_df,
        max_history_length=max_hist_len,
        precision=precision,
        collapse_consecutive_runs=use_collapsed,
    )


def conditioned_probability(
    probability_dict: ProbabilityDict,
    history: Sequence[Activity],
    next_activity: Activity,
    *,
    default: float = 0.0,
) -> float:
    """
    Look up P(next_activity | history) in a probability dictionary.

    Parameters
    ----------
    probability_dict:
        Dictionary returned by ``build_conditioned_probability_dict``.
    history:
        History sequence in chronological order.
    next_activity:
        Activity whose conditional probability should be returned.
    default:
        Value returned when the history or next activity was not observed.
    """
    return probability_dict.get(tuple(history), {}).get(next_activity, default)


def _extract_activity_sequences(
    data: Any,
    *,
    case_column: str,
    activity_column: str,
) -> List[List[Activity]]:
    """
    Extract activity sequences from a DataFrame-like object or list of traces.
    """
    if _looks_like_dataframe(data):
        missing_columns = [
            column
            for column in (case_column, activity_column)
            if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required column(s): {missing_columns}")

        return [
            group[activity_column].tolist()
            for _, group in data.groupby(case_column, sort=False)
        ]

    if isinstance(data, (str, bytes)):
        raise TypeError(
            "data must be a DataFrame-like object or an iterable of traces, "
            "not a single string"
        )

    if not isinstance(data, Iterable):
        raise TypeError(
            "data must be a DataFrame-like object or an iterable of traces"
        )

    return [list(sequence) for sequence in data]


def _looks_like_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "groupby")


if __name__ == "__main__":
    example_traces = [
        ["A", "A", "B", "B", "C"],
        ["A", "C"],
    ]

    collapsed = build_conditioned_probability_dict(
        example_traces,
        max_history_length=2,
        collapse_consecutive_runs=True,
    )
    uncollapsed = build_conditioned_probability_dict(
        example_traces,
        max_history_length=2,
        collapse_consecutive_runs=False,
    )

    print("Collapsed run-to-run probabilities:")
    print(collapsed)
    print("\nUncollapsed event-to-event probabilities:")
    print(uncollapsed)
