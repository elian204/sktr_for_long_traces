"""
Trace export utilities for event logs.

Functions to collapse consecutive repeated activities in traces and
write them to a line-based text format.
"""

from __future__ import annotations

from typing import Iterable, List, Any

import pandas as pd


def collapse_consecutive_runs(activities: Iterable[Any]) -> List[Any]:
    """
    Collapse consecutive repeated items in an iterable, preserving order.

    NaN values are considered equal to NaN for the purpose of runs.
    Returns a list.
    """
    activities_list = list(activities)
    if not activities_list:
        return []

    def _equal(a, b) -> bool:
        if pd.isna(a) and pd.isna(b):
            return True
        return a == b

    collapsed: List[Any] = []
    current = activities_list[0]
    for item in activities_list[1:]:
        if not _equal(item, current):
            collapsed.append(current)
            current = item
    collapsed.append(current)
    return collapsed


def write_collapsed_traces(
    df: pd.DataFrame,
    output_file_path: str,
    case_column: str = 'case:concept:name',
    activity_column: str = 'concept:name',
    separator: str = ' ',
    line_prefix: str = '* ',
    line_suffix: str = ' #',
) -> None:
    """
    Write one collapsed trace per line to a text file.

    - Consecutive repeated activities are collapsed into one
    - Line format: "{line_prefix}{activity1}{sep}{activity2}...{line_suffix}\n"
    - Defaults match "* ... #" format
    """
    if case_column not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{case_column}'")
    if activity_column not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{activity_column}'")

    unique_case_ids = df[case_column].drop_duplicates().tolist()

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for case_id in unique_case_ids:
            case_activities = df[df[case_column] == case_id][activity_column].tolist()
            if not case_activities:
                f.write(f"{line_prefix}{line_suffix}\n")
                continue

            collapsed = collapse_consecutive_runs(case_activities)
            line = separator.join(str(x) for x in collapsed)
            f.write(f"{line_prefix}{line}{line_suffix}\n")


