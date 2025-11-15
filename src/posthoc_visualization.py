"""
Visualization and display utilities for post-hoc analysis of SKTR results.

This module provides functions for displaying violation analysis results in
human-readable formats.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def display_violations_summary(
    violations_df: pd.DataFrame,
    top_n_cases: Optional[int] = None
) -> None:
    """
    Display a summary of violations with n-gram probabilities in a readable format.

    This function prints a formatted summary of violations, showing:
    - Overall statistics (total violations, cases affected, average per case)
    - Per-case breakdown with detailed violation information
    - N-gram transition probabilities from training data
    - Softmax probabilities at violation timestamps

    Parameters
    ----------
    violations_df : pd.DataFrame
        The violations dataframe returned by summarize_wrong_transitions from
        the posthoc_analysis module. Expected to contain columns:
        - case:concept:name
        - violation_index, prev_index, streak_length
        - prev_activity, predicted_activity, ground_truth
        - *_history_key, *_transition_prob (for unigram, bigram, trigram)
        - predicted_prob_softmax
    top_n_cases : Optional[int], default=None
        If specified, only show the first N cases. If None, shows all cases.

    Examples
    --------
    >>> from src.posthoc_analysis import summarize_wrong_transitions
    >>> from src.posthoc_visualization import display_violations_summary
    >>> violations_df = summarize_wrong_transitions(results_df, prob_dict_collapsed)
    >>> display_violations_summary(violations_df, top_n_cases=5)
    """
    if violations_df.empty:
        print("No violations found!")
        return

    # Summary statistics
    print("=" * 80)
    print("VIOLATION ANALYSIS SUMMARY")
    print("=" * 80)
    total_violations = len(violations_df)
    unique_cases = violations_df['case:concept:name'].nunique()
    avg_per_case = total_violations / unique_cases if unique_cases > 0 else 0

    print(f"Total violations: {total_violations}")
    print(f"Cases with violations: {unique_cases}")
    print(f"Average violations per case: {avg_per_case:.2f}")
    print()

    # Display violations by case
    cases = violations_df['case:concept:name'].unique()
    if top_n_cases is not None:
        cases = cases[:top_n_cases]

    for case_id in cases:
        case_violations = violations_df[violations_df['case:concept:name'] == case_id]

        print(f"\n{'=' * 80}")
        print(f"CASE: {case_id} - {len(case_violations)} violation(s)")
        print('=' * 80)

        for _, row in case_violations.iterrows():
            print(
                f"\nViolation at index {row['violation_index']} "
                f"(after {row['streak_length']} correct predictions):"
            )
            print(
                f"  Transition: {row['prev_activity']} → {row['predicted_activity']} "
                f"(should be {row['ground_truth']})"
            )

            # Display n-gram probabilities
            for gram in ['unigram', 'bigram', 'trigram']:
                history_col = f'{gram}_history_key'
                prob_col = f'{gram}_transition_prob'

                if history_col in row and row[history_col] is not None:
                    history = row[history_col]
                    print(f"  {gram.capitalize()} transition probability (history: {history}):")

                    pred_prob = row[prob_col]
                    if pred_prob is not None:
                        print(
                            f"    P({history} → {row['predicted_activity']}) = "
                            f"{pred_prob:.4f}"
                        )
                    else:
                        print(f"    P({history} → {row['predicted_activity']}) = N/A")

            # Display softmax probability
            print("  Softmax probability (at violation timestamp):")

            pred_softmax = row['predicted_prob_softmax']
            if pred_softmax is not None:
                print(f"    P({row['predicted_activity']}) = {pred_softmax:.4f}")
            else:
                print(f"    P({row['predicted_activity']}) = N/A")
