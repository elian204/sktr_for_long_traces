from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "independent_acceptance_verifier"
sys.path.insert(0, str(SCRIPT_DIR))

from common import select_budget_rows, validate_oof_corpus  # noqa: E402
from run_v0 import candidate_net_effect, majority_label  # noqa: E402


def synthetic_segments() -> pd.DataFrame:
    rows = []
    for index, (start, end, score) in enumerate([(0, 4, 0.9), (4, 10, 0.8)]):
        row = {
            "segment_id": index,
            "outer_fold": 1,
            "scope": "oof_validation",
            "mode": "official",
            "inner_fold": 1,
            "case_id": "case",
            "segment_index": index,
            "start": start,
            "end": end,
            "length": end - start,
            "predicted_label": 1,
            "correct_label": 1,
            "correct_label_fraction": 1.0,
            "base_score": score,
        }
        for rank in range(1, 6):
            row[f"candidate_rank_{rank}_class_id"] = rank - 1
            row[f"candidate_rank_{rank}_label"] = f"c{rank - 1}"
            row[f"candidate_rank_{rank}_mean_probability"] = 1.0 / rank
        rows.append(row)
    return pd.DataFrame(rows)


def test_exact_budget_and_centered_partial_cutoff() -> None:
    selected = select_budget_rows(synthetic_segments(), total_frames=10, budget=0.6)
    assert int(selected.selected_frames.sum()) == 6
    assert int(selected.is_partial_budget_cutoff.sum()) == 1
    cutoff = selected[selected.is_partial_budget_cutoff].iloc[0]
    assert (int(cutoff.selected_start), int(cutoff.selected_end)) == (6, 8)


def test_budget_tie_order_is_deterministic() -> None:
    frame = synthetic_segments()
    frame["base_score"] = 0.5
    first = select_budget_rows(frame, total_frames=10, budget=0.4)
    second = select_budget_rows(frame.sample(frac=1.0, random_state=7), total_frames=10, budget=0.4)
    assert first[["segment_id", "selected_start", "selected_end"]].to_dict("records") == second[["segment_id", "selected_start", "selected_end"]].to_dict("records")


def test_validate_oof_rejects_outer_scope_before_row_use() -> None:
    frame = pd.concat([synthetic_segments()] * 18097, ignore_index=True)
    frame["segment_id"] = np.arange(len(frame))
    frame.loc[0, "scope"] = "outer_test"
    with pytest.raises(ValueError, match="outer-test"):
        validate_oof_corpus(frame)


def test_majority_contract_distinguishes_ties_and_strict_majorities() -> None:
    label, fraction, strict = majority_label(np.asarray([2, 2, 3], dtype=int))
    assert (label, fraction, strict) == (2, pytest.approx(2 / 3), True)
    label, fraction, strict = majority_label(np.asarray([2, 3], dtype=int))
    assert (label, fraction, strict) == (2, 0.5, False)


def test_net_frame_effect_labels_help_harm_and_lateral() -> None:
    gt = np.asarray([1, 2, 2, 2])
    assert candidate_net_effect(gt, 1, 2) == (1, 3, 2, "helpful")
    assert candidate_net_effect(gt, 2, 1) == (3, 1, -2, "harmful")
    assert candidate_net_effect(gt, 1, 3) == (1, 0, -1, "harmful")
    assert candidate_net_effect(gt, 3, 4) == (0, 0, 0, "lateral")
