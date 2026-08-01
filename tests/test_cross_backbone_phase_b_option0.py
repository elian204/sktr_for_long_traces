from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "cross_backbone_error_audit"
sys.path.insert(0, str(SCRIPT_DIR))

from finalize_phase_b_option0 import reconciliation_status  # noqa: E402
from phase_b_option0_common import (  # noqa: E402
    ASFORMER_ARCHIVE_SHA256,
    OUTER_FOLDS,
    RESIDUAL_MSTCN2_PLAN,
    normalize_case,
    residual_plan,
)


def test_author_archive_digest_and_fold_grid_are_frozen() -> None:
    assert len(ASFORMER_ARCHIVE_SHA256) == 64
    assert OUTER_FOLDS == (1, 2, 3, 4)


def test_bundle_case_normalization() -> None:
    assert normalize_case("path/video.txt") == "video"
    assert normalize_case("video") == "video"


def test_official_pass_removes_asformer_from_residual_training() -> None:
    plan = residual_plan("PASS")
    assert plan["cells"] == 13
    assert all(row["backbone"] == "mstcn2" for row in plan["training_pairs"])
    assert plan["active_gpu_hours"] == RESIDUAL_MSTCN2_PLAN["active_gpu_hours"]


def test_official_failure_preserves_all_seventeen_cells() -> None:
    plan = residual_plan("FAIL")
    assert plan["cells"] == 17
    assert {row["backbone"] for row in plan["training_pairs"]} == {"mstcn2", "asformer"}


def test_reconciliation_bands_match_phase_a() -> None:
    metrics = ("acc", "edit", "f1@10", "f1@25", "f1@50")
    assert reconciliation_status({metric: 0.5 for metric in metrics}) == "PASS"
    assert reconciliation_status({metric: 4.0 for metric in metrics}) == "FAIL"
