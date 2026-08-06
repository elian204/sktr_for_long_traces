from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cross_backbone_error_audit"
sys.path.insert(0, str(SCRIPT))

from phase_b_selection_common import (  # noqa: E402
    CARVE_FRACTION,
    EPOCH_GRID,
    PHASE_C_METRIC_ADDITIONS,
    SOURCE_FILES,
    STALE_CACHE_ROOT,
    assert_not_stale_cache,
    choose_step0_branch,
    deterministic_carve,
    informative_breakfast_fold,
    select_best_checkpoint,
    validate_phase_c_inputs,
)
from phase_c_input_guard import verify_phase_c_study_records  # noqa: E402


def metric_rows(best_epoch: int = 30, best_value: float = 60.0, epoch100_value: float = 54.0) -> list[dict]:
    rows = []
    for epoch in EPOCH_GRID:
        value = best_value if epoch == best_epoch else epoch100_value if epoch == 100 else 40.0
        rows.append({"epoch": epoch, "acc": value, "edit": value, "f1@10": value, "f1@25": value, "f1@50": value})
    return rows


def test_epoch_grid_is_every_five_plus_96_to_100() -> None:
    assert EPOCH_GRID == tuple(sorted(set(range(5, 101, 5)) | set(range(96, 101))))
    assert len(EPOCH_GRID) == 24


def test_hash_carve_is_deterministic_disjoint_and_preserves_input_order() -> None:
    cases = [f"case_{index:03d}" for index in range(101)]
    train_a, validation_a, audit_a = deterministic_carve("breakfast", 1, cases)
    train_b, validation_b, audit_b = deterministic_carve("breakfast", 1, cases)
    assert (train_a, validation_a, audit_a) == (train_b, validation_b, audit_b)
    assert len(validation_a) == math.ceil(CARVE_FRACTION * len(cases))
    assert not set(train_a) & set(validation_a)
    assert set(train_a) | set(validation_a) == set(cases)
    assert train_a == [case for case in cases if case in set(train_a)]
    assert validation_a == [case for case in cases if case in set(validation_a)]


def test_hash_carve_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Invalid training bundle"):
        deterministic_carve("gtea", 1, ["same.txt", "same"])


def test_checkpoint_tie_breaks_to_earlier_epoch() -> None:
    rows = metric_rows()
    for row in rows:
        row.update({"edit": 50.0, "f1@10": 50.0, "f1@25": 50.0})
    assert select_best_checkpoint(rows)["epoch"] == min(EPOCH_GRID)


def test_checkpoint_grid_must_be_complete() -> None:
    with pytest.raises(RuntimeError, match="grid is incomplete"):
        select_best_checkpoint(metric_rows()[:-1])


def test_informative_rule_requires_early_peak_and_five_point_decline() -> None:
    assert informative_breakfast_fold(metric_rows(best_epoch=30, best_value=60, epoch100_value=55))["informative"]
    assert not informative_breakfast_fold(metric_rows(best_epoch=90, best_value=60, epoch100_value=50))["informative"]
    assert not informative_breakfast_fold(metric_rows(best_epoch=30, best_value=60, epoch100_value=55.01))["informative"]


def test_branch_a_requires_all_four_breakfast_folds_informative() -> None:
    evidence = {fold: {"informative": True} for fold in range(1, 5)}
    assert choose_step0_branch(evidence) == "A_REUSE_EXISTING_CHECKPOINTS"
    evidence[3]["informative"] = False
    assert choose_step0_branch(evidence) == "B_RETRAIN_WITH_HELD_OUT_VALIDATION"
    with pytest.raises(RuntimeError, match="incomplete"):
        choose_step0_branch({1: {"informative": True}})


def test_stale_phase_a_cache_is_rejected_directly_and_through_symlink(tmp_path: Path) -> None:
    assert_not_stale_cache(STALE_CACHE_ROOT.parent / "safe")
    with pytest.raises(RuntimeError, match="Forbidden stale"):
        assert_not_stale_cache(STALE_CACHE_ROOT / "cell.npy")
    link = tmp_path / "legacy"
    link.symlink_to(STALE_CACHE_ROOT, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Forbidden stale"):
        assert_not_stale_cache(link / "cell.npy")


def test_phase_c_inputs_must_stay_under_digest_approved_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "selected"
    allowed.mkdir()
    accepted = allowed / "fold.npy"
    accepted.write_bytes(b"locked")
    assert validate_phase_c_inputs([accepted], [allowed]) == [accepted.resolve()]
    outside = tmp_path / "outside.npy"
    outside.write_bytes(b"no")
    with pytest.raises(RuntimeError, match="outside digest-approved roots"):
        validate_phase_c_inputs([outside], [allowed])


def test_phase_c_loader_uses_frozen_study_roots_not_caller_roots(tmp_path: Path) -> None:
    study = tmp_path / "study"
    allowed = study / "exports"
    allowed.mkdir(parents=True)
    artifact = allowed / "fold.npy"
    artifact.write_bytes(b"locked")
    import hashlib, json
    (study / "study_config.json").write_text(json.dumps({"phase_c_input_policy": {"digest_verification_required": True, "allowed_roots_after_digest_review": [str(allowed)]}}))
    records = [{"path": str(artifact), "sha256": hashlib.sha256(b"locked").hexdigest()}]
    assert verify_phase_c_study_records(study, records) == [artifact.resolve()]
    outside = tmp_path / "outside.npy"
    outside.write_bytes(b"locked")
    with pytest.raises(RuntimeError, match="outside digest-approved roots"):
        verify_phase_c_study_records(study, [{"path": str(outside), "sha256": records[0]["sha256"]}])


def test_selection_finalizer_has_no_test_trajectory_file_read() -> None:
    source = (SCRIPT / "finalize_phase_b_step0.py").read_text()
    assert 'study / "test_trajectory"' not in source
    assert 'pd.read_csv(study / "test_trajectory"' not in source


def test_phase_c_spec_adds_absolute_rates_and_checkpoint_sensitivity() -> None:
    assert PHASE_C_METRIC_ADDITIONS["absolute_error_rates"] == ["errors_per_gt_segment", "errors_per_minute"]
    sensitivity = PHASE_C_METRIC_ADDITIONS["breakfast_checkpoint_sensitivity"]
    assert sensitivity["backbones"] == ["mstcn2", "asformer"]
    assert sensitivity["checkpoints"] == ["selected", "epoch100", "epoch30"]
    assert sensitivity["asformer_selected_definition"] == "author release epoch120"


def test_review_source_contract_is_complete() -> None:
    assert all((ROOT / relative).is_file() for relative in SOURCE_FILES)


def test_readme_discloses_seen_video_caveat_and_keeps_v1_v2_closed() -> None:
    readme = (SCRIPT / "README.md").read_text()
    normalized = " ".join(readme.split())
    assert "seen-video diagnostic" in readme
    assert "No test prediction or metric can enter" in readme
    assert "V1/V2 repair studies remain closed and immutable" in normalized
