#!/usr/bin/env python3
"""Stage the review-gated Phase-B checkpoint-selection and trajectory study."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase_b_selection_common import (
    ANALYSIS_FPS,
    CARVE_FRACTION,
    COMPOSITE_METRICS,
    DEFAULT_DATA_ROOT,
    DEFAULT_OPTION0_STUDY,
    DEFAULT_PARENT_STUDY,
    DEFAULT_REVIEW_STUDY,
    EPOCH_GRID,
    INFORMATIVE_MAX_PEAK_EPOCH,
    INFORMATIVE_MIN_DECLINE_TO_EPOCH100,
    PHASE_C_METRIC_ADDITIONS,
    PHASE_C_SOURCE_DISCLOSURE,
    PROTOCOL_VERSION,
    STALE_CACHE_ROOT,
    atomic_write_json,
    canonical_digest,
    deterministic_carve,
    file_sha256,
    load_json,
    manifest_digest,
    normalize_case,
    read_nonempty_lines,
    source_provenance,
)
from phase_b_training_common import DATASETS, OFFICIAL_CONFIG, QUEUE_ASSIGNMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_REVIEW_STUDY)
    parser.add_argument("--parent-phase-b-study", type=Path, default=DEFAULT_PARENT_STUDY)
    parser.add_argument("--option0-study", type=Path, default=DEFAULT_OPTION0_STUDY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--fable-approval-digest")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step-0 input {role}: {path}")
    rows.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    )


def bundle_text(cases: list[str]) -> str:
    return "".join(f"{normalize_case(case_id)}.txt\n" for case_id in cases)


def main() -> int:
    args = parse_args()
    if args.authorize_execution != bool(args.fable_approval_digest):
        raise ValueError("Execution authorization and Fable approval digest must be supplied together")
    study = args.study_dir.resolve()
    if study.exists():
        if not args.replace:
            raise FileExistsError(study)
        shutil.rmtree(study)
    for name in (
        "carves", "validation_metrics", "retrained_validation_metrics", "test_trajectory",
        "trajectory", "selection", "retrain", "exports", "reconciliation", "results",
        "status", "logs",
    ):
        (study / name).mkdir(parents=True, exist_ok=True)

    parent = args.parent_phase_b_study.resolve()
    parent_config = load_json(parent / "study_config.json")
    parent_metadata = load_json(parent / "study_metadata.json")
    parent_manifest = load_json(parent / "input_manifest.json")
    parent_complete = load_json(parent / "results" / "phase_b_complete.json")
    if parent_config["official_training_config"] != OFFICIAL_CONFIG:
        raise RuntimeError("Parent Phase-B official configuration drift")
    if parent_complete.get("status") != "complete":
        raise RuntimeError("Parent Phase-B is incomplete")
    if parent_config.get("phase_c_allowed"):
        raise RuntimeError("Unexpected open Phase-C state in parent Phase-B")

    option0 = args.option0_study.resolve()
    option0_decision = load_json(option0 / "results" / "phase_b_option0_decision.json")
    if option0_decision.get("option0_asformer_breakfast_status") != "PASS":
        raise RuntimeError("Author-released ASFormer reconciliation is not locked PASS")

    rows: list[dict[str, Any]] = []
    for name in ("study_config.json", "study_metadata.json", "input_manifest.json"):
        add_input(rows, f"parent/{name.removesuffix('.json')}", parent / name)
    for name in (
        "phase_b_complete.json", "phase_b_decision.json", "mstcn2_paper_reconciliation.csv",
    ):
        add_input(rows, f"parent/results/{name}", parent / "results" / name)
    for name in ("study_config.json", "study_metadata.json", "input_manifest.json"):
        add_input(rows, f"option0/{name.removesuffix('.json')}", option0 / name)
    for name in ("phase_b_option0_decision.json", "phase_b_option0_complete.json"):
        add_input(rows, f"option0/results/{name}", option0 / "results" / name)

    tasks: list[dict[str, Any]] = []
    carve_rows: list[dict[str, Any]] = []
    carve_hashes: dict[str, dict[str, str]] = {}
    data_root = args.data_root.resolve()
    parent_roles = {str(row["role"]): row for row in parent_manifest["files"]}
    for device, assignments in QUEUE_ASSIGNMENTS.items():
        for dataset, fold in assignments:
            tasks.append({"dataset": dataset, "fold": fold, "device_lane": device})
    if len(tasks) != 13 or len({(row["dataset"], row["fold"]) for row in tasks}) != 13:
        raise RuntimeError("Step-0 cell matrix drift")

    for task in sorted(tasks, key=lambda row: (row["dataset"], row["fold"])):
        dataset, fold = str(task["dataset"]), int(task["fold"])
        source_train = data_root / dataset / "splits" / f"train.split{fold}.bundle"
        source_test = data_root / dataset / "splits" / f"test.split{fold}.bundle"
        mapping = data_root / dataset / "mapping.txt"
        add_input(rows, f"data/{dataset}/fold{fold}/train", source_train)
        add_input(rows, f"data/{dataset}/fold{fold}/test", source_test)
        if not any(row["role"] == f"data/{dataset}/mapping" for row in rows):
            add_input(rows, f"data/{dataset}/mapping", mapping)
        cases = [normalize_case(value) for value in read_nonempty_lines(source_train)]
        train, validation, audit = deterministic_carve(dataset, fold, cases)
        if set(train).intersection(validation) or set(train).union(validation) != set(cases):
            raise RuntimeError(f"Invalid deterministic carve: {dataset}/fold{fold}")
        carve_dir = study / "carves" / dataset / f"fold{fold}"
        carve_dir.mkdir(parents=True, exist_ok=True)
        train_path = carve_dir / "train_remainder.bundle"
        val_path = carve_dir / "carved_validation.bundle"
        train_path.write_text(bundle_text(train))
        val_path.write_text(bundle_text(validation))
        carve_rows.extend(audit)
        carve_hashes[f"{dataset}/fold{fold}"] = {
            "source_train_sha256": file_sha256(source_train),
            "train_remainder_sha256": file_sha256(train_path),
            "carved_validation_sha256": file_sha256(val_path),
            "n_source_train": len(cases),
            "n_train_remainder": len(train),
            "n_carved_validation": len(validation),
        }
        runtime = parent / "cells" / dataset / f"fold{fold}" / "runtime"
        for name in ("main.py", "model.py", "batch_gen.py", "eval.py", "train.sh"):
            existing_source = next((row for row in rows if row["role"] == f"runtime_source/{name}"), None)
            if existing_source is None:
                add_input(rows, f"runtime_source/{name}", runtime / name)
            elif file_sha256(runtime / name) != existing_source["sha256"]:
                raise RuntimeError(f"Parent runtime source differs by cell: {dataset}/fold{fold}/{name}")
        for epoch in range(1, 101):
            checkpoint = runtime / "models" / dataset / f"split_{fold}" / f"epoch-{epoch}.model"
            add_input(rows, f"checkpoint/{dataset}/fold{fold}/epoch{epoch}", checkpoint)

    audit_path = study / "carves" / "carve_audit.csv"
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(carve_rows[0]))
        writer.writeheader()
        writer.writerows(carve_rows)
    carve_contract = {
        "algorithm": "rank sha256(namespace|dataset|fold|case_id); first ceil(0.15*N)",
        "fraction": CARVE_FRACTION,
        "preserves_source_order_within_partitions": True,
        "audit_sha256": file_sha256(audit_path),
        "folds": carve_hashes,
    }
    carve_contract["contract_digest"] = canonical_digest(carve_contract)
    atomic_write_json(study / "carves" / "carve_contract.json", carve_contract)

    rows.sort(key=lambda row: row["role"])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_digest": manifest_digest(rows),
    }
    branch_rule = {
        "declared_before_probe": True,
        "informative_per_breakfast_fold": {
            "composite": f"mean({', '.join(COMPOSITE_METRICS)})",
            "best_epoch_at_most": INFORMATIVE_MAX_PEAK_EPOCH,
            "minimum_decline_to_epoch100_pp": INFORMATIVE_MIN_DECLINE_TO_EPOCH100,
        },
        "if_all_four_breakfast_folds_informative": {
            "branch": "A_REUSE_EXISTING_CHECKPOINTS",
            "selection": "carved-validation best-composite checkpoint for every one of 13 cells",
            "seen_video_caveat": "carved videos participated in the existing 100-epoch training",
        },
        "otherwise": {
            "branch": "B_RETRAIN_WITH_HELD_OUT_VALIDATION",
            "selection": "retrain all 13 cells on train remainder; select on genuinely held-out carve",
        },
    }
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "Phase-B Step-0 probe, conditional selection, descriptive trajectory, selected export",
        "parent_phase_b_study": str(parent),
        "parent_input_manifest_digest": parent_manifest["manifest_digest"],
        "parent_source_digest": parent_metadata["source_provenance"]["source_digest"],
        "option0_study": str(option0),
        "option0_decision_digest": option0_decision["decision_digest"],
        "data_root": str(data_root),
        "checkpoint_epoch_grid": list(EPOCH_GRID),
        "official_training_config": OFFICIAL_CONFIG,
        "tasks": tasks,
        "queue_assignments": {str(key): [list(value) for value in values] for key, values in QUEUE_ASSIGNMENTS.items()},
        "carve_contract_digest": carve_contract["contract_digest"],
        "step0_branch_rule": branch_rule,
        "test_trajectory_policy": {
            "scope": "descriptive appendix only",
            "same_epoch_grid": list(EPOCH_GRID),
            "computed_in_study_with_checkpoint_digests": True,
            "used_for_selection": False,
        },
        "phase_c_input_policy": {
            "forbidden_root": str(STALE_CACHE_ROOT),
            "allowed_roots_after_digest_review": [
                str(study / "exports"), str(study / "selection"),
                str(option0 / "exports"),
            ],
            "digest_verification_required": True,
        },
        "phase_c_spec": {
            "metric_additions": PHASE_C_METRIC_ADDITIONS,
            "checkpoint_sensitivity": PHASE_C_METRIC_ADDITIONS["breakfast_checkpoint_sensitivity"],
            "source_disclosure": PHASE_C_SOURCE_DISCLOSURE,
            "published_metrics_remain_reference": True,
            "trajectory_explains_deviation_only": True,
        },
        "analysis_fps": ANALYSIS_FPS,
        "step0_validation_inference_allowed": bool(args.authorize_execution),
        "test_trajectory_inference_allowed": bool(args.authorize_execution),
        "conditional_branch_b_training_allowed": bool(args.authorize_execution),
        "selected_export_allowed": bool(args.authorize_execution),
        "selection_may_read_test_trajectory": False,
        "phase_c_allowed": False,
        "fable_approval_digest": args.fable_approval_digest,
        "v1_v2_studies_opened": False,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study),
        "review_state": "APPROVED_READY" if args.authorize_execution else "DIGEST_REVIEW_ONLY_NO_EXECUTION",
        "input_manifest_digest": manifest["manifest_digest"],
        "carve_contract_digest": carve_contract["contract_digest"],
        "source_provenance": provenance,
        "gpu_launched": False,
        "probe_run": False,
        "phase_c_opened": False,
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study / "study_config.json", config)
    atomic_write_json(study / "input_manifest.json", manifest)
    atomic_write_json(study / "study_metadata.json", metadata)
    (study / "DO_NOT_EDIT.txt").write_text(
        "Immutable Phase-B Step-0 review package. No execution before Fable digest approval.\n"
        "Validation selection is firewalled from descriptive test trajectories. Phase C is closed.\n"
    )

    script_dir = Path(__file__).resolve().parent
    for device, assignments in QUEUE_ASSIGNMENTS.items():
        validation = ["#!/usr/bin/env bash", "set -euo pipefail", "mkdir -p logs"]
        trajectory = ["#!/usr/bin/env bash", "set -euo pipefail", "mkdir -p logs"]
        retrain = ["#!/usr/bin/env bash", "set -euo pipefail", "mkdir -p logs"]
        export = ["#!/usr/bin/env bash", "set -euo pipefail", "mkdir -p logs"]
        for dataset, fold in assignments:
            base = (
                f"/usr/bin/python {script_dir / 'run_phase_b_epoch_grid.py'} --study-dir {study} "
                f"--dataset {dataset} --fold {fold} --device {device}"
            )
            validation.append(f"{base} --scope carved_validation 2>&1 | tee -a {study / 'logs' / f'validation_gpu{device}.log'}")
            trajectory.append(f"{base} --scope test_trajectory 2>&1 | tee -a {study / 'logs' / f'trajectory_gpu{device}.log'}")
            retrain.append(
                f"/usr/bin/python {script_dir / 'run_phase_b_retrain_cell.py'} --study-dir {study} "
                f"--dataset {dataset} --fold {fold} --device {device} 2>&1 | tee -a {study / 'logs' / f'retrain_gpu{device}.log'}"
            )
            export.append(
                f"/usr/bin/python {script_dir / 'export_phase_b_selected.py'} --study-dir {study} "
                f"--dataset {dataset} --fold {fold} --device {device} 2>&1 | tee -a {study / 'logs' / f'export_gpu{device}.log'}"
            )
        for name, lines in ((f"queue_validation_gpu{device}.sh", validation), (f"queue_trajectory_gpu{device}.sh", trajectory)):
            path = study / name
            path.write_text("\n".join(lines) + "\n")
            path.chmod(0o755)
        serial = study / f"queue_step0_gpu{device}.sh"
        serial.write_text("\n".join(validation + trajectory[3:]) + "\n")
        serial.chmod(0o755)
        for name, lines in ((f"queue_retrain_gpu{device}.sh", retrain), (f"queue_export_gpu{device}.sh", export)):
            path = study / name
            path.write_text("\n".join(lines) + "\n")
            path.chmod(0o755)
        for kind in ("step0", "retrain", "export"):
            waiter = study / f"wait_{kind}_gpu{device}.sh"
            waiter.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\ndevice=" + str(device) + "\n"
                "while true; do\n"
                "  first=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
                "  if [[ \"$first\" -eq 0 ]]; then\n"
                "    sleep 30\n"
                "    second=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
                f"    if [[ \"$second\" -eq 0 ]]; then exec ./queue_{kind}_gpu${{device}}.sh; fi\n"
                "  fi\n  sleep 30\ndone\n"
            )
            waiter.chmod(0o755)
    (study / "finalize_step0.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {script_dir / 'finalize_phase_b_step0.py'} --study-dir {study}\n"
    )
    (study / "finalize_step0.sh").chmod(0o755)
    (study / "finalize_trajectory.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {script_dir / 'finalize_phase_b_trajectory.py'} --study-dir {study}\n"
    )
    (study / "finalize_trajectory.sh").chmod(0o755)
    (study / "finalize_retrained_selection.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {script_dir / 'finalize_phase_b_selected.py'} --study-dir {study} --mode select-retrained\n"
    )
    (study / "finalize_retrained_selection.sh").chmod(0o755)
    (study / "finalize_selected_reconciliation.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {script_dir / 'finalize_phase_b_selected.py'} --study-dir {study} --mode reconcile\n"
    )
    (study / "finalize_selected_reconciliation.sh").chmod(0o755)
    launcher = study / "launch_tmux.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs\n"
        "python - <<'PY'\nimport json\nfrom pathlib import Path\nc=json.loads(Path('study_config.json').read_text())\n"
        "assert c['fable_approval_digest'] and c['step0_validation_inference_allowed']\n"
        "assert not c['phase_c_allowed'] and not c['selection_may_read_test_trajectory']\nPY\n"
        + "\n".join(
            f"tmux new-session -d -s cb_step0_g{device} 'cd {study} && ./wait_step0_gpu{device}.sh'"
            for device in QUEUE_ASSIGNMENTS
        )
        + "\n"
    )
    launcher.chmod(0o755)
    for kind, guard in (
        ("retrain", "assert d['selected_branch'] == 'B_RETRAIN_WITH_HELD_OUT_VALIDATION'"),
        ("export", "assert Path('selection/selection_records.json').is_file()"),
    ):
        path = study / f"launch_{kind}_tmux.sh"
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs\n"
            "python - <<'PY'\nimport json\nfrom pathlib import Path\n"
            "c=json.loads(Path('study_config.json').read_text())\nassert c['fable_approval_digest'] and not c['phase_c_allowed']\n"
            + ("d=json.loads(Path('selection/step0_decision.json').read_text())\n" if kind == "retrain" else "")
            + guard + "\nPY\n"
            + "\n".join(
                f"tmux new-session -d -s cb_{kind}_g{device} 'cd {study} && ./wait_{kind}_gpu{device}.sh'"
                for device in QUEUE_ASSIGNMENTS
            )
            + "\n"
        )
        path.chmod(0o755)
    print(study)
    print(f"spec_sha256={metadata['spec_sha256']}")
    print(f"source_digest={provenance['source_digest']}")
    print(f"input_manifest_digest={manifest['manifest_digest']}")
    print(f"carve_contract_digest={carve_contract['contract_digest']}")
    print(f"inputs={manifest['file_count']} bytes={manifest['total_bytes']}")
    print("execution_authorized=false" if not args.authorize_execution else "execution_authorized=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
