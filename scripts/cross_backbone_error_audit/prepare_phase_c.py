#!/usr/bin/env python3
"""Prepare the execution-disabled, provenance-locked Phase-C review study."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from phase_c_common import (
    DATASETS,
    DEFAULT_ASFORMER_CACHE,
    DEFAULT_ASFORMER_LOCAL,
    DEFAULT_ASFORMER_SOURCE,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_OPTION0_STUDY,
    DEFAULT_PARENT_MSTCN2_STUDY,
    DEFAULT_REVIEW_STUDY,
    DEFAULT_STEP0_STUDY,
    PHASE_C_SPEC,
    PROTOCOL_VERSION,
    SOURCE_FILES,
    STALE_PHASE_A_CACHE,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    manifest_digest,
    normalize_case,
    read_nonempty_lines,
    source_provenance,
)
from phase_c_taxonomy import discover_dfg


ASFORMER_OFFICIAL_HEAD = "e1bbe4f3ed083748f91467c51a63ac2a8b9277ad"
ASFORMER_ARCHIVE_SHA256 = "7b255d8cefb90012b192aedef6f10366474acc291e3988e759a0aae3dadf5909"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_REVIEW_STUDY)
    parser.add_argument("--step0-study", type=Path, default=DEFAULT_STEP0_STUDY)
    parser.add_argument("--option0-study", type=Path, default=DEFAULT_OPTION0_STUDY)
    parser.add_argument("--parent-mstcn2-study", type=Path, default=DEFAULT_PARENT_MSTCN2_STUDY)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--asformer-cache", type=Path, default=DEFAULT_ASFORMER_CACHE)
    parser.add_argument("--asformer-source", type=Path, default=DEFAULT_ASFORMER_SOURCE)
    parser.add_argument("--asformer-local", type=Path, default=DEFAULT_ASFORMER_LOCAL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--authorize-phase-c", action="store_true")
    parser.add_argument("--fable-approval-digest")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def add_input(
    rows: list[dict[str, Any]], role: str, path: Path, *, expected_sha256: str | None = None
) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase-C input {role}: {path}")
    observed = file_sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"Upstream digest mismatch for {role}")
    rows.append(
        {
            "role": role,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": observed,
        }
    )


def index_map(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in read_nonempty_lines(path):
        value, case_id = line.split("\t", maxsplit=1)
        result[normalize_case(case_id)] = int(value)
    return result


def manifest_rows_by_path(status: dict[str, Any], study: Path) -> dict[Path, str]:
    return {(study / relative).resolve(): digest for relative, digest in status["output_sha256"].items()}


def git_head(path: Path) -> str:
    return __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    if args.authorize_phase_c != bool(args.fable_approval_digest):
        raise ValueError("Phase-C authorization and Fable digest must be supplied together")
    study = args.study_dir.resolve()
    if study.exists():
        if not args.replace:
            raise FileExistsError(study)
        shutil.rmtree(study)
    for name in ("dfg", "materialized", "status", "logs", "results", "release/audit_suite"):
        (study / name).mkdir(parents=True, exist_ok=True)

    step0 = args.step0_study.resolve()
    option0 = args.option0_study.resolve()
    parent = args.parent_mstcn2_study.resolve()
    diffact_root = args.diffact_root.resolve()
    asformer_cache = args.asformer_cache.resolve()
    asformer_source = args.asformer_source.resolve()
    asformer_local = args.asformer_local.resolve()
    data_root = args.data_root.resolve()
    nested_breakfast_feature_study = Path(
        "/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2"
    ).resolve()
    if git_head(asformer_source) != ASFORMER_OFFICIAL_HEAD:
        raise RuntimeError("Official ASFormer source HEAD drift")
    archive = asformer_cache / "models.zip"
    if file_sha256(archive) != ASFORMER_ARCHIVE_SHA256:
        raise RuntimeError("Official ASFormer archive hash drift")

    step0_reconciliation = json.loads((step0 / "reconciliation" / "selected_reconciliation_complete.json").read_text())
    step0_decision = json.loads((step0 / "reconciliation" / "selected_reconciliation_decision.json").read_text())
    if step0_reconciliation.get("status") != "complete" or step0_decision.get("status") != "complete":
        raise RuntimeError("R6 Phase-B reconciliation is incomplete")
    if step0_decision.get("phase_c_allowed") is not False:
        raise RuntimeError("Phase-B source must remain Phase-C closed")
    option0_complete = json.loads((option0 / "results" / "phase_b_option0_complete.json").read_text())
    option0_decision = json.loads((option0 / "results" / "phase_b_option0_decision.json").read_text())
    if option0_complete.get("status") != "complete" or option0_decision.get("option0_asformer_breakfast_status") != "PASS":
        raise RuntimeError("Option-0 author-ASFormer source is not accepted")

    allowed_roots = [
        step0, option0, parent, diffact_root, asformer_cache, asformer_source,
        asformer_local, data_root, nested_breakfast_feature_study,
    ]
    if any(root == STALE_PHASE_A_CACHE.resolve(strict=False) for root in allowed_roots):
        raise RuntimeError("Stale Phase-A cache cannot be an allowed Phase-C root")
    rows: list[dict[str, Any]] = []
    for root_name, root, files in (
        ("step0", step0, ["study_config.json", "study_metadata.json", "input_manifest.json", "selection/selection_records.json", "reconciliation/selected_reconciliation_complete.json", "reconciliation/selected_reconciliation_decision.json"]),
        ("option0", option0, ["study_config.json", "study_metadata.json", "input_manifest.json", "results/phase_b_option0_complete.json", "results/phase_b_option0_decision.json"]),
        ("parent_mstcn2", parent, ["study_config.json", "study_metadata.json", "input_manifest.json", "results/phase_b_complete.json"]),
    ):
        for relative in files:
            add_input(rows, f"lineage/{root_name}/{relative}", root / relative)
    add_input(rows, "asformer/official/archive", archive, expected_sha256=ASFORMER_ARCHIVE_SHA256)
    for name in ("model.py", "eval.py", "README.md"):
        add_input(rows, f"asformer/official/source/{name}", asformer_source / name)
    for name in ("model.py", "eval.py", "README.md"):
        add_input(rows, f"asformer/local_sensitivity/source/{name}", asformer_local / name)
    nested_breakfast_features = nested_breakfast_feature_study / "oof_input_manifest.json"
    add_input(rows, "data/breakfast/nested_feature_manifest", nested_breakfast_features)
    nested_features_payload = json.loads(nested_breakfast_features.read_text())
    nested_feature_rows = {str(row["case_id"]): row for row in nested_features_payload["features"]}
    if len(nested_feature_rows) != 1712:
        raise RuntimeError("Breakfast nested feature manifest coverage drift")

    case_rows: list[dict[str, Any]] = []
    all_dataset_cases: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    split_cases: dict[tuple[str, int], dict[str, list[str]]] = {}
    for dataset, dataset_config in DATASETS.items():
        root = data_root / dataset
        add_input(rows, f"data/{dataset}/mapping", root / "mapping.txt")
        for fold in range(1, int(dataset_config["folds"]) + 1):
            train_bundle = root / "splits" / f"train.split{fold}.bundle"
            test_bundle = root / "splits" / f"test.split{fold}.bundle"
            add_input(rows, f"data/{dataset}/fold{fold}/train_bundle", train_bundle)
            add_input(rows, f"data/{dataset}/fold{fold}/test_bundle", test_bundle)
            train = [normalize_case(value) for value in read_nonempty_lines(train_bundle)]
            test = [normalize_case(value) for value in read_nonempty_lines(test_bundle)]
            if set(train) & set(test) or len(test) != len(set(test)):
                raise RuntimeError(f"Official split contamination/duplicates: {dataset}/fold{fold}")
            split_cases[(dataset, fold)] = {"train": train, "test": test}
            all_dataset_cases[dataset].update(train)
            all_dataset_cases[dataset].update(test)
            for order, case_id in enumerate(test):
                case_rows.append(
                    {"dataset": dataset, "fold": fold, "case_id": case_id, "test_order": order, "sample_rate": int(dataset_config["sample_rate"])}
                )
        for case_id in sorted(all_dataset_cases[dataset]):
            add_input(rows, f"data/{dataset}/ground_truth/{case_id}", root / "groundTruth" / f"{case_id}.txt")

    # Hash only GTEA/50Salads features directly. Breakfast materialization verifies
    # files through the independently hash-locked nested OOF feature manifest.
    for dataset in ("gtea", "50salads"):
        for case_id in sorted(all_dataset_cases[dataset]):
            add_input(rows, f"data/{dataset}/feature/{case_id}", data_root / dataset / "features" / f"{case_id}.npy")

    case_frame = pd.DataFrame(case_rows).sort_values(["dataset", "fold", "test_order"])
    case_frame.to_csv(study / "case_index.csv", index=False)
    case_index_digest = file_sha256(study / "case_index.csv")

    # Fold-pure DFGs are derived at staging from only hash-locked official train GT.
    dfg_hashes: dict[str, str] = {}
    for (dataset, fold), split in sorted(split_cases.items()):
        mapping = {label: int(index) for index, label in (
            line.split(maxsplit=1) for line in read_nonempty_lines(data_root / dataset / "mapping.txt")
        )}
        background = {index for label, index in mapping.items() if label == "background"}
        sample_rate = int(DATASETS[dataset]["sample_rate"])
        traces = []
        for case_id in split["train"]:
            labels = read_nonempty_lines(data_root / dataset / "groundTruth" / f"{case_id}.txt")
            traces.append([mapping[label] for label in labels][::sample_rate])
        dfg = discover_dfg(traces, background)
        payload = {
            "dataset": dataset, "fold": fold, "sample_rate": sample_rate,
            "discovery_source": "official_full_training_fold_ground_truth_only",
            "train_cases": split["train"],
            "starts": sorted(dfg.starts), "ends": sorted(dfg.ends),
            "edges": [list(edge) for edge in sorted(dfg.edges)],
            "test_gt_used": False,
        }
        payload["dfg_digest"] = canonical_digest(payload)
        path = study / "dfg" / dataset / f"fold{fold}.json"
        atomic_write_json(path, payload)
        dfg_hashes[f"{dataset}/fold{fold}"] = file_sha256(path)

    # Phase-B selected and Breakfast checkpoint-sensitivity exports.
    for dataset, dataset_config in DATASETS.items():
        for fold in range(1, int(dataset_config["folds"]) + 1):
            status_path = step0 / "status" / f"selected_export_{dataset}_fold{fold}.json"
            add_input(rows, f"step0/status/selected_export/{dataset}/fold{fold}", status_path)
            status = json.loads(status_path.read_text())
            expected = manifest_rows_by_path(status, step0)
            for case_id in split_cases[(dataset, fold)]["test"]:
                path = step0 / "exports" / "mstcn2" / "selected" / dataset / f"fold{fold}" / "softmax" / f"{case_id}.npy"
                add_input(rows, f"mstcn2/selected/{dataset}/fold{fold}/probability/{case_id}", path, expected_sha256=expected[path.resolve()])
                if dataset == "breakfast":
                    for arm in ("epoch30", "epoch100"):
                        sensitivity = step0 / "exports" / "mstcn2" / arm / dataset / f"fold{fold}" / "softmax" / f"{case_id}.npy"
                        add_input(rows, f"mstcn2/{arm}/{dataset}/fold{fold}/probability/{case_id}", sensitivity, expected_sha256=expected[sensitivity.resolve()])

    # Original full-train epoch-100 MS-TCN++ predictions for GTEA/50Salads robustness.
    for dataset in ("gtea", "50salads"):
        for fold in range(1, int(DATASETS[dataset]["folds"]) + 1):
            status_path = parent / "status" / f"{dataset}_fold{fold}_complete.json"
            add_input(rows, f"parent_mstcn2/status/{dataset}/fold{fold}", status_path)
            status = json.loads(status_path.read_text())
            expected = manifest_rows_by_path(status, parent)
            root = parent / "cells" / dataset / f"fold{fold}" / "export" / "softmax"
            map_path = root / "video_index_map.txt"
            add_input(rows, f"mstcn2/full_train_epoch100/{dataset}/fold{fold}/video_index_map", map_path, expected_sha256=expected[map_path.resolve()])
            mapping = index_map(map_path)
            for case_id in split_cases[(dataset, fold)]["test"]:
                path = root / f"{mapping[case_id]}.npy"
                add_input(rows, f"mstcn2/full_train_epoch100/{dataset}/fold{fold}/probability/{case_id}", path, expected_sha256=expected[path.resolve()])

    # Existing author-ASFormer Breakfast exports from Option 0.
    for fold in range(1, 5):
        status_path = option0 / "status" / f"fold{fold}.json"
        export_manifest = option0 / "exports" / f"fold{fold}" / "export_manifest.csv"
        add_input(rows, f"option0/status/asformer/breakfast/fold{fold}", status_path)
        add_input(rows, f"option0/export_manifest/asformer/breakfast/fold{fold}", export_manifest)
        status = json.loads(status_path.read_text())
        if file_sha256(export_manifest) != status["export_manifest_sha256"]:
            raise RuntimeError("Option-0 ASFormer export-manifest drift")
        expected = {str(row.case_id): str(row.probability_sha256) for row in pd.read_csv(export_manifest).itertuples(index=False)}
        for case_id in split_cases[("breakfast", fold)]["test"]:
            path = option0 / "exports" / f"fold{fold}" / f"{case_id}.npy"
            add_input(rows, f"asformer/official/breakfast/fold{fold}/probability/{case_id}", path, expected_sha256=expected[case_id])

    # Missing official GTEA/50Salads checkpoints and descriptive local Breakfast arms.
    for dataset in ("gtea", "50salads"):
        for fold in range(1, int(DATASETS[dataset]["folds"]) + 1):
            add_input(
                rows, f"asformer/official/{dataset}/fold{fold}/checkpoint",
                asformer_cache / "extracted" / "models" / dataset / f"split_{fold}" / "epoch-120.model",
            )
    for fold in range(1, 5):
        for epoch in (30, 100):
            add_input(
                rows, f"asformer/local_sensitivity/breakfast/fold{fold}/epoch{epoch}/checkpoint",
                asformer_local / "models" / "breakfast" / f"split_{fold}" / f"epoch-{epoch}.model",
            )

    # Official DiffAct probability and postprocessed prediction streams.
    for dataset, dataset_config in DATASETS.items():
        for fold in range(1, int(dataset_config["folds"]) + 1):
            root = diffact_root / dataset / f"softmax_fold{fold}"
            map_path = root / "video_index_map.txt"
            add_input(rows, f"diffact/official/{dataset}/fold{fold}/video_index_map", map_path)
            add_input(rows, f"diffact/official/{dataset}/fold{fold}/mapping", root / "mapping.txt")
            mapping = index_map(map_path)
            for case_id in split_cases[(dataset, fold)]["test"]:
                value = mapping[case_id]
                add_input(rows, f"diffact/official/{dataset}/fold{fold}/probability/{case_id}", root / f"{value}.npy")
                add_input(rows, f"diffact/official/{dataset}/fold{fold}/prediction/{case_id}", root / f"{value}_pred.npy")

    if len({row["role"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate Phase-C input role")
    rows.sort(key=lambda row: row["role"])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_digest": manifest_digest(rows),
    }
    arms = {
        "primary": [
            {"backbone": backbone, "dataset": dataset, "arm": "selected", "analysis_role": "primary"}
            for dataset in DATASETS for backbone in ("mstcn2", "asformer", "diffact")
        ],
        "secondary": [
            {"backbone": "mstcn2", "dataset": dataset, "arm": "full_train_epoch100", "analysis_role": "secondary_full_train_epoch100"}
            for dataset in ("gtea", "50salads")
        ],
        "sensitivity": [
            {"backbone": backbone, "dataset": "breakfast", "arm": arm, "analysis_role": "sensitivity"}
            for backbone in ("mstcn2", "asformer") for arm in ("epoch30", "epoch100")
        ],
    }
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "Track-1 Phase-C cross-backbone error taxonomy and model-generation test",
        "r6_ruling": "ACCEPTED; Phase C authorized for staging only pending digest review",
        "paths": {
            "step0_study": str(step0), "option0_study": str(option0),
            "parent_mstcn2_study": str(parent), "diffact_root": str(diffact_root),
            "asformer_cache": str(asformer_cache), "asformer_source": str(asformer_source),
            "asformer_local": str(asformer_local), "data_root": str(data_root),
        },
        "asformer_source_contract": {
            "official_git_head": ASFORMER_OFFICIAL_HEAD,
            "official_archive_sha256": ASFORMER_ARCHIVE_SHA256,
            "local_sensitivity_git_head": git_head(asformer_local),
            "local_sensitivity_source_is_descriptive_only": True,
        },
        "allowed_input_roots": [str(path) for path in allowed_roots],
        "forbidden_input_root": str(STALE_PHASE_A_CACHE),
        "phase_c_input_policy": {
            "forbidden_root": str(STALE_PHASE_A_CACHE),
            "allowed_roots_after_digest_review": [str(path) for path in allowed_roots],
            "digest_verification_required": True,
            "exclusive_loader": "phase_c_input_guard.verify_phase_c_study_records",
        },
        "phase_c_spec": PHASE_C_SPEC,
        "analysis_arms": arms,
        "case_index_sha256": case_index_digest,
        "dfg_sha256": dfg_hashes,
        "asformer_materialization_allowed": bool(args.authorize_phase_c),
        "audit_execution_allowed": bool(args.authorize_phase_c),
        "phase_c_allowed": bool(args.authorize_phase_c),
        "fable_approval_digest": args.fable_approval_digest,
        "sealed_studies_modified": False,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study),
        "review_state": "DIGEST_REVIEW_ONLY_NO_EXECUTION" if not args.authorize_phase_c else "APPROVED_READY",
        "input_manifest_digest": manifest["manifest_digest"],
        "source_provenance": provenance,
        "phase_c_launched": False,
        "phase_c_opened": bool(args.authorize_phase_c),
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study / "study_config.json", config)
    atomic_write_json(study / "input_manifest.json", manifest)
    atomic_write_json(study / "study_metadata.json", metadata)
    atomic_write_json(study / "analysis_arms.json", arms)
    (study / "DO_NOT_EDIT.txt").write_text(
        "Immutable Phase-C review package. Materialization and audit execution are disabled.\n"
        "Phase-A stale cache is forbidden. Regenerate after any protocol or source change.\n"
    )

    # Releasable source package: unique modules only, with its own digest manifest.
    release = study / "release" / "audit_suite"
    for relative in SOURCE_FILES:
        source = Path(__file__).resolve().parents[2] / relative
        destination = release / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in (
        "study_config.json", "study_metadata.json", "input_manifest.json",
        "case_index.csv", "analysis_arms.json",
    ):
        shutil.copy2(study / name, release / name)
    shutil.copytree(study / "dfg", release / "dfg", dirs_exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    tasks: list[tuple[str, int, str]] = []
    for dataset in ("gtea", "50salads"):
        tasks.extend((dataset, fold, "official") for fold in range(1, int(DATASETS[dataset]["folds"]) + 1))
    for fold in range(1, 5):
        tasks.extend(("breakfast", fold, arm) for arm in ("epoch30", "epoch100"))
    queues: dict[int, list[tuple[str, int, str]]] = {device: [] for device in range(4)}
    for index, task in enumerate(tasks):
        queues[index % 4].append(task)
    for device, assigned in queues.items():
        lines = [
            "#!/usr/bin/env bash", "set -euo pipefail", f"cd {study}", "mkdir -p logs"
        ]
        for dataset, fold, arm in assigned:
            lines.append(
                f"/usr/bin/python {script_dir / 'materialize_phase_c_asformer.py'} --study-dir {study} "
                f"--dataset {dataset} --fold {fold} --arm {arm} --device {device} "
                f"2>&1 | tee -a {study / 'logs' / f'asformer_gpu{device}.log'}"
            )
        path = study / f"queue_asformer_gpu{device}.sh"
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o755)
        waiter = study / f"wait_and_run_asformer_gpu{device}.sh"
        waiter.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"study={study}\n"
            f"device={device}\n"
            "python - \"$study\" <<'PY'\n"
            "import json,sys\nfrom pathlib import Path\n"
            "c=json.loads((Path(sys.argv[1])/'study_config.json').read_text())\n"
            "assert c['phase_c_allowed'] and c['asformer_materialization_allowed'] and c['fable_approval_digest']\nPY\n"
            "uuid=$(nvidia-smi -i \"$device\" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')\n"
            "free_checks=0\n"
            "while (( free_checks < 2 )); do\n"
            "  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' | grep -Fxq \"$uuid\"; then\n"
            "    free_checks=0\n"
            "  else\n"
            "    free_checks=$((free_checks + 1))\n"
            "  fi\n"
            "  if (( free_checks < 2 )); then sleep 30; fi\n"
            "done\n"
            f"exec bash {path}\n"
        )
        waiter.chmod(0o755)
    run = study / "run_phase_c.sh"
    run.write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\ncd {study}\nmkdir -p logs\n"
        "python - <<'PY'\nimport json\nfrom pathlib import Path\nc=json.loads(Path('study_config.json').read_text())\n"
        "assert c['phase_c_allowed'] and c['audit_execution_allowed'] and c['fable_approval_digest']\nPY\n"
        f"/usr/bin/python {script_dir / 'run_phase_c_audit.py'} --study-dir {study} 2>&1 | tee logs/phase_c.log\n"
    )
    run.chmod(0o755)

    launcher = study / "launch_phase_c_asformer_tmux.sh"
    launcher_lines = [
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"study={study}",
        "python - \"$study\" <<'PY'",
        "import json,sys", "from pathlib import Path",
        "c=json.loads((Path(sys.argv[1])/'study_config.json').read_text())",
        "assert c['phase_c_allowed'] and c['asformer_materialization_allowed'] and c['fable_approval_digest']",
        "PY",
    ]
    for device in range(4):
        session = f"cross_backbone_phasec_asf_g{device}"
        launcher_lines.extend(
            [
                f"if tmux has-session -t {session} 2>/dev/null; then echo 'session exists: {session}' >&2; exit 1; fi",
                f"tmux new-session -d -s {session} 'bash {study / f'wait_and_run_asformer_gpu{device}.sh'}'",
                f"echo {session}",
            ]
        )
    launcher.write_text("\n".join(launcher_lines) + "\n")
    launcher.chmod(0o755)

    (release / "README_RELEASE.md").write_text(
        "# Cross-backbone Phase-C audit suite\n\n"
        "This package freezes the Phase-C protocol, implementation, tests, provenance manifest, "
        "case index, fold-pure DFGs, and generated execution scripts. Raw model exports are not "
        "redistributed; `input_manifest.json` records their immutable paths and SHA-256 hashes.\n\n"
        "The staged review package is deliberately non-executable: all Phase-C permission flags "
        "are false until an approval digest is incorporated by regenerating a new immutable study.\n"
    )
    launch_release = release / "generated_launchers"
    launch_release.mkdir(parents=True, exist_ok=True)
    for path in [
        *(study / f"queue_asformer_gpu{device}.sh" for device in range(4)),
        *(study / f"wait_and_run_asformer_gpu{device}.sh" for device in range(4)),
        launcher,
        run,
    ]:
        shutil.copy2(path, launch_release / path.name)
    release_files = sorted(path for path in release.rglob("*") if path.is_file())
    release_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {"path": str(path.relative_to(release)), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in release_files
        ],
    }
    release_manifest["release_digest"] = canonical_digest(release_manifest["files"])
    atomic_write_json(study / "release" / "release_manifest.json", release_manifest)
    print(study)
    print(f"spec_sha256={metadata['spec_sha256']}")
    print(f"source_digest={provenance['source_digest']}")
    print(f"input_manifest_digest={manifest['manifest_digest']}")
    print(f"release_digest={release_manifest['release_digest']}")
    print(f"inputs={manifest['file_count']} bytes={manifest['total_bytes']}")
    print(f"phase_c_allowed={str(config['phase_c_allowed']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
