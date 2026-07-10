#!/usr/bin/env python3
"""Inspect persistent study artifacts/sentinels and write JSON/CSV status."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


STAGES = ("prepare", "train", "infer", "sktr", "aggregate")
STATUS_VALUES = ("pending", "running", "complete", "failed", "skipped")
CSV_COLUMNS = [
    "study_id",
    "task_id",
    "dataset",
    "official_fold",
    "subset_seed",
    "diffact_fraction",
    "petri_fraction",
    "condition",
    "run_type",
    "stage",
    "status",
    "artifact_scope",
    "detail",
    "heartbeat_timestamp",
    "observed_at_utc",
    "command",
    "log_path",
    "gpu",
    "pid",
    "state_path",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def expected_test_cases(experiment_dir: Path) -> int:
    return len(parse_lines(experiment_dir / "manifests" / "test_cases.txt"))


def train_artifact(task: Mapping[str, Any]) -> Tuple[str, str]:
    experiment_dir = Path(task["experiment_dir"])
    seed = int(task["subset_seed"])
    fraction = int(task["diffact_fraction"])
    run_dir = experiment_dir / "diffact" / f"seed_{seed}" / f"frac_{fraction}"
    config_path = run_dir / "config.json"
    config = read_json(config_path)
    complete = read_json(run_dir / "train_complete.json")
    if not config:
        return "missing", f"missing config {config_path}"
    actual_epochs = int(config.get("num_epochs", -1))
    expected_epochs = int(task["expected_num_epochs"])
    checkpoint = Path(str(complete.get("final_checkpoint", "")))
    completed = bool(complete.get("completed")) and checkpoint.is_file()
    if not completed and actual_epochs > 0:
        naming = (
            f"lowdata_{task['dataset']}_fold{task['official_fold']}_"
            f"seed{seed}_frac{fraction}"
        )
        checkpoint = run_dir / "training" / naming / f"epoch-{actual_epochs - 1}.model"
        completed = checkpoint.is_file()
    if not completed:
        return "missing", f"final checkpoint absent under {run_dir}"
    if actual_epochs != expected_epochs:
        return (
            "runtime_pilot",
            f"checkpoint has {actual_epochs} epochs; final schedule requires {expected_epochs}",
        )
    return "final", f"validated final checkpoint {checkpoint}"


def infer_artifact(task: Mapping[str, Any]) -> Tuple[str, str]:
    train_scope, train_detail = train_artifact(task)
    experiment_dir = Path(task["experiment_dir"])
    output_dir = (
        experiment_dir
        / "diffact"
        / f"seed_{task['subset_seed']}"
        / f"frac_{task['diffact_fraction']}"
        / "softmax_test"
    )
    map_path = output_dir / "video_index_map.txt"
    entries = parse_lines(map_path)
    if not entries:
        return "missing", f"missing or empty {map_path}"
    case_ids = [line.split()[0] for line in entries]
    missing: List[str] = []
    for case_id in case_ids:
        for suffix in (f"{case_id}_raw.npy", f"{case_id}.npy", f"{case_id}_pred.npy"):
            if not (output_dir / suffix).is_file():
                missing.append(suffix)
    if missing:
        return "missing", f"missing {len(missing)} raw/canonical/official output files"
    expected = expected_test_cases(experiment_dir)
    scope = train_scope
    if expected and len(case_ids) < expected:
        scope = "runtime_pilot"
    if scope == "missing":
        scope = "runtime_pilot"
    detail = (
        f"validated raw argmax probabilities, canonical probabilities, and official "
        f"DiffAct predictions for {len(case_ids)}/{expected or '?'} test cases"
    )
    if train_scope == "runtime_pilot":
        detail += f"; {train_detail}"
    return scope, detail


def iter_metric_rows(paths: Iterable[Path]) -> Iterable[Dict[str, str]]:
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                yield from csv.DictReader(handle)
        except (OSError, csv.Error):
            continue


def sktr_artifact(task: Mapping[str, Any]) -> Tuple[str, str]:
    infer_scope, infer_detail = infer_artifact(task)
    experiment_dir = Path(task["experiment_dir"])
    condition = str(task["condition"])
    seed = task["subset_seed"]
    diffact_fraction = task["diffact_fraction"]
    petri_fraction = task.get("petri_fraction")
    if (
        condition == "oracle_test_fold_petri"
        or petri_fraction is None
        or str(petri_fraction).strip().lower() in {"", "none", "null", "nan"}
    ):
        factorized_root = (
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"diffact_frac_{diffact_fraction}"
            / "petri_source_test"
        )
    else:
        factorized_root = (
            experiment_dir
            / "petri"
            / condition
            / f"seed_{seed}"
            / f"diffact_frac_{diffact_fraction}"
            / f"petri_frac_{petri_fraction}"
        )
    legacy_root = (
        experiment_dir
        / "petri"
        / condition
        / f"seed_{seed}"
        / f"frac_{diffact_fraction}"
    )
    roots = [factorized_root, legacy_root]
    paths = sorted(
        {
            path
            for root in roots
            for path in root.glob("**/test*/metrics.csv")
            if path.is_file()
        }
    )
    matching_rows = [
        row
        for row in iter_metric_rows(paths)
        if str(row.get("method", "")).lower().startswith(("petri_", "sktr"))
    ]
    if not paths or not matching_rows:
        return "missing", f"missing final SKTR test metrics under {factorized_root}"
    method_details = sorted({str(row.get("method", "")) for row in matching_rows})
    if "petri_conformance" not in method_details and not any(
        detail.startswith("sktr") for detail in method_details
    ):
        return "missing", f"canonical petri_conformance metrics absent; found {method_details}"
    expected = expected_test_cases(experiment_dir)
    reported_counts = []
    for row in matching_rows:
        try:
            reported_counts.append(int(float(row.get("n_cases", ""))))
        except (TypeError, ValueError):
            pass
    observed = max(reported_counts) if reported_counts else 0
    config_paths = sorted(
        {
            path
            for root in roots
            for path in root.glob("**/test*/postprocess_config.json")
            if path.is_file()
        }
    )
    for config_path in config_paths:
        config = read_json(config_path)
        observed = max(observed, len(config.get("eval_original_cases", [])))
    scope = infer_scope
    row_scopes = {
        str(row.get("artifact_scope", row.get("run_scope", ""))).lower()
        for row in matching_rows
    }
    if any("pilot" in row_scope for row_scope in row_scopes) or any(
        "runtime_pilot" in str(path) for path in paths
    ) or any(
        str(row.get("complete_evaluation_set", "")).lower() in {"false", "0", "no"}
        for row in matching_rows
    ):
        scope = "runtime_pilot"
    if expected and observed and observed < expected:
        scope = "runtime_pilot"
    if scope == "missing":
        scope = "runtime_pilot"
    detail = (
        f"validated canonical SKTR metrics for {observed or '?'}"
        f"/{expected or '?'} test cases"
    )
    if infer_scope == "runtime_pilot":
        detail += f"; {infer_detail}"
    return scope, detail


def prepare_artifact(task: Mapping[str, Any]) -> Tuple[str, str]:
    experiment_dir = Path(task["experiment_dir"])
    required = [
        experiment_dir / "experiment_metadata.json",
        experiment_dir / "manifests" / "split_manifest.csv",
        experiment_dir
        / "manifests"
        / f"seed_{task['subset_seed']}"
        / f"train_cases_frac_{task['diffact_fraction']}.txt",
        experiment_dir / "manifests" / "test_cases.txt",
    ]
    condition = str(task.get("condition", ""))
    discovery = str(task.get("petri_discovery_source", ""))
    if condition == "structural_prior_full_train_petri" or discovery == "full_train":
        required.append(experiment_dir / "manifests" / "train_pool_cases.txt")
    elif condition != "oracle_test_fold_petri" and discovery != "official_test_fold":
        petri_fraction = task.get("petri_fraction", task.get("diffact_fraction"))
        if petri_fraction is not None and str(petri_fraction).strip().lower() not in {
            "",
            "none",
            "null",
            "nan",
        }:
            required.append(
                experiment_dir
                / "manifests"
                / f"seed_{task['subset_seed']}"
                / f"train_cases_frac_{petri_fraction}.txt"
            )
    missing = [path for path in required if not path.is_file()]
    if missing:
        return "missing", f"missing {missing[0]}"
    return "final", "manifest/config artifacts present"


def _petri_fraction_matches(task_value: Any, row_value: Any) -> bool:
    task_text = "" if task_value is None else str(task_value).strip().lower()
    row_text = "" if row_value is None else str(row_value).strip().lower()
    task_null = task_text in {"", "none", "null", "nan", "<na>"}
    row_null = row_text in {"", "none", "null", "nan", "<na>"}
    if task_null or row_null:
        return task_null and row_null
    try:
        return int(float(task_text)) == int(float(row_text))
    except ValueError:
        return task_text == row_text


def aggregate_artifact(task: Mapping[str, Any], study_dir: Path) -> Tuple[str, str]:
    path = study_dir / "aggregation" / "study_paired_deltas.csv"
    if not path.is_file():
        return "missing", f"missing {path}"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return "missing", f"unreadable {path}"
    for row in rows:
        try:
            matches = (
                row.get("dataset") == str(task["dataset"])
                and int(row.get("official_fold", -1)) == int(task["official_fold"])
                and int(row.get("subset_seed", -1)) == int(task["subset_seed"])
                and int(row.get("diffact_fraction", -1)) == int(task["diffact_fraction"])
                and _petri_fraction_matches(task.get("petri_fraction"), row.get("petri_fraction"))
                and row.get("condition", task.get("condition")) == str(task["condition"])
                and row.get("method") == "sktr"
                and row.get("comparator") == "official_diffact"
                and row.get("metric") == "f1@25"
            )
        except ValueError:
            matches = False
        if matches:
            scope = row.get("artifact_scope", "final")
            return scope, f"primary paired F1@25 delta present in {path}"
    return "missing", "primary SKTR-minus-official DiffAct F1@25 delta absent"


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def pid_alive(value: Any) -> bool:
    try:
        pid = int(value)
        os.kill(pid, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def runtime_health(state: Mapping[str, Any], stale_after_seconds: int) -> Tuple[str, str]:
    state_status = str(state.get("status", ""))
    if state_status == "failed":
        return "failed", f"task wrapper exited with returncode {state.get('returncode')}"
    if state_status == "complete":
        return "failed", "task wrapper completed but required artifact is absent"
    if state_status != "running":
        return "pending", "no running or failure sentinel"
    heartbeat = parse_timestamp(state.get("heartbeat_timestamp"))
    age = (
        (datetime.now(timezone.utc) - heartbeat).total_seconds()
        if heartbeat is not None
        else float("inf")
    )
    if age > stale_after_seconds or not pid_alive(state.get("child_pid", state.get("wrapper_pid"))):
        return "failed", f"stale running sentinel (heartbeat age {age:.0f}s)"
    return "running", f"active task heartbeat age {age:.0f}s"


def inspect_task(
    metadata: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    study_dir: Path,
    observed_at: str,
    stale_after_seconds: int,
) -> List[Dict[str, Any]]:
    state_path = Path(task["state_path"])
    state = read_json(state_path)
    artifact_checks = {
        "prepare": prepare_artifact(task),
        "train": train_artifact(task),
        "infer": infer_artifact(task),
        "sktr": sktr_artifact(task),
        "aggregate": aggregate_artifact(task, study_dir),
    }
    runtime_status, runtime_detail = runtime_health(state, stale_after_seconds)
    first_incomplete = next(
        (
            stage
            for stage in ("train", "infer", "sktr")
            if artifact_checks[stage][0] == "missing"
        ),
        None,
    )
    rows: List[Dict[str, Any]] = []
    for stage in STAGES:
        scope, detail = artifact_checks[stage]
        if scope == "final":
            status = "complete"
        elif scope == "runtime_pilot" and task.get("run_type") == "final":
            status = "skipped"
            detail = "Runtime-only partial pilot; final artifact still required. " + detail
        elif scope != "missing":
            status = "complete"
        elif stage == first_incomplete and runtime_status in {"running", "failed"}:
            status = runtime_status
            detail = runtime_detail + ". " + detail
        else:
            status = "pending"
        if status not in STATUS_VALUES:
            raise AssertionError(status)
        rows.append(
            {
                "study_id": metadata["study_id"],
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "official_fold": task["official_fold"],
                "subset_seed": task["subset_seed"],
                "diffact_fraction": task["diffact_fraction"],
                "petri_fraction": task["petri_fraction"],
                "condition": task["condition"],
                "run_type": task["run_type"],
                "stage": stage,
                "status": status,
                "artifact_scope": scope,
                "detail": detail,
                "heartbeat_timestamp": state.get("heartbeat_timestamp", ""),
                "observed_at_utc": observed_at,
                "command": shlex.join([str(part) for part in task["command"]]),
                "log_path": task["log_path"],
                "gpu": task["gpu"],
                "pid": state.get("child_pid", state.get("wrapper_pid", "")),
                "state_path": str(state_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stale-after-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.stale_after_seconds < 1:
        raise ValueError("--stale-after-seconds must be positive")

    study_dir = args.study_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else study_dir
    metadata_path = study_dir / "study_metadata.json"
    metadata = read_json(metadata_path)
    if not metadata:
        raise FileNotFoundError(f"Missing or unreadable study metadata: {metadata_path}")
    observed_at = utc_now()
    rows: List[Dict[str, Any]] = []
    for task in metadata.get("tasks", []):
        rows.extend(
            inspect_task(
                metadata,
                task,
                study_dir=study_dir,
                observed_at=observed_at,
                stale_after_seconds=args.stale_after_seconds,
            )
        )
    counts = Counter(row["status"] for row in rows)
    payload = {
        "schema_version": 1,
        "study_id": metadata["study_id"],
        "spec_sha256": metadata.get("spec_sha256"),
        "observed_at_utc": observed_at,
        "heartbeat_policy_seconds": args.stale_after_seconds,
        "counts": {status: counts.get(status, 0) for status in STATUS_VALUES},
        "rows": rows,
    }
    json_path = output_dir / "study_status.json"
    csv_path = output_dir / "study_status.csv"
    atomic_json(json_path, payload)
    temporary_csv = csv_path.with_suffix(csv_path.suffix + f".tmp.{os.getpid()}")
    temporary_csv.parent.mkdir(parents=True, exist_ok=True)
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    print(f"Study status JSON: {json_path}")
    print(f"Study status CSV: {csv_path}")
    print(" ".join(f"{status}={counts.get(status, 0)}" for status in STATUS_VALUES))


if __name__ == "__main__":
    main()
