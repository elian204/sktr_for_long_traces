#!/usr/bin/env python3
"""
Run ceiling diagnostics one case per subprocess with timeout/memory isolation.

This is meant for datasets such as 50Salads where collapsed-sequence
conformance can grow large.  The harness is intentionally conservative:
timeout, memory-limit, non-zero exit, missing output, and malformed output are
all recorded as incomplete statuses.  They are never converted into a
"non-fit" or "not accepted" research finding.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


RUN_DIRS: Dict[str, Path] = {
    "50salads": Path(
        "/data1/eli-bogdanov/sktr_runs/"
        "diffact_50salads_allfolds_resumable_6ba8868_chunk11"
    ),
    "gtea": Path(
        "/data1/eli-bogdanov/sktr_runs/"
        "diffact_gtea_allfolds_resumable_6ba8868_chunk11_w7"
    ),
    "breakfast": Path(
        "/data1/eli-bogdanov/sktr_runs/"
        "diffact_breakfast_unique199_f14fd99_chunk11_w10"
    ),
}


@dataclass(frozen=True)
class CaseTask:
    dataset: str
    fold: int
    case_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run analyze_sktr_ceiling.py in isolated one-case subprocesses."
    )
    parser.add_argument("--dataset", choices=sorted(RUN_DIRS), default="50salads")
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        default=(
            "/data1/eli-bogdanov/sktr_runs/"
            "sktr_ceiling_50salads_isolated_v1"
        ),
    )
    parser.add_argument(
        "--fitness-chunking",
        choices=["full", "run"],
        default="run",
        help="Forwarded to analyze_sktr_ceiling.py.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Wall-clock limit per case subprocess.",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=6144,
        help="RSS limit per case subprocess. 0 disables the monitor limit.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Subprocess polling interval.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed per-case status/output when present.",
    )
    parser.add_argument(
        "--keep-running-after-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue to the next case after incomplete/error status.",
    )
    return parser.parse_args()


def discover_tasks(dataset: str, folds: Sequence[int], case_ids: Optional[Sequence[str]], case_limit: Optional[int]) -> List[CaseTask]:
    run_dir = RUN_DIRS[dataset]
    wanted = {str(c) for c in case_ids} if case_ids else None
    tasks: List[CaseTask] = []
    for fold in folds:
        case_dir = run_dir / "case_outputs" / f"{dataset}_fold{fold}"
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        ids = sorted(p.stem for p in case_dir.glob("*.csv"))
        if wanted is not None:
            ids = [case_id for case_id in ids if case_id in wanted]
        for case_id in ids:
            tasks.append(CaseTask(dataset=dataset, fold=int(fold), case_id=str(case_id)))
    if wanted is not None:
        found = {(task.case_id) for task in tasks}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Requested case IDs not found in selected folds: {missing}")
    if case_limit is not None:
        tasks = tasks[:case_limit]
    return tasks


def read_rss_mb(pid: int) -> Optional[float]:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / 1024.0
    return None


def kill_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def command_for_case(task: CaseTask, case_out_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_sktr_ceiling.py"),
        "--datasets",
        task.dataset,
        "--fold",
        str(task.fold),
        "--case-ids",
        task.case_id,
        "--skip-oracle",
        "--fitness-chunking",
        "run",
        "--out-dir",
        str(case_out_dir),
    ]


def status_path_for(out_dir: Path, task: CaseTask) -> Path:
    return out_dir / "statuses" / f"{task.dataset}_fold{task.fold}_case{task.case_id}.json"


def case_work_dir(out_dir: Path, task: CaseTask) -> Path:
    return out_dir / "case_runs" / f"{task.dataset}_fold{task.fold}" / f"case_{task.case_id}"


def load_completed_status(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if data.get("status") == "completed":
        case_csv = data.get("case_csv")
        if case_csv and Path(case_csv).is_file():
            return data
    return None


def validate_completed_output(task: CaseTask, case_dir: Path) -> Path:
    case_csv = case_dir / f"{task.dataset}_fold{task.fold}_ceiling_cases.csv"
    if not case_csv.is_file():
        raise FileNotFoundError(f"missing one-case output: {case_csv}")
    df = pd.read_csv(case_csv)
    if len(df) != 1:
        raise ValueError(f"{case_csv} expected exactly one row, got {len(df)}")
    row = df.iloc[0]
    if str(row["case_id"]) != str(task.case_id):
        raise ValueError(
            f"{case_csv} case mismatch: row={row['case_id']} expected={task.case_id}"
        )
    return case_csv


def run_one_case(args: argparse.Namespace, out_dir: Path, task: CaseTask) -> Dict[str, Any]:
    status_path = status_path_for(out_dir, task)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        completed = load_completed_status(status_path)
        if completed is not None:
            completed["resumed"] = True
            return completed

    work_dir = case_work_dir(out_dir, task)
    work_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"
    command = command_for_case(task, work_dir)
    command[command.index("--fitness-chunking") + 1] = args.fitness_chunking

    started = time.time()
    max_rss_mb = 0.0
    stop_reason: Optional[str] = None

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        while proc.poll() is None:
            elapsed = time.time() - started
            rss_mb = read_rss_mb(proc.pid)
            if rss_mb is not None:
                max_rss_mb = max(max_rss_mb, rss_mb)
            if args.memory_limit_mb > 0 and rss_mb is not None and rss_mb > args.memory_limit_mb:
                stop_reason = "memory_limit"
                kill_process_group(proc)
                break
            if elapsed > args.timeout_seconds:
                stop_reason = "timeout"
                kill_process_group(proc)
                break
            time.sleep(args.poll_seconds)

    elapsed = time.time() - started
    return_code = proc.poll()
    status = "completed"
    error: Optional[str] = None
    case_csv: Optional[Path] = None

    if stop_reason is not None:
        status = f"incomplete_{stop_reason}"
    elif return_code != 0:
        status = "incomplete_error"
        error = f"subprocess exited with code {return_code}"
    else:
        try:
            case_csv = validate_completed_output(task, work_dir)
        except Exception as exc:  # noqa: BLE001 - status artifact must capture any validation failure
            status = "incomplete_invalid_output"
            error = str(exc)

    result: Dict[str, Any] = {
        "dataset": task.dataset,
        "fold": task.fold,
        "case_id": task.case_id,
        "status": status,
        "is_complete": status == "completed",
        "runtime_seconds": elapsed,
        "max_rss_mb": max_rss_mb,
        "return_code": return_code,
        "stop_reason": stop_reason,
        "error": error,
        "command": command,
        "work_dir": str(work_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "case_csv": str(case_csv) if case_csv is not None else None,
        "contributes_fitness": status == "completed",
        "incomplete_not_interpreted_as_nonfit": True,
        "resumed": False,
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_aggregate(out_dir: Path, statuses: Sequence[Dict[str, Any]]) -> None:
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out_dir / "case_status.csv", index=False)

    completed_frames: List[pd.DataFrame] = []
    for status in statuses:
        if status.get("status") != "completed":
            continue
        case_csv = status.get("case_csv")
        if not case_csv:
            continue
        df = pd.read_csv(case_csv)
        for key in ("status", "runtime_seconds", "max_rss_mb"):
            df[key] = status.get(key)
        completed_frames.append(df)
    if completed_frames:
        pd.concat(completed_frames, ignore_index=True).to_csv(
            out_dir / "completed_ceiling_cases.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(out_dir / "completed_ceiling_cases.csv", index=False)

    summary = {
        "n_cases": int(len(status_df)),
        "n_completed": int((status_df["status"] == "completed").sum()) if not status_df.empty else 0,
        "n_incomplete": int((status_df["status"] != "completed").sum()) if not status_df.empty else 0,
        "status_counts": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
        "incomplete_is_not_nonfit": True,
        "outputs": {
            "case_status_csv": str(out_dir / "case_status.csv"),
            "completed_ceiling_cases_csv": str(out_dir / "completed_ceiling_cases.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(args.dataset, args.folds, args.case_ids, args.case_limit)
    if not tasks:
        raise ValueError("No cases selected")

    statuses: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        print(
            f"[{idx}/{len(tasks)}] {task.dataset} fold {task.fold} case {task.case_id}",
            flush=True,
        )
        status = run_one_case(args, out_dir, task)
        statuses.append(status)
        print(
            f"  status={status['status']} runtime={status['runtime_seconds']:.1f}s "
            f"max_rss={status['max_rss_mb']:.1f}MB",
            flush=True,
        )
        write_aggregate(out_dir, statuses)
        if status["status"] != "completed" and not args.keep_running_after_failure:
            break

    write_aggregate(out_dir, statuses)
    print(f"Wrote isolated harness outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
