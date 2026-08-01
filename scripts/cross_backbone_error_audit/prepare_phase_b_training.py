#!/usr/bin/env python3
"""Prepare the approved immutable MS-TCN++ Phase-B retraining study."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase_b_training_common import (
    DATASETS,
    OFFICIAL_CONFIG,
    OFFICIAL_MSTCN2_HEAD,
    OFFICIAL_SOURCE_FILES,
    PROTOCOL_VERSION,
    QUEUE_ASSIGNMENTS,
    atomic_write_json,
    canonical_digest,
    compatibility_patched_model,
    file_sha256,
    git_clean,
    git_head,
    source_provenance,
)


DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_mstcn2_v1"
)
DEFAULT_OPTION0_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_option0_review_v1"
)
DEFAULT_MSTCN2_SOURCE = Path("/home/dsi/eli-bogdanov/MS-TCN2_official_phase_b")
DEFAULT_DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--option0-dir", type=Path, default=DEFAULT_OPTION0_DIR)
    parser.add_argument("--mstcn2-source", type=Path, default=DEFAULT_MSTCN2_SOURCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--fable-approval-digest")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase-B input {role}: {path}")
    rows.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    )


def main() -> int:
    args = parse_args()
    if args.authorize_training != bool(args.fable_approval_digest):
        raise ValueError("Training authorization and Fable approval digest must be supplied together")
    study_dir = args.study_dir.resolve()
    if study_dir.exists():
        if not args.replace:
            raise FileExistsError(study_dir)
        shutil.rmtree(study_dir)
    for name in ("cells", "logs", "results", "status"):
        (study_dir / name).mkdir(parents=True, exist_ok=True)

    official = args.mstcn2_source.resolve()
    if git_head(official) != OFFICIAL_MSTCN2_HEAD or not git_clean(official):
        raise RuntimeError("MS-TCN++ source must be a clean worktree at the official Git HEAD")
    option0 = args.option0_dir.resolve()
    decision_path = option0 / "results" / "phase_b_option0_decision.json"
    decision = __import__("json").loads(decision_path.read_text())
    if decision.get("option0_asformer_breakfast_status") != "PASS":
        raise RuntimeError("Author-ASFormer reconciliation did not pass")
    if decision.get("mstcn2_official_checkpoint_status") != "UNAVAILABLE_IN_OFFICIAL_REPOSITORY":
        raise RuntimeError("MS-TCN++ official-checkpoint inventory contract changed")
    if int(decision["residual_training_plan"]["cells"]) != 13:
        raise RuntimeError("Residual Phase-B cell count is not 13")

    rows: list[dict[str, Any]] = []
    for name in (
        "phase_b_option0_decision.json",
        "phase_b_option0_complete.json",
        "asformer_breakfast_per_fold.csv",
    ):
        add_input(rows, f"option0/results/{name}", option0 / "results" / name)
    add_input(rows, "option0/input_manifest", option0 / "input_manifest.json")
    for name in OFFICIAL_SOURCE_FILES:
        add_input(rows, f"official_source/{name}", official / name)

    data_root = args.data_root.resolve()
    for dataset, dataset_config in DATASETS.items():
        root = data_root / dataset
        add_input(rows, f"data/{dataset}/mapping", root / "mapping.txt")
        for fold in range(1, int(dataset_config["folds"]) + 1):
            add_input(rows, f"data/{dataset}/fold{fold}/train", root / "splits" / f"train.split{fold}.bundle")
            add_input(rows, f"data/{dataset}/fold{fold}/test", root / "splits" / f"test.split{fold}.bundle")
        for path in sorted((root / "features").glob("*.npy")):
            add_input(rows, f"data/{dataset}/feature/{path.name}", path)
        for path in sorted((root / "groundTruth").glob("*")):
            if path.is_file():
                add_input(rows, f"data/{dataset}/ground_truth/{path.name}", path)
    rows.sort(key=lambda row: row["role"])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_digest": canonical_digest(
            [{"role": row["role"], "sha256": row["sha256"]} for row in rows]
        ),
    }

    tasks = [
        {"dataset": dataset, "fold": fold, "device_lane": device}
        for device, assignments in QUEUE_ASSIGNMENTS.items()
        for dataset, fold in assignments
    ]
    if len(tasks) != 13 or len({(row["dataset"], row["fold"]) for row in tasks}) != 13:
        raise RuntimeError("Phase-B queue matrix drift")
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "MS-TCN++ official-config retraining only",
        "option0_dir": str(option0),
        "option0_decision_digest": decision["decision_digest"],
        "official_source": str(official),
        "official_source_head": OFFICIAL_MSTCN2_HEAD,
        "data_root": str(data_root),
        "official_training_config": OFFICIAL_CONFIG,
        "runtime_compatibility_patch": {
            "file": "model.py",
            "reason": "official Git HEAD contains a stray MS_TCB token that makes Python parsing fail",
            "scope": "remove the stray token and restore indentation of MS_TCN2.__init__; no architecture or arithmetic change",
        },
        "tasks": tasks,
        "queue_assignments": {str(key): [list(value) for value in values] for key, values in QUEUE_ASSIGNMENTS.items()},
        "gpu_training_allowed": bool(args.authorize_training),
        "phase_c_allowed": False,
        "sealed_studies_opened": False,
        "fable_approval_digest": args.fable_approval_digest,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "review_state": "PHASE_B_APPROVED_READY" if args.authorize_training else "PHASE_B_REVIEW_ONLY",
        "input_manifest_digest": manifest["manifest_digest"],
        "source_provenance": provenance,
        "official_source_head": OFFICIAL_MSTCN2_HEAD,
        "gpu_launched": False,
        "phase_c_opened": False,
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study_dir / "study_config.json", config)
    atomic_write_json(study_dir / "input_manifest.json", manifest)
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    (study_dir / "DO_NOT_EDIT.txt").write_text(
        "Immutable approved MS-TCN++ Phase-B retraining study.\n"
        "Only task outputs, logs, status, and final results may be written.\n"
        "Phase C remains closed pending reconciliation review.\n"
    )

    for task in tasks:
        dataset, fold = str(task["dataset"]), int(task["fold"])
        runtime = study_dir / "cells" / dataset / f"fold{fold}" / "runtime"
        runtime.mkdir(parents=True)
        (runtime / "logs").mkdir()
        for name in OFFICIAL_SOURCE_FILES:
            if name == "model.py":
                (runtime / name).write_text(compatibility_patched_model(official / name))
            else:
                shutil.copy2(official / name, runtime / name)
        os.symlink(data_root, runtime / "data", target_is_directory=True)

    script_dir = Path(__file__).resolve().parent
    (study_dir / "preflight.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {script_dir / 'verify_phase_b_training.py'} --study-dir {study_dir} --full-hash "
        f"2>&1 | tee {study_dir / 'logs' / 'preflight.log'}\n"
    )
    (study_dir / "preflight.sh").chmod(0o755)
    for device, assignments in QUEUE_ASSIGNMENTS.items():
        queue = study_dir / f"queue_gpu{device}.sh"
        commands = ["#!/usr/bin/env bash", "set -euo pipefail", "mkdir -p logs"]
        for dataset, fold in assignments:
            commands.append(
                f"/usr/bin/python {script_dir / 'run_phase_b_training_task.py'} --study-dir {study_dir} "
                f"--dataset {dataset} --fold {fold} --device {device} 2>&1 | tee -a "
                f"{study_dir / 'logs' / f'gpu{device}.log'}"
            )
        queue.write_text("\n".join(commands) + "\n")
        queue.chmod(0o755)
        waiter = study_dir / f"wait_gpu{device}.sh"
        waiter.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"device={device}\n"
            "while true; do\n"
            "  first=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
            "  if [[ \"$first\" -eq 0 ]]; then\n"
            "    sleep 30\n"
            "    second=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
            "    if [[ \"$second\" -eq 0 ]]; then exec ./queue_gpu${device}.sh; fi\n"
            "  fi\n"
            "  sleep 30\n"
            "done\n"
        )
        waiter.chmod(0o755)
    finalize = study_dir / "finalize.sh"
    finalize.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {script_dir / 'finalize_phase_b_training.py'} --study-dir {study_dir} "
        f"2>&1 | tee {study_dir / 'logs' / 'finalize.log'}\n"
    )
    finalize.chmod(0o755)
    final_waiter = study_dir / "wait_and_finalize.sh"
    final_waiter.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "while true; do\n"
        "  if ls status/*_failed.json >/dev/null 2>&1; then echo 'A Phase-B task failed'; exit 1; fi\n"
        "  count=$(find status -maxdepth 1 -name '*_fold*_complete.json' | wc -l)\n"
        "  if [[ \"$count\" -eq 13 ]]; then exec ./finalize.sh; fi\n"
        "  sleep 60\n"
        "done\n"
    )
    final_waiter.chmod(0o755)
    launcher = study_dir / "launch_tmux.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs\n./preflight.sh\n"
        + "\n".join(
            f"tmux new-session -d -s cb_mstcn2_g{device} 'cd {study_dir} && ./wait_gpu{device}.sh'"
            for device in QUEUE_ASSIGNMENTS
        )
        + "\n"
        + f"tmux new-session -d -s cb_mstcn2_finalize 'cd {study_dir} && ./wait_and_finalize.sh'\n"
    )
    launcher.chmod(0o755)
    print(study_dir)
    print(f"spec_sha256={metadata['spec_sha256']}")
    print(f"source_digest={provenance['source_digest']}")
    print(f"input_manifest_digest={manifest['manifest_digest']}")
    print(f"inputs={manifest['file_count']} bytes={manifest['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
