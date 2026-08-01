#!/usr/bin/env python3
"""Prepare a review-only V2 corrected-sampler B0 validity study."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2_common import (
    B1_KILL_BARS,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_SELECTOR_DIR,
    DEFAULT_STUDY_DIR,
    DEFAULT_V0_DIR,
    EARLY_STOP_MIN_SAMPLES,
    EARLY_STOP_PATIENCE,
    HALO_WIDTHS,
    INNER_FOLDS,
    MAX_SAMPLES,
    OUTER_FOLDS,
    PROTOCOL_VERSION,
    RESTART_TIMES,
    atomic_write_json,
    canonical_digest,
    checkpoint_and_config,
    file_sha256,
    load_json,
    source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--v0-dir", type=Path, default=DEFAULT_V0_DIR)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--authorize-b0", action="store_true")
    parser.add_argument("--authorize-b1", action="store_true")
    parser.add_argument("--fable-approval-digest")
    parser.add_argument(
        "--v1-study-dir",
        type=Path,
        default=Path(
            "/data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v1_nested_oof_v2"
        ),
    )
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V2 B0 input {role}: {path}")
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
    if args.authorize_b1 and not args.authorize_b0:
        raise ValueError("B1 authorization requires B0 authorization")
    if args.authorize_b0 and not args.fable_approval_digest:
        raise ValueError("Authorized B0 generation requires --fable-approval-digest")
    if not args.authorize_b0 and args.fable_approval_digest:
        raise ValueError("Review-only B0 must not claim an approval digest")
    study_dir = args.study_dir.resolve()
    if study_dir.exists():
        if not args.replace:
            raise FileExistsError(study_dir)
        shutil.rmtree(study_dir)
    for name in ("logs", "results", "status"):
        (study_dir / name).mkdir(parents=True, exist_ok=True)

    v0 = args.v0_dir.resolve()
    selector = args.selector_dir.resolve()
    diffact = args.diffact_root.resolve()
    v0_input = load_json(v0 / "input_manifest.json")
    v0_paths = {str(row["role"]): Path(row["path"]) for row in v0_input["files"]}
    rows: list[dict[str, Any]] = []
    for name in ("candidate_corpus.csv", "flagged_oof_spans.csv", "v0_complete.json"):
        add_input(rows, f"v0/results/{name}", v0 / "results" / name)
    add_input(rows, "v0/input_manifest", v0 / "input_manifest.json")
    add_input(rows, "v0/oof_segment_corpus", v0_paths["selector/oof_segment_corpus"])
    for role, path in sorted(v0_paths.items()):
        if role.startswith("ground_truth/"):
            add_input(rows, f"v0_nested/{role}", path)
    add_input(rows, "features/nested_manifest", args.feature_manifest.resolve())
    add_input(rows, "breakfast/mapping", Path("/home/dsi/eli-bogdanov/data/data/breakfast/mapping.txt"))
    for name in ("model.py", "main.py", "dataset.py", "utils.py"):
        add_input(rows, f"diffact/source/{name}", diffact / name)
    task_rows: list[dict[str, Any]] = []
    for outer in OUTER_FOLDS:
        for inner in INNER_FOLDS:
            checkpoint, config_path = checkpoint_and_config(selector, outer, inner)
            add_input(rows, f"task/outer{outer}/inner{inner}/checkpoint", checkpoint)
            add_input(rows, f"task/outer{outer}/inner{inner}/config", config_path)
            task_rows.append(
                {
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "config": str(config_path.resolve()),
                    "config_sha256": file_sha256(config_path),
                }
            )
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
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "OOF_only_B0_validity",
        "diffact_root": str(diffact),
        "tasks": task_rows,
        "sampler": {
            "three_regions": ["free_core", "halo_buffer", "clamped_exterior"],
            "halo_widths": list(HALO_WIDTHS),
            "restart_times": list(RESTART_TIMES),
            "pure_noise_start": 999,
            "sampling_timesteps": 25,
            "context_noise": "one_fixed_tensor_shared_across_reverse_steps",
            "postprocess": {"type": "median", "value": 15},
            "non_core_restore_after_postprocess": True,
            "max_samples_per_setting": MAX_SAMPLES,
            "early_stop_min_samples": EARLY_STOP_MIN_SAMPLES,
            "early_stop_patience_no_new_collapsed_trace": EARLY_STOP_PATIENCE,
            "candidate": "cluster_medoid_normalized_segmental_edit_threshold_0.25",
            "per_frame_voting_allowed": False,
        },
        "b0_checks": ["postprocess_exterior_invariance", "empty_mask_identity", "seeded_replay"],
        "b1_kill_bars": B1_KILL_BARS,
        "b0_sampling_allowed": bool(args.authorize_b0),
        "b1_oracle_allowed": bool(args.authorize_b1),
        "v1_candidate_join_allowed": False,
        "v1_study_dir": str(args.v1_study_dir.resolve()),
        "outer_test_open_allowed": False,
        "v3_outer_evaluation_allowed": False,
        "fable_approval_digest": args.fable_approval_digest,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "review_state": (
            "V2_B0_REVIEW_ONLY"
            if not args.authorize_b0
            else "V2_B0_B1_APPROVED_READY"
            if args.authorize_b1
            else "V2_B0_APPROVED_READY"
        ),
        "input_manifest_digest": manifest["manifest_digest"],
        "source_provenance": provenance,
        "outer_test_opened": False,
        "gpu_launched": False,
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study_dir / "study_config.json", config)
    atomic_write_json(study_dir / "input_manifest.json", manifest)
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    (study_dir / "DO_NOT_EDIT.txt").write_text(
        "Immutable V2 B0 corrected-sampler validity study.\n"
        "Review-only unless a Fable approval digest is embedded.\n"
        "B1 is enabled only when study_config carries the reviewed approval; candidate joining and outer-test access stay disabled.\n"
    )
    for outer in OUTER_FOLDS:
        for inner in INNER_FOLDS:
            device = (outer + inner - 2) % 4
            script = study_dir / f"run_outer{outer}_inner{inner}.sh"
            script.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f"/usr/bin/python {Path(__file__).resolve().parent / 'run_v2_b0_validity.py'} "
                f"--study-dir {study_dir} --outer-fold {outer} --inner-fold {inner} "
                f"--device {device} 2>&1 | tee {study_dir / 'logs' / f'outer{outer}_inner{inner}.log'}\n"
            )
            script.chmod(0o755)
    task_keys = [(outer, inner) for outer in OUTER_FOLDS for inner in INNER_FOLDS]
    queue_assignments = {
        queue: task_keys[queue::3]
        for queue in range(3)
    }
    for queue_index, assignments in queue_assignments.items():
        queue = study_dir / f"run_queue{queue_index}_on_device.sh"
        commands = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "if [[ $# -ne 1 ]]; then echo 'usage: run_queue_on_device DEVICE'; exit 2; fi",
            "device=$1",
            "mkdir -p logs status",
        ]
        for outer, inner in assignments:
            commands.append(
                f"/usr/bin/python {Path(__file__).resolve().parent / 'run_v2_b0_validity.py'} "
                f"--study-dir {study_dir} --outer-fold {outer} --inner-fold {inner} "
                f"--device \"$device\" 2>&1 | tee -a {study_dir / 'logs' / f'queue{queue_index}.log'}"
            )
        commands.extend(
            [
                "while true; do",
                "  if ls status/*_b0_failed.json >/dev/null 2>&1; then echo 'B0 failed'; exit 1; fi",
                "  count=$(find status -maxdepth 1 -name '*_b0.json' | wc -l)",
                "  if [[ \"$count\" -eq 12 ]]; then break; fi",
                "  sleep 30",
                "done",
            ]
        )
        for outer, inner in assignments:
            commands.append(
                f"/usr/bin/python {Path(__file__).resolve().parent / 'run_v2_b1_oracle.py'} "
                f"--study-dir {study_dir} --outer-fold {outer} --inner-fold {inner} "
                f"--device \"$device\" 2>&1 | tee -a {study_dir / 'logs' / f'queue{queue_index}.log'}"
            )
        queue.write_text("\n".join(commands) + "\n")
        queue.chmod(0o755)
        waiter = study_dir / f"wait_queue{queue_index}.sh"
        waiter.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs status\n"
            f"v1_status={args.v1_study_dir.resolve() / 'status' / 'chosen_gpu.txt'}\n"
            "while [[ ! -f \"$v1_status\" ]]; do sleep 30; done\n"
            "v1_gpu=$(cat \"$v1_status\")\n"
            "while true; do\n"
            "  for device in 0 1 2 3; do\n"
            "    if [[ \"$device\" -eq \"$v1_gpu\" ]]; then continue; fi\n"
            "    first=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
            "    if [[ \"$first\" -eq 0 ]]; then\n"
            "      sleep 30\n"
            "      second=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i \"$device\" | sed '/^$/d' | wc -l)\n"
            "      if [[ \"$second\" -eq 0 ]] && mkdir \"status/gpu${device}.claim\" 2>/dev/null; then\n"
            f"        printf '%s\\n' \"$device\" > status/queue{queue_index}_gpu.txt\n"
            f"        exec ./run_queue{queue_index}_on_device.sh \"$device\"\n"
            "      fi\n"
            "    fi\n"
            "  done\n"
            "  sleep 30\n"
            "done\n"
        )
        waiter.chmod(0o755)
    aggregate = study_dir / "finalize_b1.sh"
    aggregate.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {Path(__file__).resolve().parent / 'finalize_v2_b1.py'} "
        f"--study-dir {study_dir} 2>&1 | tee {study_dir / 'logs' / 'finalize_b1.log'}\n"
    )
    aggregate.chmod(0o755)
    final_waiter = study_dir / "wait_and_finalize_b1.sh"
    final_waiter.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "while true; do\n"
        "  if ls status/*_b1_failed.json >/dev/null 2>&1; then echo 'B1 failed'; exit 1; fi\n"
        "  count=$(find status -maxdepth 1 -name '*_b1.json' | wc -l)\n"
        "  if [[ \"$count\" -eq 12 ]]; then exec ./finalize_b1.sh; fi\n"
        "  sleep 60\n"
        "done\n"
    )
    final_waiter.chmod(0o755)
    launcher = study_dir / "launch_tmux.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs status\n"
        + "\n".join(
            f"tmux new-session -d -s iav_v2_q{queue} 'cd {study_dir} && ./wait_queue{queue}.sh'"
            for queue in range(3)
        )
        + "\n"
        + f"tmux new-session -d -s iav_v2_finalize 'cd {study_dir} && ./wait_and_finalize_b1.sh'\n"
    )
    launcher.chmod(0o755)
    print(study_dir)
    print(f"spec_sha256={metadata['spec_sha256']}")
    print(f"source_digest={provenance['source_digest']}")
    print(f"input_manifest_digest={manifest['manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
