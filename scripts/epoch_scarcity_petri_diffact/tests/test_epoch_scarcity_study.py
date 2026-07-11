from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import epoch_scarcity_aggregation as aggregation  # noqa: E402
import epoch_scarcity_common as common  # noqa: E402
import epoch_scarcity_status as status  # noqa: E402
import export_epoch_checkpoint as exporter  # noqa: E402
import generate_epoch_scarcity_study as generator  # noqa: E402
import import_d100_trajectory as importer  # noqa: E402
import run_epoch_petri_bounds as bounds  # noqa: E402
import run_epoch_scarcity_task as task_runner  # noqa: E402
from src import incremental_softmax_recovery as recovery  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CheckpointProtocolTests(unittest.TestCase):
    def test_gtea_checkpoint_grid_and_step_counts_are_exact(self) -> None:
        records = common.checkpoint_records("gtea", n_train_cases=21, batch_size=3)
        self.assertEqual([record.epoch_index for record in records], [200, 1000, 2000, 5000, 10000])
        self.assertEqual(records[0].completed_epochs, 201)
        self.assertEqual(records[0].training_item_presentations, 201 * 21)
        self.assertEqual(records[0].trainer_step, 1 + 201 * 21)
        self.assertEqual(records[0].optimizer_updates, (201 * 21) // 3)
        self.assertEqual(records[-1].schedule_fraction_pct, 100.0)

    def test_condition_to_discovery_source_binding_is_exact(self) -> None:
        common.validate_condition_source(common.PRIMARY_CONDITION, common.DISCOVERY_FULL_TRAIN)
        common.validate_condition_source(common.ORACLE_CONDITION, common.DISCOVERY_OFFICIAL_TEST)
        with self.assertRaisesRegex(ValueError, "requires petri_discovery_source"):
            common.validate_condition_source(
                common.PRIMARY_CONDITION,
                common.DISCOVERY_OFFICIAL_TEST,
            )
        with self.assertRaisesRegex(ValueError, "requires petri_discovery_source"):
            common.validate_condition_source(common.ORACLE_CONDITION, common.DISCOVERY_FULL_TRAIN)

    def test_only_pre_specified_score_streams_are_retained(self) -> None:
        frame = pd.DataFrame(
            {
                "method": [
                    "raw_argmax",
                    "canonical_softmax_argmax",
                    "official_diffact",
                    "petri_conformance",
                ],
                "metric_value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        retained = bounds.retain_approved_score_methods(frame)
        self.assertEqual(set(retained["method"]), bounds.APPROVED_SCORE_METHODS)
        with self.assertRaisesRegex(ValueError, "missing"):
            bounds.retain_approved_score_methods(frame[frame["method"] != "raw_argmax"])

    def test_recovery_kwargs_are_derived_from_canonical_decoder(self) -> None:
        config = bounds.recovery_config("gtea", seed=0, workers=7, out_dir=Path("/tmp/out"))
        bounds.validate_canonical_recovery_config(config)
        for key in bounds.CANONICAL_RECOVERY_KEYS:
            self.assertEqual(config[key], common.CANONICAL_DECODER[key])
        config["candidate_top_k"] = 2
        with self.assertRaisesRegex(ValueError, "CANONICAL_DECODER"):
            bounds.validate_canonical_recovery_config(config)


class GeneratorTests(unittest.TestCase):
    def test_gtea_seed0_matrix_has_32_ordered_dependency_aware_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_tasks = []
            for fold in range(1, 5):
                source_tasks.append(
                    {
                        "task_id": f"source_fold{fold}_d100",
                        "dataset": "gtea",
                        "official_fold": fold,
                        "subset_seed": 0,
                        "diffact_fraction": 100,
                        "condition": "honest_low_data_petri",
                        "state_path": str(source_dir / "state" / f"fold{fold}.json"),
                        "experiment_dir": str(source_dir / "experiments" / f"fold_{fold}"),
                    }
                )
            write_json(
                source_dir / "study_metadata.json",
                {
                    "immutable": True,
                    "spec_sha256": "source-spec",
                    "tasks": source_tasks,
                },
            )
            args = argparse.Namespace(
                study_dir=root / "epoch",
                study_id="epoch_test",
                source_study_dir=source_dir,
                data_root=root / "data",
                diffact_root=root / "diffact",
                seed=0,
                gpus=[0, 1, 2, 3],
                petri_workers=7,
            )
            with mock.patch.object(generator, "source_provenance", return_value={}):
                spec = generator.build_spec(args)

        tasks = spec["tasks"]
        self.assertEqual(len(tasks), 32)
        self.assertEqual(
            {gpu: sum(int(task["gpu"]) == gpu for task in tasks) for gpu in range(4)},
            {0: 8, 1: 8, 2: 8, 3: 8},
        )
        self.assertEqual(
            {task["condition"] for task in tasks if task["condition"] is not None},
            {common.PRIMARY_CONDITION, common.ORACLE_CONDITION},
        )
        for fold in range(1, 5):
            fold_tasks = [task for task in tasks if int(task["official_fold"]) == fold]
            importer = next(task for task in fold_tasks if task["task_type"] == "import_trajectory")
            exports = [task for task in fold_tasks if task["task_type"] == "export_checkpoint"]
            decodes = [task for task in fold_tasks if task["task_type"] == "decode_checkpoint_grid"]
            self.assertEqual(len(importer["external_dependencies"]), 1)
            self.assertEqual(len(exports), 5)
            self.assertEqual(len(decodes), 2)
            self.assertTrue(all(task["depends_on_task_ids"] == [importer["task_id"]] for task in exports))
            export_ids = [task["task_id"] for task in exports]
            self.assertTrue(all(task["depends_on_task_ids"] == export_ids for task in decodes))
            self.assertEqual({task["queue"] for task in fold_tasks}, {importer["queue"]})


class WrapperTests(unittest.TestCase):
    def test_internal_and_external_dependencies_require_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            internal_state = root / "internal.json"
            external_state = root / "external.json"
            producer = {"task_id": "producer", "state_path": str(internal_state)}
            consumer = {
                "task_id": "consumer",
                "depends_on_task_ids": ["producer"],
                "external_dependencies": [
                    {"dependency_id": "outside", "state_path": str(external_state)}
                ],
            }
            metadata = {"tasks": [producer, consumer]}
            write_json(internal_state, {"status": "complete", "returncode": 0})
            write_json(external_state, {"status": "running", "returncode": None})
            with self.assertRaisesRegex(RuntimeError, "outside"):
                task_runner.validate_dependencies(metadata, consumer)
            write_json(external_state, {"status": "complete", "returncode": 0})
            task_runner.validate_dependencies(metadata, consumer)

    def test_source_digest_mismatch_is_a_hard_failure(self) -> None:
        metadata = {
            "diffact_root": "/tmp/diffact",
            "source_provenance": {
                "source_digest": "old",
                "file_sha256": {"a.py": "before"},
            },
        }
        with mock.patch.object(
            task_runner,
            "source_provenance",
            return_value={"source_digest": "new", "file_sha256": {"a.py": "after"}},
        ):
            with self.assertRaisesRegex(RuntimeError, "changed_files=\['a.py'\]"):
                task_runner.validate_source_provenance(metadata)


class RecoveryContextTests(unittest.TestCase):
    def test_reusable_context_is_built_once_and_validated(self) -> None:
        class FakeModel:
            places = {"p"}
            transitions = {"t"}

            def enable_caching(self, enabled: bool, max_size: int) -> None:
                self.cache_settings = (enabled, max_size)

        train = pd.DataFrame(
            {
                "case:concept:name": ["0", "0", "1", "1"],
                "concept:name": ["0", "1", "0", "1"],
            }
        )
        with (
            mock.patch.object(recovery, "discover_petri_net", return_value=FakeModel()) as discover,
            mock.patch.object(
                recovery,
                "build_probability_dict",
                side_effect=[{("0",): {"1": 1.0}}, {("0",): {"1": 1.0}}],
            ) as probabilities,
        ):
            context = recovery.build_recovery_context(
                train,
                max_hist_len=3,
                enabled_cache_size=100,
                compute_marking_transition_map=False,
                precompute_marking_transition_map=False,
                conformance_switch_penalty_weight=1.0,
                conditioning_alpha=0.5,
                save_model=False,
            )
        self.assertEqual(discover.call_count, 1)
        self.assertEqual(probabilities.call_count, 2)
        recovery._validate_recovery_context(
            context,
            train,
            max_hist_len=3,
            enabled_cache_size=100,
            compute_marking_transition_map=False,
            precompute_marking_transition_map=False,
            conformance_switch_penalty_weight=1.0,
            conditioning_alpha=0.5,
        )
        with self.assertRaisesRegex(ValueError, "enabled_cache_size"):
            recovery._validate_recovery_context(
                context,
                train,
                max_hist_len=3,
                enabled_cache_size=101,
                compute_marking_transition_map=False,
                precompute_marking_transition_map=False,
                conformance_switch_penalty_weight=1.0,
                conditioning_alpha=0.5,
            )

    def test_context_reuse_preserves_recovery_predictions(self) -> None:
        frame = pd.DataFrame(
            {
                "case:concept:name": ["0"] * 4 + ["1"] * 4 + ["2"] * 4,
                "concept:name": ["0", "0", "1", "1"] * 3,
            }
        )
        matrix = np.asarray(
            [[0.95, 0.90, 0.10, 0.05], [0.05, 0.10, 0.90, 0.95]],
            dtype=np.float32,
        )
        softmax = [matrix.copy() for _ in range(3)]
        kwargs = {
            "train_cases": ["0", "1"],
            "test_cases": ["2"],
            "n_train_traces": 2,
            "n_test_traces": 1,
            "n_indices": 10**9,
            "sequential_sampling": False,
            "independent_sampling": True,
            "save_model": False,
            "chunk_size": 4,
            "conformance_switch_penalty_weight": 0.0,
            "conditioning_alpha": None,
            "candidate_top_k": 2,
            "restrict_log_moves": True,
            "restrict_model_moves_to_tau": True,
            "max_consecutive_tau_moves": 8,
            "compute_marking_transition_map": False,
            "dataset_parallelization": False,
            "verbose": False,
        }
        legacy, legacy_accuracy, _ = recovery.incremental_softmax_recovery(
            frame,
            softmax,
            **kwargs,
        )
        context = recovery.build_recovery_context(
            frame[frame["case:concept:name"].isin(["0", "1"])],
            max_hist_len=3,
            enabled_cache_size=10000,
            compute_marking_transition_map=False,
            precompute_marking_transition_map=False,
            conformance_switch_penalty_weight=0.0,
            conditioning_alpha=None,
            save_model=False,
        )
        reused, reused_accuracy, _ = recovery.incremental_softmax_recovery(
            frame,
            softmax,
            recovery_context=context,
            **kwargs,
        )
        columns = [
            "case:concept:name",
            "step",
            "sktr_activity",
            "argmax_activity",
            "ground_truth",
            "sktr_move_cost",
        ]
        pd.testing.assert_frame_equal(
            legacy[columns].reset_index(drop=True),
            reused[columns].reset_index(drop=True),
        )
        self.assertEqual(legacy_accuracy, reused_accuracy)

    def test_context_cache_is_checkpoint_order_independent(self) -> None:
        frame = pd.DataFrame(
            {
                "case:concept:name": ["0"] * 4 + ["1"] * 4 + ["2"] * 4,
                "concept:name": ["0", "0", "1", "1"] * 3,
            }
        )
        matrix_a = np.asarray(
            [[0.95, 0.90, 0.10, 0.05], [0.05, 0.10, 0.90, 0.95]],
            dtype=np.float32,
        )
        matrix_b = np.asarray(
            [[0.90, 0.20, 0.80, 0.10], [0.10, 0.80, 0.20, 0.90]],
            dtype=np.float32,
        )
        kwargs = {
            "train_cases": ["0", "1"],
            "test_cases": ["2"],
            "n_train_traces": 2,
            "n_test_traces": 1,
            "n_indices": 10**9,
            "sequential_sampling": False,
            "independent_sampling": True,
            "save_model": False,
            "chunk_size": 4,
            "conformance_switch_penalty_weight": 0.0,
            "conditioning_alpha": None,
            "candidate_top_k": 2,
            "restrict_log_moves": True,
            "restrict_model_moves_to_tau": True,
            "max_consecutive_tau_moves": 8,
            "compute_marking_transition_map": False,
            "dataset_parallelization": False,
            "verbose": False,
        }

        def context() -> recovery.RecoveryContext:
            return recovery.build_recovery_context(
                frame[frame["case:concept:name"].isin(["0", "1"])],
                max_hist_len=3,
                enabled_cache_size=10000,
                compute_marking_transition_map=False,
                precompute_marking_transition_map=False,
                conformance_switch_penalty_weight=0.0,
                conditioning_alpha=None,
                save_model=False,
            )

        def decode(matrix: np.ndarray, value: recovery.RecoveryContext) -> pd.DataFrame:
            result, _, _ = recovery.incremental_softmax_recovery(
                frame,
                [matrix.copy() for _ in range(3)],
                recovery_context=value,
                **kwargs,
            )
            return result[["sktr_activity", "argmax_activity", "sktr_move_cost"]].reset_index(
                drop=True
            )

        context_ab = context()
        a_after_none = decode(matrix_a, context_ab)
        b_after_a = decode(matrix_b, context_ab)
        context_ba = context()
        b_after_none = decode(matrix_b, context_ba)
        a_after_b = decode(matrix_a, context_ba)
        pd.testing.assert_frame_equal(a_after_none, a_after_b)
        pd.testing.assert_frame_equal(b_after_a, b_after_none)


class ImportIntegrityTests(unittest.TestCase):
    def test_completed_import_hashes_and_provenance_detect_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            source = root / "source"
            source_config = source / "config.json"
            source_complete = source / "train_complete.json"
            write_json(source_config, {"batch_size": 4})
            write_json(source_complete, {"completed": True, "returncode": 0})
            write_json(
                experiment / "experiment_metadata.json",
                {
                    "dataset": "gtea",
                    "fold": 1,
                    "trajectory_seed": 0,
                    "source_d100_task_id": "source_d100",
                },
            )
            records = []
            for checkpoint in common.checkpoint_records(
                "gtea", n_train_cases=21, batch_size=4
            ):
                epoch = checkpoint.epoch_index
                source_model = source / f"epoch-{epoch}.model"
                source_model.parent.mkdir(parents=True, exist_ok=True)
                source_model.write_bytes(f"checkpoint-{epoch}".encode())
                copied = common.checkpoint_model_path(experiment, 0, epoch)
                copied.parent.mkdir(parents=True, exist_ok=True)
                copied.write_bytes(source_model.read_bytes())
                row = {
                    **common.record_to_dict(checkpoint),
                    "dataset": "gtea",
                    "fold": 1,
                    "trajectory_seed": 0,
                    "source_checkpoint": str(source_model),
                    "source_checkpoint_sha256": common.file_sha256(source_model),
                    "copied_checkpoint": str(copied),
                    "source_config": str(source_config),
                    "source_config_sha256": common.file_sha256(source_config),
                    "n_train_cases": 21,
                    "batch_size": 4,
                }
                records.append(row)
                write_json(common.checkpoint_metadata_path(experiment, 0, epoch), row)
            write_json(
                experiment / "trajectory" / "seed_0" / "import_complete.json",
                {
                    "completed": True,
                    "dataset": "gtea",
                    "fold": 1,
                    "trajectory_seed": 0,
                    "source_task_id": "source_d100",
                    "source_train_complete": str(source_complete),
                    "source_train_complete_sha256": common.file_sha256(source_complete),
                    "checkpoint_records": records,
                },
            )
            importer.validate_completed_import(experiment)
            common.checkpoint_model_path(experiment, 0, 1000).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "copied_sha256"):
                importer.validate_completed_import(experiment)


class ExportIntegrityTests(unittest.TestCase):
    def test_completed_export_hashes_detect_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            study_dir = Path(temporary) / "study"
            experiment = study_dir / "experiments" / "gtea" / "fold_1"
            seed = 0
            epoch = 200
            (experiment / "manifests").mkdir(parents=True)
            (experiment / "manifests" / "test_cases.txt").write_text(
                "case_a\n",
                encoding="utf-8",
            )
            checkpoint = common.checkpoint_model_path(experiment, seed, epoch)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha = common.file_sha256(checkpoint)
            write_json(
                common.checkpoint_metadata_path(experiment, seed, epoch),
                {"source_checkpoint_sha256": checkpoint_sha},
            )
            output = common.softmax_dir(experiment, seed, epoch)
            output.mkdir(parents=True)
            (output / "video_index_map.txt").write_text("0\tcase_a\n", encoding="utf-8")
            (output / "mapping.txt").write_text("0 action\n", encoding="utf-8")
            (output / "ground_truth.csv").write_text("case,step,label\n", encoding="utf-8")
            np.save(output / "0.npy", np.asarray([[1.0, 1.0]], dtype=np.float32))
            np.save(output / "0_raw.npy", np.asarray([[1.0, 1.0]], dtype=np.float32))
            np.save(output / "0_pred.npy", np.asarray([0, 0], dtype=np.int32))
            artifact_sha = exporter.bundle_sha256(output, 1)
            write_json(
                output / "export_complete.json",
                {
                    "completed": True,
                    "returncode": 0,
                    "trajectory_seed": seed,
                    "checkpoint_epoch_index": epoch,
                    "checkpoint_sha256": checkpoint_sha,
                    "n_test_cases": 1,
                    "artifact_sha256": artifact_sha,
                },
            )
            exporter.validate_completed_export(
                experiment,
                seed=seed,
                checkpoint_epoch=epoch,
            )
            state_path = study_dir / "state" / "export.json"
            write_json(state_path, {"status": "complete", "returncode": 0})
            task = {
                "task_id": "export",
                "task_type": "export_checkpoint",
                "dataset": "gtea",
                "official_fold": 1,
                "trajectory_seed": seed,
                "checkpoint_epoch_index": epoch,
                "condition": None,
                "gpu": 0,
                "queue": "gpu0",
                "state_path": str(state_path),
                "log_path": str(study_dir / "logs" / "export.log"),
            }
            self.assertEqual(status.task_row(task, study_dir)["effective_status"], "complete")
            np.save(output / "0.npy", np.asarray([[0.5, 0.5]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "artifact_sha256"):
                exporter.validate_completed_export(
                    experiment,
                    seed=seed,
                    checkpoint_epoch=epoch,
                )
            row = status.task_row(task, study_dir)
            self.assertEqual(row["effective_status"], "inconsistent")
            self.assertFalse(row["artifact_complete"])
            self.assertIn("artifact_sha256", row["artifact_detail"])


class DecodeIntegrityTests(unittest.TestCase):
    def test_completed_decode_grid_revalidates_every_epoch_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = (Path(temporary) / "experiment").resolve()
            write_json(
                experiment / "experiment_metadata.json",
                {"dataset": "gtea", "fold": 1, "trajectory_seed": 0},
            )
            manifests = experiment / "manifests"
            manifests.mkdir(parents=True)
            (manifests / "train_pool_cases.txt").write_text("train_a\n", encoding="utf-8")
            (manifests / "test_cases.txt").write_text("test_a\n", encoding="utf-8")
            for epoch in common.dataset_spec("gtea")["checkpoint_epochs"]:
                out_dir = bounds.output_dir_for(
                    experiment,
                    common.PRIMARY_CONDITION,
                    0,
                    epoch,
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                for name in ("results.csv", "runtime.csv", "per_case_runtime.csv"):
                    (out_dir / name).write_text("placeholder\n", encoding="utf-8")
                write_json(out_dir / "accuracy_dict.json", {})
                rows = [
                    {
                        "method": method,
                        "checkpoint_epoch_index": epoch,
                        "condition": common.PRIMARY_CONDITION,
                        "petri_discovery_source": common.DISCOVERY_FULL_TRAIN,
                        "oracle_upper_bound": False,
                        "is_oracle_condition": False,
                    }
                    for method in sorted(bounds.APPROVED_SCORE_METHODS)
                ]
                pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
                pd.DataFrame(rows).to_csv(out_dir / "per_case_metrics.csv", index=False)
                config = bounds.recovery_config("gtea", 0, 7, out_dir)
                write_json(
                    out_dir / "postprocess_config.json",
                    {
                        "condition": common.PRIMARY_CONDITION,
                        "petri_discovery_source": common.DISCOVERY_FULL_TRAIN,
                        "oracle_upper_bound": False,
                        "petri_discovery_uses_test_gt": False,
                        "is_oracle_condition": False,
                        "checkpoint": {"epoch_index": epoch},
                        "canonical_decoder": common.CANONICAL_DECODER,
                        "recovery_config": bounds.serializable_config(config),
                        "context_reused_across_checkpoint_grid": True,
                    },
                )
            marker = (
                experiment
                / "petri"
                / common.PRIMARY_CONDITION
                / "seed_0"
                / "decode_grid_complete.json"
            )
            write_json(
                marker,
                {
                    "completed": True,
                    "condition": common.PRIMARY_CONDITION,
                    "petri_discovery_source": common.DISCOVERY_FULL_TRAIN,
                    "is_oracle_condition": False,
                    "checkpoint_epoch_indices": list(
                        common.dataset_spec("gtea")["checkpoint_epochs"]
                    ),
                    "petri_context_discovery_count_this_invocation": 1,
                    "petri_discovery_manifest": str(manifests / "train_pool_cases.txt"),
                    "petri_discovery_manifest_sha256": common.file_sha256(
                        manifests / "train_pool_cases.txt"
                    ),
                },
            )
            with mock.patch.object(bounds, "validate_completed_export", return_value={}):
                bounds.validate_completed_decode_grid(
                    experiment,
                    seed=0,
                    condition=common.PRIMARY_CONDITION,
                    discovery_source=common.DISCOVERY_FULL_TRAIN,
                )
                tampered_dir = bounds.output_dir_for(
                    experiment, common.PRIMARY_CONDITION, 0, 200
                )
                tampered = json.loads((tampered_dir / "postprocess_config.json").read_text())
                tampered["recovery_config"]["candidate_top_k"] = 2
                write_json(tampered_dir / "postprocess_config.json", tampered)
                with self.assertRaisesRegex(ValueError, "recovery_config"):
                    bounds.validate_completed_decode_grid(
                        experiment,
                        seed=0,
                        condition=common.PRIMARY_CONDITION,
                        discovery_source=common.DISCOVERY_FULL_TRAIN,
                    )


class AggregationTests(unittest.TestCase):
    def test_aggregation_refuses_incomplete_task_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "task.json"
            write_json(state_path, {"status": "running", "returncode": None})
            with self.assertRaisesRegex(RuntimeError, "must complete successfully"):
                aggregation.validate_completed_task_states(
                    {"tasks": [{"task_id": "task", "state_path": str(state_path)}]}
                )

    def test_primary_and_oracle_epoch_curves_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            study_dir = Path(temporary) / "study"
            tasks = []
            experiments = []
            for fold in (1, 2):
                experiments.append(
                    {
                        "dataset": "gtea",
                        "official_fold": fold,
                        "trajectory_seed": 0,
                    }
                )
                state_path = study_dir / "state" / f"fold_{fold}.json"
                write_json(state_path, {"status": "complete", "returncode": 0})
                tasks.append({"task_id": f"fold_{fold}", "state_path": str(state_path)})
            write_json(
                study_dir / "study_metadata.json",
                {
                    "study_id": "epoch_test",
                    "study_type": common.STUDY_TYPE,
                    "checkpoint_grid": {"gtea": [200, 1000]},
                    "experiments": experiments,
                    "conditions": {
                        common.PRIMARY_CONDITION: {
                            "petri_discovery_source": common.DISCOVERY_FULL_TRAIN,
                            "is_oracle_condition": False,
                        },
                        common.ORACLE_CONDITION: {
                            "petri_discovery_source": common.DISCOVERY_OFFICIAL_TEST,
                            "is_oracle_condition": True,
                        },
                    },
                    "tasks": tasks,
                },
            )
            for fold in (1, 2):
                for epoch in (200, 1000):
                    for condition, source, oracle, bonus in (
                        (common.PRIMARY_CONDITION, common.DISCOVERY_FULL_TRAIN, False, 4.0),
                        (common.ORACLE_CONDITION, common.DISCOVERY_OFFICIAL_TEST, True, 8.0),
                    ):
                        rows = []
                        for case in ("case_a", "case_b"):
                            base = 60.0 + fold + epoch / 10000.0
                            for metric in aggregation.CORE_METRICS:
                                shared = {
                                    "dataset": "gtea",
                                    "official_fold": fold,
                                    "trajectory_seed": 0,
                                    "checkpoint_epoch_index": epoch,
                                    "completed_epochs": epoch + 1,
                                    "schedule_fraction_pct": epoch / 100.0,
                                    "training_item_presentations": (epoch + 1) * 21,
                                    "trainer_step": 1 + (epoch + 1) * 21,
                                    "optimizer_updates": ((epoch + 1) * 21) // 3,
                                    "condition": condition,
                                    "petri_discovery_source": source,
                                    "is_oracle_condition": oracle,
                                    "split": "test",
                                    "original_case_id": case,
                                    "n_frames": 100,
                                    "metric_name": metric,
                                }
                                rows.extend(
                                    [
                                        {**shared, "method": "raw_argmax", "metric_value": base - 2.0},
                                        {**shared, "method": "official_diffact", "metric_value": base},
                                        {**shared, "method": "petri_conformance", "metric_value": base + bonus},
                                    ]
                                )
                        output = (
                            study_dir
                            / "experiments"
                            / "gtea"
                            / f"fold_{fold}"
                            / "petri"
                            / condition
                            / f"epoch_{epoch}"
                            / "per_case_metrics.csv"
                        )
                        output.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).to_csv(output, index=False)
                        global_rows = []
                        global_base = 55.0 + fold + epoch / 10000.0
                        for metric in aggregation.CORE_METRICS:
                            global_shared = {
                                "dataset": "gtea",
                                "official_fold": fold,
                                "trajectory_seed": 0,
                                "checkpoint_epoch_index": epoch,
                                "completed_epochs": epoch + 1,
                                "schedule_fraction_pct": epoch / 100.0,
                                "training_item_presentations": (epoch + 1) * 21,
                                "trainer_step": 1 + (epoch + 1) * 21,
                                "optimizer_updates": ((epoch + 1) * 21) // 3,
                                "condition": condition,
                                "petri_discovery_source": source,
                                "is_oracle_condition": oracle,
                                "split": "test",
                                "n_cases": 2,
                                "metric_name": metric,
                            }
                            global_rows.extend(
                                [
                                    {
                                        **global_shared,
                                        "method": "raw_argmax",
                                        "metric_value": global_base - 2.0,
                                    },
                                    {
                                        **global_shared,
                                        "method": "official_diffact",
                                        "metric_value": global_base,
                                    },
                                    {
                                        **global_shared,
                                        "method": "petri_conformance",
                                        "metric_value": global_base + bonus,
                                    },
                                ]
                            )
                        pd.DataFrame(global_rows).to_csv(output.parent / "metrics.csv", index=False)

            paths = aggregation.aggregate_study(study_dir, n_bootstrap=20)
            deltas = pd.read_csv(paths["paired_case_deltas"])
            primary = pd.read_csv(paths["primary_curve"])
            oracle = pd.read_csv(paths["oracle_curve"])
            global_primary = pd.read_csv(paths["fold_global_primary_curve"])
            global_oracle = pd.read_csv(paths["fold_global_oracle_curve"])
            global_deltas = pd.read_csv(paths["fold_global_deltas"])
            victim = next(
                study_dir.glob(
                    "experiments/gtea/fold_1/petri/official_full_train_petri/"
                    "epoch_200/per_case_metrics.csv"
                )
            )
            victim.unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete per-case checkpoint grid"):
                aggregation.aggregate_study(study_dir, n_bootstrap=0)

        self.assertEqual(set(primary["checkpoint_epoch_index"]), {200, 1000})
        self.assertEqual(set(primary["condition"]), {common.PRIMARY_CONDITION})
        self.assertTrue((primary["mean"] - 4.0).abs().lt(1e-10).all())
        self.assertEqual(set(oracle["condition"]), {common.ORACLE_CONDITION})
        self.assertTrue(oracle["is_oracle_condition"].astype(bool).all())
        self.assertEqual(set(global_primary["checkpoint_epoch_index"]), {200, 1000})
        self.assertEqual(set(global_primary["paired_unit"]), {"official_fold_global_tas"})
        self.assertTrue((global_primary["mean"] - 4.0).abs().lt(1e-10).all())
        self.assertEqual(set(global_oracle["condition"]), {common.ORACLE_CONDITION})
        self.assertEqual(int(global_deltas["is_secondary_fold_global_endpoint"].sum()), 4)
        self.assertFalse(global_deltas["is_primary_endpoint"].astype(bool).any())
        self.assertFalse(
            deltas.loc[deltas["is_oracle_condition"].astype(bool), "is_primary_endpoint"]
            .astype(bool)
            .any()
        )


if __name__ == "__main__":
    unittest.main()
