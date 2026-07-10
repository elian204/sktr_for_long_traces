from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_low_data_study as study_generator  # noqa: E402
import run_low_data_study_task as study_task  # noqa: E402
import run_petri_postprocessing_low_data as postprocess  # noqa: E402
import study_aggregation  # noqa: E402


class ConditionDiscoveryInvariantTests(unittest.TestCase):
    def test_approved_condition_sources_are_accepted(self) -> None:
        approved = [
            (25, 25, postprocess.COUPLED_CONDITION, postprocess.PETRI_DISCOVERY_NESTED),
            (
                50,
                100,
                postprocess.DIFFACT_LOW_PETRI_FULL_CONDITION,
                postprocess.PETRI_DISCOVERY_FULL_TRAIN,
            ),
            (
                100,
                25,
                postprocess.DIFFACT_FULL_PETRI_LOW_CONDITION,
                postprocess.PETRI_DISCOVERY_NESTED,
            ),
            (
                100,
                None,
                postprocess.ORACLE_TEST_FOLD_CONDITION,
                postprocess.PETRI_DISCOVERY_OFFICIAL_TEST,
            ),
        ]
        for diffact_fraction, petri_fraction, condition, source in approved:
            with self.subTest(condition=condition, source=source):
                postprocess.validate_fraction_condition(
                    diffact_fraction=diffact_fraction,
                    petri_fraction=petri_fraction,
                    condition=condition,
                    petri_discovery_source=source,
                )

    def test_mislabeled_discovery_sources_are_rejected(self) -> None:
        invalid = [
            (25, 25, postprocess.COUPLED_CONDITION, postprocess.PETRI_DISCOVERY_OFFICIAL_TEST),
            (
                100,
                25,
                postprocess.DIFFACT_FULL_PETRI_LOW_CONDITION,
                postprocess.PETRI_DISCOVERY_FULL_TRAIN,
            ),
            (
                25,
                100,
                postprocess.DIFFACT_LOW_PETRI_FULL_CONDITION,
                postprocess.PETRI_DISCOVERY_NESTED,
            ),
            (
                25,
                None,
                postprocess.ORACLE_TEST_FOLD_CONDITION,
                postprocess.PETRI_DISCOVERY_NESTED,
            ),
        ]
        for diffact_fraction, petri_fraction, condition, source in invalid:
            with self.subTest(condition=condition, source=source):
                with self.assertRaisesRegex(ValueError, "requires petri_discovery_source"):
                    postprocess.validate_fraction_condition(
                        diffact_fraction=diffact_fraction,
                        petri_fraction=petri_fraction,
                        condition=condition,
                        petri_discovery_source=source,
                    )

    def test_missing_softmax_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment_dir = root / "experiment"
            seed_manifest_dir = experiment_dir / "manifests" / "seed_0"
            seed_manifest_dir.mkdir(parents=True)
            (seed_manifest_dir / "train_cases_frac_25.txt").write_text(
                "train_case\n", encoding="utf-8"
            )
            (experiment_dir / "manifests" / "test_cases.txt").write_text(
                "test_case\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(FileNotFoundError, "Missing DiffAct softmax directory"):
                postprocess.run_one(
                    experiment_dir=experiment_dir,
                    data_root=root / "data",
                    dataset="gtea",
                    seed=0,
                    diffact_fraction=25,
                    petri_fraction=25,
                    split="test",
                    condition=postprocess.COUPLED_CONDITION,
                    protocol=postprocess.PRIMARY_PROTOCOL,
                    method=postprocess.PAPER_DIFFACT_SKTR_METHOD,
                    chunk_size=postprocess.PAPER_DIFFACT_SKTR_CHUNK_SIZE,
                    prob_threshold=1e-6,
                    model_move_cost=1.0,
                    conformance_switch_penalty_weight=1.0,
                    progress_log_interval_chunks=20,
                    petri_discovery_representation=(
                        postprocess.PAPER_DIFFACT_SKTR_DISCOVERY_REPRESENTATION
                    ),
                    petri_discovery_source=postprocess.PETRI_DISCOVERY_NESTED,
                    transition_illegal_penalty=2.0,
                    calibrated=False,
                    temp_bounds=(1.0, 10.0),
                    workers=postprocess.PAPER_DIFFACT_SKTR_WORKERS,
                    inner_parallel=True,
                    runtime_pilot_case_limit=None,
                    require_diffact_comparators=True,
                    force=False,
                )


class StudyDependencyTests(unittest.TestCase):
    def test_bounds_share_their_producer_queue_and_declare_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                study_dir=Path(temporary) / "study",
                study_id="test_gtea_seed0_bounds",
                data_root=Path(temporary) / "data",
                diffact_root=Path(temporary) / "diffact",
                seeds=[0],
                fractions=[25, 50, 75, 100],
                datasets=["gtea"],
                petri_workers=7,
                runtime_pilot=False,
                runtime_pilot_case_limit=1,
            )
            with mock.patch.object(study_generator, "source_provenance", return_value={}):
                spec = study_generator.build_spec(args)

        tasks = spec["tasks"]
        self.assertEqual(len(tasks), 48)
        self.assertEqual({task["gpu"] for task in tasks}, {0, 1, 2, 3})
        self.assertEqual(
            {gpu: sum(task["gpu"] == gpu for task in tasks) for gpu in range(4)},
            {0: 12, 1: 12, 2: 12, 3: 12},
        )
        queue_positions = {
            queue: {
                task["task_id"]: index
                for index, task in enumerate(
                    candidate for candidate in tasks if candidate["queue"] == queue
                )
            }
            for queue in {task["queue"] for task in tasks}
        }
        honest = {
            (
                task["dataset"],
                task["official_fold"],
                task["subset_seed"],
                task["diffact_fraction"],
            ): task
            for task in tasks
            if task["condition"] == study_generator.COUPLED_CONDITION
        }
        for task in tasks:
            if task["condition"] == study_generator.COUPLED_CONDITION:
                self.assertEqual(task["depends_on_task_ids"], [])
                continue
            producer = honest[
                (
                    task["dataset"],
                    task["official_fold"],
                    task["subset_seed"],
                    task["diffact_fraction"],
                )
            ]
            self.assertEqual(task["queue"], producer["queue"])
            self.assertEqual(task["gpu"], producer["gpu"])
            self.assertEqual(task["depends_on_task_ids"], [producer["task_id"]])
            positions = queue_positions[task["queue"]]
            self.assertLess(positions[producer["task_id"]], positions[task["task_id"]])

    def test_wrapper_requires_successful_dependency_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "producer.json"
            producer = {"task_id": "producer", "state_path": str(state_path)}
            consumer = {"task_id": "consumer", "depends_on_task_ids": ["producer"]}
            metadata = {"tasks": [producer, consumer]}

            state_path.write_text(
                json.dumps({"status": "running", "returncode": None}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "cannot start before dependency"):
                study_task.validate_completed_dependencies(metadata, consumer)

            state_path.write_text(
                json.dumps({"status": "complete", "returncode": 0}), encoding="utf-8"
            )
            study_task.validate_completed_dependencies(metadata, consumer)


class AggregationRobustnessTests(unittest.TestCase):
    def test_string_false_oracle_flag_does_not_blank_fraction(self) -> None:
        fraction = study_aggregation._petri_fraction(
            {"petri_fraction": 25, "is_oracle_condition": "False"},
            study_aggregation.HONEST_CONDITION,
            25,
        )
        self.assertEqual(fraction, 25)
        oracle_fraction = study_aggregation._petri_fraction(
            {"petri_fraction": 25, "is_oracle_condition": "True"},
            study_aggregation.HONEST_CONDITION,
            25,
        )
        self.assertTrue(pd.isna(oracle_fraction))

    def test_aggregation_writes_explicit_bound_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment_dir = root / "experiment"
            metrics_dir = experiment_dir / "petri"
            metrics_dir.mkdir(parents=True)
            (experiment_dir / "experiment_metadata.json").write_text(
                json.dumps({"dataset": "gtea", "fold": 1, "run_type": "final"}),
                encoding="utf-8",
            )
            rows = []
            conditions = [
                (
                    study_aggregation.HONEST_CONDITION,
                    25,
                    "nested_train_fraction",
                    False,
                    75.0,
                ),
                (
                    "structural_prior_full_train_petri",
                    100,
                    "full_train",
                    False,
                    78.0,
                ),
                (
                    study_aggregation.ORACLE_CONDITION,
                    None,
                    "official_test_fold",
                    True,
                    82.0,
                ),
            ]
            for condition, petri_fraction, source, is_oracle, sktr_value in conditions:
                common = {
                    "dataset": "gtea",
                    "official_fold": 1,
                    "seed": 0,
                    "diffact_fraction": 25,
                    "petri_fraction": petri_fraction,
                    "condition": condition,
                    "petri_discovery_source": source,
                    "is_oracle_condition": is_oracle,
                    "split": "test",
                    "metric_name": "f1@25",
                    "n_cases": 7,
                    "complete_evaluation_set": True,
                }
                rows.append({**common, "method": "petri_conformance", "metric_value": sktr_value})
                rows.append({**common, "method": "official_diffact", "metric_value": 70.0})
            pd.DataFrame(rows).to_csv(metrics_dir / "metrics.csv", index=False)

            paths = study_aggregation.aggregate_study(
                [experiment_dir], root / "aggregation", n_bootstrap=0
            )
            bound_rows = pd.read_csv(paths["bound_paired_deltas"])
            bound_curves = pd.read_csv(paths["bound_curves"])
            all_deltas = pd.read_csv(paths["paired_deltas"])

        self.assertEqual(
            set(bound_rows["condition"]),
            {"structural_prior_full_train_petri", study_aggregation.ORACLE_CONDITION},
        )
        self.assertEqual(set(bound_curves["condition"]), set(bound_rows["condition"]))
        self.assertNotIn(study_aggregation.HONEST_CONDITION, set(bound_curves["condition"]))
        self.assertEqual(int(all_deltas["is_primary_endpoint"].sum()), 1)
        oracle_primary = all_deltas.loc[
            all_deltas["is_oracle_condition"].astype(bool), "is_primary_endpoint"
        ]
        self.assertFalse(oracle_primary.astype(bool).any())


if __name__ == "__main__":
    unittest.main()
