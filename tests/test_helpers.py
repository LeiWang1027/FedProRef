import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


PLAN_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PartitionReportTests(unittest.TestCase):
    def test_reports_missing_pairs_and_anchor_coverage(self):
        module = load_module("partition_report", PLAN_ROOT / "partition_report.py")
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        clients = np.array([
            np.array([0, 1, 3]),
            np.array([2, 4, 5, 6]),
            np.array([7, 8]),
        ], dtype=object)

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.npz"
            np.savez(cache, train_labels=labels, client_indices=clients)
            report = module.build_report(cache, "fixture", 42, 3, 2)

        self.assertEqual(report["total_client_class_pairs"], 9)
        self.assertEqual(report["zero_pairs"], 3)
        self.assertAlmostEqual(report["zero_pair_percentage"], 100.0 / 3.0)
        self.assertEqual(report["observed_weak_pairs"], 3)
        self.assertEqual(report["strong_pairs"], 3)
        self.assertEqual(report["missing_classes_per_client"], [1, 0, 2])
        self.assertEqual(report["client_total_samples"], [3, 4, 2])
        self.assertEqual(report["empty_clients"], 0)
        self.assertEqual(report["global_anchor_classes"], 3)
        self.assertEqual(report["classes_without_global_anchor"], [])
        self.assertEqual(report["augmentable_missing_pairs"], 3)
        self.assertEqual(len(report["partition_index_sha256"]), 64)


class AccuracySummaryTests(unittest.TestCase):
    def test_summarizes_training_seeds_with_sample_sd(self):
        module = load_module(
            "summarize_missing_class", PLAN_ROOT / "summarize_missing_class.py")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, accuracy in ((42, 70.0), (43, 71.0), (44, 72.0)):
                run_id = f"mcstress_cifar100_a001_ps42_ts{seed}_proto_aug"
                run_dir = root / run_id
                run_dir.mkdir()
                (run_dir / ".done").touch()
                (run_dir / "console.log").write_text(
                    f"Best: Round 88 | Acc={accuracy:.2f}% | ECE=0.1 | NLL=1.0\n",
                    encoding="utf-8",
                )

            rows = module.collect_runs(root)
            partitions = module.partition_summaries(rows)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["dataset"], "cifar100")
        self.assertEqual(partitions[0]["n_training_seeds"], 3)
        self.assertAlmostEqual(partitions[0]["mean_best_acc"], 71.0)
        self.assertAlmostEqual(partitions[0]["sample_sd_best_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
