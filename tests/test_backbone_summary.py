import json
import math
import tempfile
import unittest
from pathlib import Path

from summarize_backbone_rn50 import (
    aggregate_runs,
    paired_deltas,
    parse_completed_runs,
    write_outputs,
)


def fixture_rows():
    values = {
        0: {
            "fedavg": [60.0, 61.0, 62.0],
            "proto_aug": [66.0, 67.0, 68.0],
            "direct_anchor_aug": [67.0, 69.0, 71.0],
            "fedproref": [70.0, 72.0, 74.0],
        },
        1: {
            "fedavg": [80.0, 81.0, 82.0],
            "proto_aug": [83.0, 84.0, 85.0],
            "direct_anchor_aug": [84.0, 86.0, 88.0],
            "fedproref": [85.0, 84.0, 89.0],
        },
    }
    rows = []
    for floor, by_method in values.items():
        for method, accuracies in by_method.items():
            for seed, accuracy in zip((42, 43, 44), accuracies):
                rows.append({
                    "dataset": "cifar100",
                    "backbone": "RN50",
                    "pretrained": "openai",
                    "feature_dim": 1024,
                    "min_require_size": floor,
                    "partition_seed": 42,
                    "training_seed": seed,
                    "method": method,
                    "best_accuracy": accuracy,
                    "best_round": 100,
                    "checkpoint_hash": "abc",
                    "cache_file": "cache.pt",
                    "log_file": "console.log",
                    "run_status": "complete",
                })
    return rows


class SummaryStatisticsTests(unittest.TestCase):
    def test_aggregate_uses_sample_standard_deviation(self):
        result = aggregate_runs(fixture_rows())
        item = next(row for row in result if row["min_require_size"] == 0
                    and row["method"] == "fedproref")
        self.assertEqual(item["n"], 3)
        self.assertTrue(item["complete"])
        self.assertEqual(item["training_seeds"], "42;43;44")
        self.assertAlmostEqual(item["mean_accuracy"], 72.0)
        self.assertAlmostEqual(item["sample_sd"], 2.0)

    def test_paired_deltas_are_per_seed_and_count_strict_wins(self):
        result = paired_deltas(fixture_rows())
        proto = [row for row in result if row["min_require_size"] == 0
                 and row["control_method"] == "proto_aug"]
        self.assertEqual([row["training_seed"] for row in proto], [42, 43, 44])
        self.assertEqual([row["delta"] for row in proto], [4.0, 5.0, 6.0])
        self.assertTrue(all(row["win"] for row in proto))
        self.assertTrue(all(row["win_count"] == 3 for row in proto))
        self.assertTrue(all(math.isclose(row["mean_delta"], 5.0) for row in proto))

    def test_floor_zero_is_never_paired_with_floor_one(self):
        result = paired_deltas(fixture_rows())
        floor_one = [row for row in result if row["min_require_size"] == 1
                     and row["control_method"] == "proto_aug"]
        self.assertEqual([row["delta"] for row in floor_one], [2.0, 0.0, 4.0])
        self.assertEqual(floor_one[0]["win_count"], 2)

    def test_duplicate_cells_are_rejected(self):
        rows = fixture_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_runs(rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_deltas(rows)


class SummaryParsingAndOutputTests(unittest.TestCase):
    def test_only_strictly_completed_runs_are_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            good.mkdir()
            (good / ".done").touch()
            (good / "config.json").write_text(json.dumps({
                "run_id": "good",
                "dataset": "cifar100",
                "backbone": "RN50",
                "pretrained": "openai",
                "min_require_size": 1,
                "partition_seed": 42,
                "training_seed": 43,
                "method": "fedproref",
            }), encoding="utf-8")
            (good / "console.log").write_text(
                "  Checkpoint SHA-256: abcdef\n"
                "  Feature dimension: 1024\n"
                "  feature_cache_file             = /tmp/cache.pt\n"
                "Round   87 | acc: 74.2543 | ece: 0.1\n"
                "Round  100 | acc: 73.5000 | ece: 0.1\n"
                "  Best: Round 87 | Acc=74.25% | ECE=0.1 | NLL=1.0\n",
                encoding="utf-8",
            )
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "config.json").write_text("{}", encoding="utf-8")
            (incomplete / "console.log").write_text("Round  100 | acc: 99\n", encoding="utf-8")

            rows = parse_completed_runs(root)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["best_accuracy"], 74.2543)
            self.assertEqual(row["best_round"], 87)
            self.assertEqual(row["feature_dim"], 1024)
            self.assertEqual(row["checkpoint_hash"], "abcdef")
            self.assertEqual(row["cache_file"], "/tmp/cache.pt")

    def test_outputs_have_stable_files_and_separate_floor_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(fixture_rows(), Path(tmp))
            for name in ("runs.csv", "aggregate.csv", "paired_deltas.csv", "summary.md"):
                self.assertTrue((Path(tmp) / name).is_file(), name)
            markdown = (Path(tmp) / "summary.md").read_text(encoding="utf-8")
            self.assertIn("min_require_size=0", markdown)
            self.assertIn("min_require_size=1", markdown)
            self.assertNotIn("significance", markdown.lower())


if __name__ == "__main__":
    unittest.main()
