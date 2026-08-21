import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from data_utils import _feature_cache_file, _load_feature_cache, dirichlet_partition


class DirichletPartitionTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(8, dtype=np.int64), 25)

    def test_all_clients_nonempty_without_requiring_every_class(self):
        partitions = dirichlet_partition(
            self.labels,
            num_clients=6,
            alpha=0.05,
            num_classes=8,
            seed=17,
            max_attempts=10_000,
        )

        self.assertTrue(all(len(indices) > 0 for indices in partitions))
        per_client_class_counts = np.stack([
            np.bincount(self.labels[indices], minlength=8)
            for indices in partitions
        ])
        self.assertTrue(np.any(per_client_class_counts == 0))

    def test_repairs_extreme_sparse_partition_without_adding_a_class_floor(self):
        labels = np.repeat(np.arange(10, dtype=np.int64), 100)

        partitions = dirichlet_partition(
            labels,
            num_clients=50,
            alpha=0.01,
            num_classes=10,
            min_require_size=0,
            seed=42,
            max_attempts=1,
        )

        self.assertTrue(all(len(indices) > 0 for indices in partitions))
        assigned = np.concatenate(partitions)
        np.testing.assert_array_equal(
            np.sort(assigned), np.arange(len(labels), dtype=np.int64))
        per_client_class_counts = np.stack([
            np.bincount(labels[indices], minlength=10)
            for indices in partitions
        ])
        self.assertTrue(np.any(per_client_class_counts == 0))

    def test_default_uses_the_fiftieth_valid_draw_before_fallback_repair(self):
        labels = np.repeat(np.arange(2, dtype=np.int64), 3)

        partitions = dirichlet_partition(
            labels,
            num_clients=3,
            alpha=0.02,
            num_classes=2,
            min_require_size=0,
            seed=190,
        )

        self.assertEqual(
            [indices.tolist() for indices in partitions],
            [[3], [4, 5], [0, 1, 2]],
        )

    def test_floor_one_gives_every_client_every_class(self):
        partitions = dirichlet_partition(
            self.labels,
            num_clients=6,
            alpha=0.05,
            num_classes=8,
            min_require_size=1,
            seed=17,
        )

        per_client_class_counts = np.stack([
            np.bincount(self.labels[indices], minlength=8)
            for indices in partitions
        ])
        self.assertTrue(np.all(per_client_class_counts >= 1))

    def test_floor_one_rejects_an_impossible_class_floor(self):
        labels = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "class 0"):
            dirichlet_partition(
                labels,
                num_clients=3,
                alpha=0.1,
                num_classes=2,
                min_require_size=1,
                seed=42,
            )

    def test_rejects_negative_class_floor(self):
        with self.assertRaisesRegex(ValueError, "min_require_size"):
            dirichlet_partition(
                self.labels,
                num_clients=6,
                alpha=0.1,
                num_classes=8,
                min_require_size=-1,
                seed=42,
            )

    def test_assigns_every_source_index_exactly_once(self):
        partitions = dirichlet_partition(
            self.labels, 6, 0.2, 8, seed=23, max_attempts=10_000)

        assigned = np.concatenate(partitions)
        np.testing.assert_array_equal(
            np.sort(assigned), np.arange(len(self.labels), dtype=np.int64))

    def test_same_seed_reproduces_the_same_partition(self):
        first = dirichlet_partition(
            self.labels, 6, 0.1, 8, seed=31, max_attempts=10_000)
        second = dirichlet_partition(
            self.labels, 6, 0.1, 8, seed=31, max_attempts=10_000)

        self.assertEqual(len(first), len(second))
        for first_client, second_client in zip(first, second):
            np.testing.assert_array_equal(first_client, second_client)

    def test_rejects_more_clients_than_samples(self):
        with self.assertRaises(ValueError):
            dirichlet_partition(
                np.array([0, 1], dtype=np.int64),
                num_clients=3,
                alpha=0.1,
                num_classes=2,
            )

    def test_rejects_nonpositive_alpha(self):
        for alpha in (0.0, -0.1):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                dirichlet_partition(
                    self.labels, 6, alpha, 8, seed=1, max_attempts=10_000)

    def test_rejects_nonpositive_client_count(self):
        for num_clients in (0, -1):
            with self.subTest(num_clients=num_clients), self.assertRaises(ValueError):
                dirichlet_partition(
                    self.labels,
                    num_clients,
                    0.1,
                    8,
                    seed=1,
                    max_attempts=10_000,
                )


class CacheNamingTests(unittest.TestCase):
    def setUp(self):
        self.identity = {
            "backbone": "ViT-B-16",
            "pretrained": "/models/clip.safetensors",
            "source": "/models/clip.safetensors",
            "checkpoint_hash": "a" * 64,
        }

    def test_cache_name_identifies_the_nonempty_partition_scheme(self):
        args = SimpleNamespace(
            data_dir="/tmp/fedproref-test-data",
            dataset="cifar100",
            backbone="ViT-B-16",
            pretrained="/models/clip.safetensors",
            alpha=0.01,
            num_clients=10,
            min_require_size=0,
            partition_seed=42,
            seed=99,
        )

        cache_file = _feature_cache_file(args, self.identity)

        self.assertEqual(
            os.path.basename(cache_file),
            "ViT-B-16_clip.safetensors_aaaaaaaaaaaa_minpc0_partition-nonempty-v4_"
            "alpha0.01_c10_ps42.npz",
        )

    def test_cache_name_separates_floor_zero_and_floor_one(self):
        base = dict(
            data_dir="/tmp/fedproref-test-data",
            dataset="cifar100",
            backbone="ViT-B-16",
            pretrained="/models/clip.safetensors",
            alpha=0.01,
            num_clients=10,
            partition_seed=42,
            seed=99,
        )

        floor_zero = _feature_cache_file(
            SimpleNamespace(**base, min_require_size=0), self.identity)
        floor_one = _feature_cache_file(
            SimpleNamespace(**base, min_require_size=1), self.identity)

        self.assertIn("minpc0", os.path.basename(floor_zero))
        self.assertIn("minpc1", os.path.basename(floor_one))
        self.assertNotEqual(floor_zero, floor_one)

    def test_cache_name_separates_backbones_and_checkpoint_hashes(self):
        args = SimpleNamespace(
            data_dir="/tmp/fedproref-test-data",
            dataset="cifar100",
            backbone="RN50",
            pretrained="openai",
            alpha=0.01,
            num_clients=10,
            min_require_size=1,
            partition_seed=42,
            seed=99,
        )
        rn50_identity = {
            "backbone": "RN50",
            "pretrained": "openai",
            "source": "https://example.invalid/RN50.pt",
            "checkpoint_hash": "b" * 64,
        }

        cache_file = _feature_cache_file(args, rn50_identity)

        self.assertIn("RN50_openai_bbbbbbbbbbbb_minpc1", os.path.basename(cache_file))

    def test_multidomain_cache_name_keeps_its_existing_partition_scheme(self):
        args = SimpleNamespace(
            data_dir="/tmp/fedproref-test-data",
            dataset="pacs",
            backbone="ViT-B-16",
            pretrained="/models/clip.safetensors",
            alpha=0.3,
            num_clients=4,
            min_require_size=0,
            partition_seed=42,
            seed=99,
        )

        cache_file = _feature_cache_file(args, self.identity)

        self.assertEqual(
            os.path.basename(cache_file),
            "ViT-B-16_clip.safetensors_aaaaaaaaaaaa_minpc0_multidomain-v1_"
            "alpha0.3_c4_ps42.npz",
        )

    def test_cache_loader_rejects_feature_width_metadata_mismatch(self):
        args = SimpleNamespace(
            data_dir="/tmp/fedproref-test-data",
            dataset="cifar100",
            backbone="ViT-B-16",
            pretrained="/models/clip.safetensors",
            alpha=0.01,
            num_clients=2,
            min_require_size=0,
            partition_seed=42,
            seed=99,
        )
        metadata = {
            "dataset": "cifar100",
            "backbone": "ViT-B-16",
            "pretrained": "/models/clip.safetensors",
            "checkpoint_hash": "a" * 64,
            "feature_dim": 512,
            "alpha": 0.01,
            "num_clients": 2,
            "partition_seed": 42,
            "min_require_size": 0,
            "partition_scheme": "partition-nonempty-v4",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "bad.npz")
            np.savez(
                cache_file,
                metadata=np.array(json.dumps(metadata)),
                train_features=np.zeros((4, 1024), dtype=np.float32),
                train_labels=np.array([0, 0, 1, 1], dtype=np.int64),
                test_features=np.zeros((2, 1024), dtype=np.float32),
                test_labels=np.array([0, 1], dtype=np.int64),
                client_indices=np.array([
                    np.array([0, 1]), np.array([2, 3])], dtype=object),
            )

            with self.assertRaisesRegex(ValueError, "feature_dim"):
                _load_feature_cache(cache_file, args, self.identity)

    def test_cache_loader_accepts_relocated_local_checkpoint_with_same_hash(self):
        args = SimpleNamespace(
            data_dir="/tmp/fedproref-test-data",
            dataset="cifar100",
            backbone="ViT-B-16",
            pretrained="/new/project/models/clip.safetensors",
            alpha=0.01,
            num_clients=2,
            min_require_size=0,
            partition_seed=42,
            seed=99,
        )
        relocated_identity = {
            "backbone": "ViT-B-16",
            "pretrained": "/new/project/models/clip.safetensors",
            "source": "/new/project/models/clip.safetensors",
            "checkpoint_hash": "a" * 64,
        }
        metadata = {
            "dataset": "cifar100",
            "backbone": "ViT-B-16",
            "pretrained": "/old/project/models/clip.safetensors",
            "source": "/old/project/models/clip.safetensors",
            "checkpoint_hash": "a" * 64,
            "feature_dim": 512,
            "alpha": 0.01,
            "num_clients": 2,
            "partition_seed": 42,
            "min_require_size": 0,
            "partition_scheme": "partition-nonempty-v4",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "relocated.npz")
            np.savez(
                cache_file,
                metadata=np.array(json.dumps(metadata)),
                train_features=np.zeros((4, 512), dtype=np.float32),
                train_labels=np.array([0, 0, 1, 1], dtype=np.int64),
                test_features=np.zeros((2, 512), dtype=np.float32),
                test_labels=np.array([0, 1], dtype=np.int64),
                client_indices=np.array([
                    np.array([0, 1]), np.array([2, 3])], dtype=object),
            )

            client_data, test_features, test_labels = _load_feature_cache(
                cache_file, args, relocated_identity)

        self.assertEqual(len(client_data), 2)
        self.assertEqual(tuple(test_features.shape), (2, 512))
        self.assertEqual(test_labels.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
