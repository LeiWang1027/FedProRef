import unittest

import numpy as np

from partition_report import audit_index_partition


class PartitionAuditTests(unittest.TestCase):
    def test_index_audit_counts_missing_cells_and_proves_exact_coverage(self):
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        indices = [
            np.asarray([0, 2, 4], dtype=np.int64),
            np.asarray([1, 3, 5], dtype=np.int64),
        ]

        result = audit_index_partition(labels, indices, num_classes=3)

        self.assertEqual(result["matrix"].tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertTrue(result["integrity"]["all_training_samples_assigned_once"])
        self.assertTrue(result["integrity"]["no_out_of_range_indices"])
        self.assertTrue(result["integrity"]["no_within_client_duplicates"])
        self.assertTrue(result["integrity"]["no_cross_client_duplicates"])

    def test_index_audit_rejects_cross_client_duplicate(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        indices = [
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1, 2, 3], dtype=np.int64),
        ]

        with self.assertRaisesRegex(ValueError, "cross-client duplicate"):
            audit_index_partition(labels, indices, num_classes=2)


if __name__ == "__main__":
    unittest.main()
