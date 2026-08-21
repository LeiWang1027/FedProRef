from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from direct_anchor_aug import DirectAnchorAugmentation, generate_direct_anchor_features
from proposal import sample_count_weighted_anchor_indices


def _anchors_and_counts():
    anchors = F.normalize(torch.tensor([[3.0, 4.0], [4.0, -3.0]]), dim=1)
    return anchors, torch.tensor([1.0, 9.0])


def _provider_args(num_samples=5, num_classes=1):
    return SimpleNamespace(
        seed=42,
        partition_seed=42,
        feat_dim=2,
        aug_gen_per_class=num_samples,
        min_samples_per_class=10,
        num_classes=num_classes,
        proposal_sigma=0.05,
        weak_class_percentile=0.0,
        proto_similarity_merge=True,
    )


class DirectAnchorAugTests(unittest.TestCase):
    def test_output_is_the_count_weighted_sampled_clean_anchor(self):
        anchors, counts = _anchors_and_counts()
        expected_generator = torch.Generator().manual_seed(7)
        expected_indices = sample_count_weighted_anchor_indices(
            counts, 11, expected_generator)

        output, indices = generate_direct_anchor_features(
            anchor_vectors=anchors,
            anchor_counts=counts,
            num_samples=11,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertTrue(torch.equal(indices, expected_indices))
        torch.testing.assert_close(output, anchors[expected_indices])
        self.assertFalse(output.requires_grad)

    def test_sampling_is_count_weighted(self):
        anchors, counts = _anchors_and_counts()

        output, indices = generate_direct_anchor_features(
            anchor_vectors=anchors,
            anchor_counts=counts,
            num_samples=4000,
            generator=torch.Generator().manual_seed(11),
        )

        selected_second = (indices == 1).float().mean().item()
        self.assertGreater(selected_second, 0.84)
        self.assertLess(selected_second, 0.96)
        torch.testing.assert_close(output, anchors[indices])

    def test_generation_does_not_modify_the_anchor_pool(self):
        anchors, counts = _anchors_and_counts()
        before_anchors = anchors.clone()
        before_counts = counts.clone()

        generate_direct_anchor_features(
            anchor_vectors=anchors,
            anchor_counts=counts,
            num_samples=10,
            generator=torch.Generator().manual_seed(1),
        )

        self.assertTrue(torch.equal(anchors, before_anchors))
        self.assertTrue(torch.equal(counts, before_counts))

    def test_provider_never_uses_noise_or_refiner_generation(self):
        anchors, _ = _anchors_and_counts()
        pool = {0: {0: [(anchors[0], 5), (anchors[1], 1)]}}
        provider = DirectAnchorAugmentation(_provider_args())

        with mock.patch(
                "server_calibration.generate_calibration_features",
                side_effect=AssertionError("proposal/refiner path called")):
            features, labels = provider(
                client_id=0,
                client_labels=torch.empty(0, dtype=torch.long).numpy(),
                global_prototypes=pool,
                round_id=1,
                device="cpu",
            )

        self.assertEqual(tuple(features.shape), (5, 2))
        self.assertTrue(torch.equal(labels, torch.zeros(5, dtype=torch.long)))
        metadata = provider.metadata()
        self.assertFalse(metadata["uses_proposal_noise"])
        self.assertFalse(metadata["uses_refiner"])
        self.assertTrue(metadata["fedavg_weight_uses_original_real_count"])

    def test_provider_skips_an_empty_pool_with_dynamic_width(self):
        provider = DirectAnchorAugmentation(_provider_args())

        features, labels = provider(
            client_id=0,
            client_labels=torch.empty(0, dtype=torch.long).numpy(),
            global_prototypes={0: {}},
            round_id=1,
            device="cpu",
        )

        self.assertEqual(tuple(features.shape), (0, 2))
        self.assertEqual(tuple(labels.shape), (0,))

    def test_synthetic_cache_is_separated_by_round(self):
        anchors, _ = _anchors_and_counts()
        pool = {0: {0: [(anchors[0], 5), (anchors[1], 1)]}}
        provider = DirectAnchorAugmentation(_provider_args())

        for round_id in (1, 2):
            provider(
                client_id=0,
                client_labels=torch.empty(0, dtype=torch.long).numpy(),
                global_prototypes=pool,
                round_id=round_id,
                device="cpu",
            )

        self.assertEqual(provider.cache_misses, 2)
        self.assertEqual(provider.cache_hits, 0)

    def test_synthetic_cache_discards_prior_round_entries(self):
        anchors, _ = _anchors_and_counts()
        pool = {0: {0: [(anchors[0], 5), (anchors[1], 1)]}}
        provider = DirectAnchorAugmentation(_provider_args())

        for round_id in range(1, 21):
            provider(
                client_id=0,
                client_labels=torch.empty(0, dtype=torch.long).numpy(),
                global_prototypes=pool,
                round_id=round_id,
                device="cpu",
            )

        self.assertEqual(len(provider.cache), 1)


if __name__ == "__main__":
    unittest.main()
