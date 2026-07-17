"""Unit tests for the isolated DirectAnchorAug generation path."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase, mock

import torch
import torch.nn.functional as F

import federated_loop as fl
from direct_anchor_aug import DirectAnchorAugmentation, generate_direct_anchor_features
from proposal import sample_count_weighted_anchor_indices


ATOL = 1e-7
RTOL = 1e-6


def anchors_and_counts():
    anchors = F.normalize(torch.tensor([[3.0, 4.0], [4.0, -3.0]]), dim=1)
    return anchors, torch.tensor([1.0, 9.0])


def provider_args(num_samples=5, num_classes=1):
    return SimpleNamespace(
        seed=42,
        partition_seed=42,
        aug_gen_per_class=num_samples,
        min_samples_per_class=10,
        num_classes=num_classes,
        proposal_sigma=0.05,
        weak_class_percentile=0.0,
        proto_similarity_merge=True,
    )


class DirectAnchorAugTests(TestCase):
    def test_direct_anchor_output_equals_sampled_anchor(self):
        anchors, counts = anchors_and_counts()
        expected_generator = torch.Generator().manual_seed(7)
        expected_indices = sample_count_weighted_anchor_indices(counts, 11, expected_generator)
        generator = torch.Generator().manual_seed(7)
        output, indices = generate_direct_anchor_features(
            anchor_vectors=anchors, anchor_counts=counts, num_samples=11, generator=generator)
        self.assertTrue(torch.equal(indices, expected_indices))
        torch.testing.assert_close(output, anchors[expected_indices], atol=ATOL, rtol=RTOL)
        self.assertFalse(output.requires_grad)

    def test_direct_anchor_does_not_add_noise(self):
        anchors, counts = anchors_and_counts()
        with mock.patch("proposal.sample_proposal", side_effect=AssertionError("noise path called")), \
             mock.patch("proposal.torch.randn_like", side_effect=AssertionError("noise called")):
            output, _ = generate_direct_anchor_features(
                anchor_vectors=anchors, anchor_counts=counts, num_samples=4,
                generator=torch.Generator().manual_seed(3))
        self.assertEqual(tuple(output.shape), (4, 2))

    def test_direct_anchor_never_calls_refiner(self):
        anchors, _ = anchors_and_counts()
        pool = {0: {0: [(anchors[0], 5), (anchors[1], 1)]}}
        provider = DirectAnchorAugmentation(provider_args())
        with mock.patch("server_calibration.generate_calibration_features", side_effect=AssertionError("refiner path called")):
            features, labels = provider(
                client_id=0, client_labels=torch.empty(0, dtype=torch.long).numpy(),
                global_prototypes=pool, round_id=1, device="cpu")
        self.assertEqual(features.shape[0], 5)
        self.assertTrue(torch.equal(labels, torch.zeros(5, dtype=torch.long)))

    def test_direct_anchor_generates_exactly_M_samples(self):
        anchors, counts = anchors_and_counts()
        features, _ = generate_direct_anchor_features(
            anchor_vectors=anchors, anchor_counts=counts, num_samples=13,
            generator=torch.Generator().manual_seed(9))
        self.assertEqual(tuple(features.shape), (13, 2))

    def test_direct_anchor_labels_are_correct(self):
        anchors, _ = anchors_and_counts()
        provider = DirectAnchorAugmentation(provider_args(num_samples=6, num_classes=2))
        pool = {0: {}, 1: {0: [(anchors[0], 1), (anchors[1], 2)]}}
        features, labels = provider(
            client_id=1, client_labels=torch.zeros(3, dtype=torch.long).numpy(),
            global_prototypes=pool, round_id=1, device="cpu")
        self.assertEqual(features.shape[0], 6)
        self.assertTrue(torch.equal(labels, torch.ones(6, dtype=torch.long)))

    def test_direct_anchor_uses_count_weighted_sampling(self):
        anchors, counts = anchors_and_counts()
        output, indices = generate_direct_anchor_features(
            anchor_vectors=anchors, anchor_counts=counts, num_samples=4000,
            generator=torch.Generator().manual_seed(11))
        self.assertGreater((indices == 1).float().mean().item(), 0.84)
        self.assertLess((indices == 1).float().mean().item(), 0.96)
        torch.testing.assert_close(output, anchors[indices], atol=ATOL, rtol=RTOL)

    def test_direct_anchor_skips_empty_pool(self):
        provider = DirectAnchorAugmentation(provider_args())
        features, labels = provider(
            client_id=0, client_labels=torch.empty(0, dtype=torch.long).numpy(),
            global_prototypes={0: {}}, round_id=1, device="cpu")
        self.assertEqual(features.numel(), 0)
        self.assertEqual(labels.numel(), 0)

    def test_direct_anchor_uses_same_merged_pool(self):
        anchors, _ = anchors_and_counts()
        provider = DirectAnchorAugmentation(provider_args(num_samples=20))
        merged_pool = {0: {3: [(anchors[0], 1), (anchors[1], 9)]}}
        features, _ = provider(
            client_id=0, client_labels=torch.empty(0, dtype=torch.long).numpy(),
            global_prototypes=merged_pool, round_id=1, device="cpu")
        allowed = [torch.allclose(row, anchors[0], atol=ATOL, rtol=RTOL) or
                   torch.allclose(row, anchors[1], atol=ATOL, rtol=RTOL) for row in features]
        self.assertTrue(all(allowed))

    def test_direct_anchor_does_not_modify_anchor_pool(self):
        anchors, counts = anchors_and_counts()
        before_anchors = anchors.clone()
        before_counts = counts.clone()
        generate_direct_anchor_features(
            anchor_vectors=anchors, anchor_counts=counts, num_samples=10,
            generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(anchors, before_anchors))
        self.assertTrue(torch.equal(counts, before_counts))

    def test_direct_anchor_does_not_affect_fedavg_weight(self):
        first = fl.create_head("linear", 2, 1)
        second = fl.create_head("linear", 2, 1)
        for parameter in first.parameters():
            parameter.data.zero_()
        for parameter in second.parameters():
            parameter.data.fill_(2.0)
        global_head = fl.create_head("linear", 2, 1)
        result = fl.fedavg_heads(global_head, [first, second], [2 / 5, 3 / 5])
        for parameter in result.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 1.2), atol=ATOL, rtol=RTOL)

    def test_existing_methods_regression(self):
        args = provider_args()
        self.assertEqual(args.aug_gen_per_class, 5)
        self.assertNotEqual("fedproref", "direct_anchor_aug")
        self.assertNotEqual("proto_aug", "direct_anchor_aug")
        self.assertNotEqual("proto_cal", "direct_anchor_aug")
