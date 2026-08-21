import hashlib
import os
import tempfile
import unittest

import numpy as np

from backbone_utils import (
    open_clip_load_kwargs,
    resolve_feature_dim,
    resolve_pretrained_identity,
)
from server_calibration import generate_calibration_features


class PretrainedIdentityTests(unittest.TestCase):
    def test_local_checkpoint_uses_full_sha256(self):
        payload = b"checkpoint-bytes"
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            checkpoint.write(payload)
            checkpoint.flush()
            identity = resolve_pretrained_identity("ViT-B-16", checkpoint.name)

        self.assertEqual(identity["checkpoint_hash"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(identity["pretrained"], os.path.realpath(checkpoint.name))
        self.assertEqual(identity["source"], os.path.realpath(checkpoint.name))

    def test_rn50_openai_identity_uses_official_checksum(self):
        identity = resolve_pretrained_identity("RN50", "openai")

        self.assertEqual(
            identity["checkpoint_hash"],
            "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
        )
        self.assertEqual(identity["backbone"], "RN50")
        self.assertEqual(identity["pretrained"], "openai")

    def test_missing_local_checkpoint_fails_as_a_path(self):
        with self.assertRaises(FileNotFoundError):
            resolve_pretrained_identity("ViT-B-16", "/missing/model.safetensors")

    def test_official_rn50_torchscript_uses_pytorch_26_compatibility(self):
        identity = {
            "backbone": "RN50",
            "checkpoint_hash": (
                "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"),
        }

        self.assertEqual(open_clip_load_kwargs(identity), {"weights_only": False})

    def test_other_checkpoints_keep_open_clip_default(self):
        identity = {"backbone": "ViT-B-16", "checkpoint_hash": "abc"}

        self.assertEqual(open_clip_load_kwargs(identity), {})


class FeatureDimensionTests(unittest.TestCase):
    def test_auto_dimension_uses_feature_width(self):
        resolved = resolve_feature_dim(
            None,
            np.zeros((4, 1024), dtype=np.float32),
            np.zeros((2, 1024), dtype=np.float32),
        )

        self.assertEqual(resolved, 1024)

    def test_expected_dimension_mismatch_fails(self):
        with self.assertRaisesRegex(
                ValueError, "expected feature dimension 512.*resolved 1024"):
            resolve_feature_dim(
                512,
                np.zeros((4, 1024), dtype=np.float32),
                np.zeros((2, 1024), dtype=np.float32),
            )

    def test_train_test_dimension_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "train/test feature dimensions differ"):
            resolve_feature_dim(
                None,
                np.zeros((4, 1024), dtype=np.float32),
                np.zeros((2, 512), dtype=np.float32),
            )

    def test_non_matrix_features_fail(self):
        with self.assertRaisesRegex(ValueError, "rank-two"):
            resolve_feature_dim(
                None,
                np.zeros((4, 2, 2), dtype=np.float32),
                np.zeros((2, 4), dtype=np.float32),
            )


class DynamicCalibrationShapeTests(unittest.TestCase):
    def test_empty_calibration_features_use_requested_width(self):
        features, labels = generate_calibration_features(
            None,
            {},
            num_classes=3,
            gen_per_class=5,
            proposal_sigma=0.05,
            device="cpu",
            feat_dim=1024,
            use_refiner=False,
        )

        self.assertEqual(tuple(features.shape), (0, 1024))
        self.assertEqual(tuple(labels.shape), (0,))


if __name__ == "__main__":
    unittest.main()
