"""Tests for augmentation pipeline integration with Feeder.

The augmentation functions themselves are tested in pybvh-ml.
These tests verify that the Feeder correctly applies/skips augmentation
in train/test mode, and that the pipeline integration works end-to-end.

Representation tokens and angle units follow pybvh 0.8 / pybvh-ml 0.5:
short tokens (``quat``) and radians (``angle=``, ``sigma=``).
"""

import json
import math

import numpy as np
import pytest
import torch

import pybvh_ml
from emo_mocap.cli.train import _build_pipeline
from emo_mocap.data.feeder import Feeder
from emo_mocap.tools.config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_single_clip_npz(tmp_path, name, F=100, J=24, seed=42,
                          repr_token="quat"):
    """Create a single-clip npz with quaternion data in pybvh-ml format.

    ``repr_token`` is the representation string written into the dataset
    metadata; pass ``"quaternion"`` to simulate a dataset preprocessed by
    pybvh-ml < 0.5.
    """
    rng = np.random.RandomState(seed)
    root_pos = rng.randn(F, 3).astype(np.float64)
    quats = rng.randn(F, J, 4).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    skel_info = {"num_joints": J, "euler_orders": ["ZYX"] * J,
                 "joint_names": [f"j{i}" for i in range(J)],
                 "edges": [], "lr_pairs": []}
    path = tmp_path / f"{name}.npz"
    np.savez(path,
             clip_0_root_pos=root_pos,
             clip_0_joint_rot=quats,
             labels=np.array([0], dtype=np.int64),
             filenames=np.array(["test"]),
             num_clips=np.array(1),
             representation=np.array(repr_token),
             skeleton_info_json=np.array(json.dumps(skel_info)),
             mean=np.zeros(3 + J * 4), std=np.ones(3 + J * 4))
    return str(path)


DIEMA_LR_PAIRS = [
    (12, 8), (13, 9), (14, 10), (15, 11),
    (20, 16), (21, 17), (22, 18), (23, 19),
]


@pytest.fixture
def sample_npz(tmp_path):
    """Create a sample npz file for testing."""
    return _make_single_clip_npz(tmp_path, "sample", F=100, J=24)


@pytest.fixture
def noise_pipeline():
    """Pipeline that only adds noise (guaranteed to change data)."""
    return pybvh_ml.AugmentationPipeline([
        (pybvh_ml.add_joint_rotation_noise, 1.0,
         {"sigma": math.radians(10.0), "representation": "quat"}),
    ])


@pytest.fixture
def full_pipeline():
    """Pipeline with rotation, mirror, and noise."""
    return pybvh_ml.AugmentationPipeline([
        (pybvh_ml.rotate_vertical, 1.0, {
            "angle": lambda rng: rng.uniform(-math.pi, math.pi),
            "up_axis": "+z", "representation": "quat",
        }),
        (pybvh_ml.mirror, 0.5, {
            "lr_joint_pairs": DIEMA_LR_PAIRS,
            "lateral_axis": "+x", "representation": "quat",
        }),
        (pybvh_ml.add_joint_rotation_noise, 1.0,
         {"sigma": math.radians(1.0), "representation": "quat"}),
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineShapePreservation:
    """All augmentations preserve the output tensor shape."""

    def test_noise_preserves_shape(self, sample_npz, noise_pipeline):
        feeder = Feeder(sample_npz, clip_length=64, target_repr="quat",
                        test=False, augmentation_pipeline=noise_pipeline)
        tensor, _, _ = feeder[0]
        assert tensor.shape == (4, 64, 25)

    def test_full_pipeline_preserves_shape(self, sample_npz, full_pipeline):
        feeder = Feeder(sample_npz, clip_length=64, target_repr="quat",
                        test=False, augmentation_pipeline=full_pipeline)
        tensor, _, _ = feeder[0]
        assert tensor.shape == (4, 64, 25)

    def test_euler_output_shape(self, sample_npz, noise_pipeline):
        feeder = Feeder(sample_npz, clip_length=64, target_repr="euler",
                        test=False, augmentation_pipeline=noise_pipeline,
                        euler_orders=["ZYX"] * 24)
        tensor, _, _ = feeder[0]
        assert tensor.shape == (3, 64, 25)


class TestAugmentationApplied:
    """Pipeline is applied in train mode and skipped in test mode."""

    def test_feeder_skips_pipeline_in_test_mode(self, sample_npz, noise_pipeline):
        """Test mode: pipeline is ignored even if provided."""
        feeder_no_aug = Feeder(sample_npz, clip_length=64, target_repr="quat",
                               test=True)
        feeder_with_aug = Feeder(sample_npz, clip_length=64, target_repr="quat",
                                  test=True, augmentation_pipeline=noise_pipeline)
        t1, _, _ = feeder_no_aug[0]
        t2, _, _ = feeder_with_aug[0]
        assert torch.allclose(t1, t2), "Test mode should ignore augmentation pipeline"

    def test_feeder_applies_pipeline_in_train_mode(self, sample_npz, noise_pipeline):
        """Train mode: pipeline is applied, data differs from unaugmented."""
        feeder_no_aug = Feeder(sample_npz, clip_length=64, target_repr="quat",
                               test=True)
        feeder_with_aug = Feeder(sample_npz, clip_length=64, target_repr="quat",
                                  test=False, augmentation_pipeline=noise_pipeline)
        t_clean, _, _ = feeder_no_aug[0]
        t_augmented, _, _ = feeder_with_aug[0]
        # With sigma = 10 degrees the data should be noticeably different
        assert not torch.allclose(t_clean, t_augmented, atol=0.1), \
            "Train mode with noise should change the data"


class TestDeterminism:
    """Same seed produces same augmentation."""

    def test_deterministic_with_same_seed(self, sample_npz, noise_pipeline):
        feeder1 = Feeder(sample_npz, clip_length=64, target_repr="quat",
                         test=False, seed=42, augmentation_pipeline=noise_pipeline)
        feeder2 = Feeder(sample_npz, clip_length=64, target_repr="quat",
                         test=False, seed=42, augmentation_pipeline=noise_pipeline)
        t1, _, _ = feeder1[0]
        t2, _, _ = feeder2[0]
        assert torch.allclose(t1, t2), "Same seed should produce identical results"

    def test_different_seeds_differ(self, sample_npz, noise_pipeline):
        feeder1 = Feeder(sample_npz, clip_length=64, target_repr="quat",
                         test=False, seed=42, augmentation_pipeline=noise_pipeline)
        feeder2 = Feeder(sample_npz, clip_length=64, target_repr="quat",
                         test=False, seed=99, augmentation_pipeline=noise_pipeline)
        t1, _, _ = feeder1[0]
        t2, _, _ = feeder2[0]
        assert not torch.allclose(t1, t2), "Different seeds should produce different results"


class TestNoAugmentation:
    """Feeder works correctly without augmentation."""

    def test_no_pipeline_no_crash(self, sample_npz):
        feeder = Feeder(sample_npz, clip_length=64, target_repr="quat", test=False)
        tensor, label, filename = feeder[0]
        assert tensor.shape == (4, 64, 25)
        assert not torch.isnan(tensor).any()


class TestBuildPipeline:
    """_build_pipeline translates YAML config into a pybvh-ml pipeline."""

    def test_disabled_returns_none(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]]}\n"
            "augmentation: {enabled: false}\n"
        )
        assert _build_pipeline(load_config(cfg_path)) is None

    def test_dropout_is_wired(self, tmp_path):
        """The dropout flag on the YAML must produce a dropout_arrays step."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]],\n"
            "           lr_joint_pairs: [], up_axis: '+y', lateral_axis: '+x'}\n"
            "augmentation:\n"
            "  enabled: true\n"
            "  dropout: true\n"
            "  dropout_prob: 0.3\n"
            "  dropout_rate: 0.15\n"
        )
        pipeline = _build_pipeline(load_config(cfg_path))
        # pybvh-ml >= 0.5 steps are AugmentationStep named tuples.
        steps = {step.fn: step for step in pipeline.augmentations}
        assert pybvh_ml.dropout_arrays in steps
        assert steps[pybvh_ml.dropout_arrays].prob == 0.3
        assert steps[pybvh_ml.dropout_arrays].kwargs["drop_rate"] == 0.15

    def test_probabilities_threaded_through(self, tmp_path):
        """rotate_prob and speed_prob flags must land on the pipeline step."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]],\n"
            "           lr_joint_pairs: [], up_axis: '+y', lateral_axis: '+x'}\n"
            "augmentation:\n"
            "  enabled: true\n"
            "  rotate: true\n"
            "  rotate_prob: 0.7\n"
            "  speed: true\n"
            "  speed_prob: 0.4\n"
        )
        pipeline = _build_pipeline(load_config(cfg_path))
        by_fn = {step.fn: step.prob for step in pipeline.augmentations}
        assert by_fn[pybvh_ml.rotate_vertical] == 0.7
        assert by_fn[pybvh_ml.speed_perturbation_arrays] == 0.4


class TestRecipeConfigShape:
    """The shipped recipe must augment with mirror and nothing else.

    This is not a style preference. The recipe carries position channels, and
    rotation noise is applied in rotation space where forward kinematics
    amplifies it down the kinematic chain: a small jitter at the shoulder
    becomes a large displacement at the hand. Turning it on costs several
    points and stops the network fitting its training set at all. If this test
    ever fails, the config has drifted away from the measured recipe.
    """

    @pytest.mark.parametrize("config", [
        "configs/diema7_stgcn_recipe.yaml",
        "configs/diema13_stgcn_recipe.yaml",
    ])
    def test_recipe_is_mirror_only(self, config):
        cfg = load_config(config)
        pipeline = _build_pipeline(cfg)
        assert pipeline is not None
        fns = [step.fn for step in pipeline.augmentations]
        assert fns == [pybvh_ml.mirror], fns
