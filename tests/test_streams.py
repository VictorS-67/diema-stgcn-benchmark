"""Position/velocity stream support (pybvh-ml >= 0.6) through the Feeder.

Builds a small real preprocessed dataset (with positions) from a fixture BVH
via pybvh_ml.preprocess_directory — the genuine producer, not a mock — and
checks the packed shapes, the derived-stream semantics, the config shape
cross-check, and the augmentation wiring for position-carrying samples.
"""

import shutil

import numpy as np
import pytest
import torch

import pybvh_ml

from emo_mocap.data.feeder import Feeder
from emo_mocap.cli.train import (
    _build_pipeline,
    _check_streams_shape,
    _needed_position_fields,
)

# bvh_example has 56 motion frames; standard_skeleton is a 1-frame rest pose
# and cannot exercise temporal sampling or velocities.
FIXTURE_BVH = "tests/fixtures/bvh/bvh_example.bvh"


@pytest.fixture(scope="module")
def pos_dataset(tmp_path_factory):
    """A 3-clip preprocessed dataset carrying joint positions."""
    root = tmp_path_factory.mktemp("streams")
    bvh_dir = root / "bvh"
    bvh_dir.mkdir()
    for i in range(3):
        shutil.copy(FIXTURE_BVH, bvh_dir / f"clip_{i}.bvh")
    out = root / "pos.npz"
    pybvh_ml.preprocess_directory(
        bvh_dir=bvh_dir,
        output_path=out,
        representation="quat",
        center_root=True,
        include_positions=True,
        position_space="joint",
        position_centering="skeleton",
        label_fn=lambda stem: int(stem.split("_")[-1]) % 3,
    )
    return str(out)


@pytest.fixture(scope="module")
def rot_dataset(tmp_path_factory):
    """The same clips preprocessed WITHOUT positions (0.5-style file)."""
    root = tmp_path_factory.mktemp("streams_rot")
    bvh_dir = root / "bvh"
    bvh_dir.mkdir()
    shutil.copy(FIXTURE_BVH, bvh_dir / "clip_0.bvh")
    out = root / "rot.npz"
    pybvh_ml.preprocess_directory(
        bvh_dir=bvh_dir, output_path=out, representation="quat",
        center_root=True, label_fn=lambda stem: 0,
    )
    return str(out)


def _num_joints(path):
    return pybvh_ml.load_preprocessed(path)["skeleton_info"]["num_joints"]


class TestFeederStreams:
    def test_joint_pos_shape(self, pos_dataset):
        J = _num_joints(pos_dataset)
        f = Feeder(pos_dataset, clip_length=16, test=True,
                   streams=("joint_pos",))
        x, y, name = f[0]
        assert x.shape == (3, 16, J)
        assert x.dtype == torch.float32

    def test_pos_vel_acc_shape(self, pos_dataset):
        J = _num_joints(pos_dataset)
        f = Feeder(pos_dataset, clip_length=16, test=True,
                   streams=("joint_pos", "joint_vel", "joint_acc"))
        x, _, _ = f[0]
        assert x.shape == (9, 16, J)

    def test_vel_only_shape(self, pos_dataset):
        J = _num_joints(pos_dataset)
        f = Feeder(pos_dataset, clip_length=16, test=True,
                   streams=("joint_vel",))
        x, _, _ = f[0]
        assert x.shape == (3, 16, J)

    def test_default_streams_keep_root_vertex(self, pos_dataset):
        """streams=None on a positions-carrying file: the 0.5 packing."""
        J = _num_joints(pos_dataset)
        f = Feeder(pos_dataset, clip_length=16, test=True, target_repr="quat")
        x, _, _ = f[0]
        assert x.shape == (4, 16, 1 + J)

    def test_default_streams_ignore_positions_entirely(self, pos_dataset,
                                                       rot_dataset):
        """A rotation-only run must not depend on the file carrying
        positions: same clip, with and without stored positions, packs
        identically."""
        fa = Feeder(pos_dataset, clip_length=16, test=True, target_repr="quat")
        fb = Feeder(rot_dataset, clip_length=16, test=True, target_repr="quat")
        xa, _, _ = fa[0]
        xb, _, _ = fb[0]
        assert torch.equal(xa, xb)

    def test_missing_positions_error_names_the_fix(self, rot_dataset):
        with pytest.raises(ValueError, match="include-positions"):
            Feeder(rot_dataset, streams=("joint_pos",))
        with pytest.raises(ValueError, match="include-positions"):
            Feeder(rot_dataset, streams=("joint_vel",))

    def test_velocity_derived_before_temporal_sampling(self, pos_dataset):
        """The packed velocity is the full-clip derivative *sampled*, not the
        derivative of the sampled positions — PYSKL's GenSkeFeat-before-
        UniformSample ordering. With a clip_length shorter than the clip, the
        sampled frames are non-contiguous, so the two orderings genuinely
        differ; the Feeder must land on the derive-first side."""
        loaded = pybvh_ml.load_preprocessed(pos_dataset)
        clip = loaded["clips"][0]
        F = clip["joint_pos"].shape[0]
        L = F // 2  # forces a strided deterministic test-mode sample
        f = Feeder(pos_dataset, clip_length=L, test=True,
                   streams=("joint_pos", "joint_vel"))
        x, _, _ = f[0]
        pos, vel = x[:3].numpy(), x[3:].numpy()

        arrays = pybvh_ml.MotionArrays(
            root_pos=clip["root_pos"], joint_rot=clip["joint_rot"],
            joint_pos=clip["joint_pos"],
            position_centering=loaded["position_centering"])
        full = pybvh_ml.pack_to_ctv(
            arrays, streams=("joint_pos", "joint_vel"), center_root=False)
        idx = pybvh_ml.uniform_temporal_sample(F, L, mode="test", rng=None) % F
        expect = full[:, idx, :].astype(np.float32)
        np.testing.assert_array_equal(pos, expect[:3])
        np.testing.assert_array_equal(vel, expect[3:])

        # and it must NOT equal diff-of-sampled-positions (they only agree
        # when the sample stride is 1)
        naive = np.zeros_like(pos)
        naive[:, 1:] = pos[:, 1:] - pos[:, :-1]
        assert not np.allclose(vel, naive)

    def test_augmented_positions_stay_fk_coherent(self, pos_dataset):
        """A pipeline built with the position wiring runs on a positions
        sample; determinism per (seed, epoch, idx) still holds."""
        loaded = pybvh_ml.load_preprocessed(pos_dataset)
        si = loaded["skeleton_info"]
        pipeline = pybvh_ml.AugmentationPipeline(
            [(pybvh_ml.rotate_vertical, 1.0,
              {"angle": lambda rng: rng.uniform(-3.14, 3.14),
               "up_axis": si["world_up"] or "+y"}),
             (pybvh_ml.add_joint_rotation_noise, 1.0,
              {"sigma": 0.02,
               "fk_topology": pybvh_ml.build_fk_topology(si)})],
            representation="quat")
        f = Feeder(pos_dataset, clip_length=16, test=False, seed=1,
                   augmentation_pipeline=pipeline, streams=("joint_pos",))
        f.set_epoch(0)
        x1, _, _ = f[0]
        x2, _, _ = f[0]
        assert torch.equal(x1, x2)
        f.set_epoch(1)
        x3, _, _ = f[0]
        assert not torch.equal(x1, x3)


class TestStreamsShapeCheck:
    def _cfg(self, in_channels, num_nodes):
        from types import SimpleNamespace
        return SimpleNamespace(
            model=SimpleNamespace(in_channels=in_channels),
            skeleton=SimpleNamespace(num_nodes=num_nodes),
        )

    def test_channel_arithmetic(self):
        si = {"num_joints": 24}
        _check_streams_shape(self._cfg(3, 24), ["joint_pos"], "6d", si)
        _check_streams_shape(self._cfg(9, 24),
                             ["joint_pos", "joint_vel", "joint_acc"], "6d", si)
        _check_streams_shape(self._cfg(6, 25), None, "6d", si)
        # root_pos adds vertex 0, never channels
        _check_streams_shape(self._cfg(4, 25), ["root_pos", "joint_rot"],
                             "quat", si)
        _check_streams_shape(self._cfg(3, 25), ["root_pos", "joint_pos"],
                             "6d", si)

    def test_channel_mismatch_raises(self):
        with pytest.raises(ValueError, match="in_channels"):
            _check_streams_shape(self._cfg(6, 24), ["joint_pos"], "6d",
                                 {"num_joints": 24})

    def test_vertex_mismatch_raises(self):
        with pytest.raises(ValueError, match="root vertex"):
            _check_streams_shape(self._cfg(3, 25), ["joint_pos"], "6d",
                                 {"num_joints": 24})

    def test_no_skeleton_info_skips_vertex_check(self):
        _check_streams_shape(self._cfg(3, 999), ["joint_pos"], "6d", {})

    def test_needed_position_fields(self):
        assert _needed_position_fields(None) == set()
        assert _needed_position_fields(["joint_pos"]) == {"joint_pos"}
        assert _needed_position_fields(["joint_vel"]) == {"joint_pos"}
        assert _needed_position_fields(["node_acc", "joint_rot"]) == {"node_pos"}


# --------------------------------------------------------------------------
# scale_normalize — dividing body size out of the position streams
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mixed_size_dataset(tmp_path_factory):
    """Two clips of the same motion on skeletons of different size.

    The second clip's BVH has every OFFSET doubled, so it is the same
    performance by a performer twice as large. That is exactly the nuisance
    variable `scale_normalize` exists to remove, and it cannot be exercised by
    copies of one fixture — copies all have the same bone lengths and the
    correction is the identity.
    """
    root = tmp_path_factory.mktemp("scaled")
    bvh_dir = root / "bvh"
    bvh_dir.mkdir()
    shutil.copy(FIXTURE_BVH, bvh_dir / "clip_0.bvh")

    def _double_offsets(line):
        stripped = line.strip()
        if not stripped.startswith("OFFSET"):
            return line
        head, *vals = stripped.split()
        return (line[: len(line) - len(line.lstrip())]
                + head + " " + " ".join(f"{2 * float(v):.6f}" for v in vals)
                + "\n")

    src = (bvh_dir / "clip_0.bvh").read_text().splitlines(keepends=True)
    (bvh_dir / "clip_1.bvh").write_text("".join(map(_double_offsets, src)))

    out = root / "mixed.npz"
    pybvh_ml.preprocess_directory(
        bvh_dir=bvh_dir, output_path=out, representation="quat",
        center_root=True, include_positions=True, position_space="joint",
        position_centering="skeleton", label_fn=lambda stem: 0,
    )
    return str(out)


class TestScaleNormalize:
    def test_skeleton_scale_is_total_bone_length(self):
        """Three collinear joints one unit apart: scale = 2."""
        from emo_mocap.data.feeder import skeleton_scale
        pos = np.zeros((4, 3, 3), dtype=np.float32)
        pos[:, 1, 0] = 1.0
        pos[:, 2, 0] = 2.0
        edges = np.array([[1, 0], [2, 1]])
        assert skeleton_scale(pos, edges) == pytest.approx(2.0)

    def test_skeleton_scale_ignores_translation(self):
        """A per-frame offset cancels in the parent-child difference, so the
        centering convention of the stored positions cannot change it."""
        from emo_mocap.data.feeder import skeleton_scale
        pos = np.zeros((2, 2, 3), dtype=np.float32)
        pos[:, 1, 2] = 3.0
        edges = np.array([[1, 0]])
        shifted = pos + np.array([10.0, -5.0, 2.0], dtype=np.float32)
        assert skeleton_scale(shifted, edges) == pytest.approx(
            skeleton_scale(pos, edges))

    def test_twice_the_performer_packs_the_same(self, mixed_size_dataset):
        """The point of the whole option: same motion, two body sizes, one
        packed tensor."""
        f = Feeder(mixed_size_dataset, clip_length=16, test=True,
                   streams=("joint_pos", "joint_vel", "joint_acc"),
                   scale_normalize=True)
        small, _, _ = f[0]
        large, _, _ = f[1]
        assert torch.allclose(small, large, atol=1e-4)

    def test_without_the_option_they_differ(self, mixed_size_dataset):
        f = Feeder(mixed_size_dataset, clip_length=16, test=True,
                   streams=("joint_pos", "joint_vel", "joint_acc"))
        small, _, _ = f[0]
        large, _, _ = f[1]
        assert not torch.allclose(small, large, atol=1e-4)
        # The larger skeleton is twice the size, so its packed positions are
        # twice the magnitude — the nuisance signal, quantified.
        assert large.abs().mean() == pytest.approx(2 * small.abs().mean(),
                                                   rel=1e-3)

    def test_reference_is_corpus_wide_not_per_subset(self, mixed_size_dataset):
        """A Feeder holding only the large performer must divide by the same
        constant as one holding both — otherwise train and test end up in
        different units and the checkpoint means nothing on the test split."""
        both = Feeder(mixed_size_dataset, clip_length=16, test=True,
                      streams=("joint_pos",), scale_normalize=True)
        large_only = Feeder(mixed_size_dataset, indices=[1], clip_length=16,
                            test=True, streams=("joint_pos",),
                            scale_normalize=True)
        assert torch.allclose(both[1][0], large_only[0][0], atol=1e-5)

    def test_ignored_without_position_streams(self, mixed_size_dataset):
        """Rotations are already scale-free; asking for the correction on a
        rotation run must be a no-op rather than a silent rescale of the root
        vertex."""
        plain = Feeder(mixed_size_dataset, clip_length=16, test=True,
                       target_repr="quat")
        asked = Feeder(mixed_size_dataset, clip_length=16, test=True,
                       target_repr="quat", scale_normalize=True)
        assert torch.equal(plain[1][0], asked[1][0])
