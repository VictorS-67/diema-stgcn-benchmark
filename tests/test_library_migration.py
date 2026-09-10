"""Regressions for the pybvh 0.8 / pybvh-ml 0.5 migration.

Each test here pins one behaviour that the upgrade changed, so a future
version bump that silently reverts it fails loudly instead of quietly
corrupting training data.

Covered:
  * representation tokens: the retired ``"quaternion"`` spelling is rejected
    (not translated) on both the config surface and the dataset metadata
  * radians-first augmentation kwargs (``angle=`` / ``sigma=``), with the
    config's degrees declared at the call site via ``degrees=True``
  * ``MotionArrays`` as the array-level container, and ``joint_rot`` as the
    name of the rotation stream
  * skeleton up/lateral axis resolution against the dataset's own metadata
  * ``center_root`` metadata driving (or not driving) a second centering
  * per-``(seed, epoch, idx)`` augmentation draws
  * test-mode temporal sampling staying deterministic now that pybvh-ml
    honours a caller-supplied ``rng`` in test mode
  * the primitives pybvh 0.8.1 / pybvh-ml 0.5 made public after this
    migration reported them missing: ``pybvh.parse_axis``,
    ``pybvh_ml.torch.rng_for`` / ``EpochState``, ``AugmentationStep``,
    pipeline-level ``representation``, and ``target_fps`` resampling —
    each replacing a local reimplementation that must not come back
"""

import json
import math
import warnings

import numpy as np
import pytest
import torch

import pybvh_ml

from emo_mocap.cli.train import _build_pipeline, _resolve_axis, _third_axis
from emo_mocap.data.feeder import Feeder, check_repr
from emo_mocap.tools.config import load_config, load_config_with_overrides


J = 24
V = J + 1


def _write_npz(path, *, F=40, repr_token="quat", joint_rot=None,
               skel_extra=None, center_root=None, seed=0):
    """Write a one-clip pybvh-ml-format dataset with controllable metadata."""
    rng = np.random.RandomState(seed)
    root_pos = rng.randn(F, 3).astype(np.float64)
    if joint_rot is None:
        joint_rot = rng.randn(F, J, 4).astype(np.float64)
        joint_rot /= np.linalg.norm(joint_rot, axis=-1, keepdims=True)
    skel_info = {
        "num_joints": J,
        "euler_orders": ["ZYX"] * J,
        "joint_names": [f"j{i}" for i in range(J)],
        "edges": [], "lr_pairs": [],
    }
    skel_info.update(skel_extra or {})
    arrays = dict(
        clip_0_root_pos=root_pos,
        clip_0_joint_rot=joint_rot,
        labels=np.array([0], dtype=np.int64),
        filenames=np.array(["clip"]),
        num_clips=np.array(1),
        representation=np.array(repr_token),
        skeleton_info_json=np.array(json.dumps(skel_info)),
        mean=np.zeros(3 + J * joint_rot.shape[-1]),
        std=np.ones(3 + J * joint_rot.shape[-1]),
    )
    if center_root is not None:
        arrays["center_root"] = np.array(center_root)
    np.savez(path, **arrays)
    return str(path), root_pos


def _config(tmp_path, body):
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return load_config(path)


# ---------------------------------------------------------------------------
# Representation tokens
# ---------------------------------------------------------------------------

class TestRepresentationTokens:
    def test_current_tokens_pass_through(self):
        for token in ("quat", "6d", "euler", "axisangle", "rotmat"):
            assert check_repr(token, "test") == token

    def test_retired_dataset_token_is_rejected(self, tmp_path):
        """A dataset stamped "quaternion" predates pybvh 0.8 — regenerate it,
        don't translate it: its rotations came out of the old math."""
        path, _ = _write_npz(tmp_path / "stale.npz", repr_token="quaternion")
        with pytest.raises(ValueError, match="retired representation token"):
            Feeder(path, clip_length=16, test=True, target_repr="6d")

    def test_retired_target_repr_is_rejected(self, tmp_path):
        path, _ = _write_npz(tmp_path / "d.npz")
        with pytest.raises(ValueError, match="emo-preprocess"):
            Feeder(path, clip_length=16, test=True, target_repr="quaternion")

    def test_current_dataset_loads(self, tmp_path):
        path, _ = _write_npz(tmp_path / "ok.npz", repr_token="quat")
        feeder = Feeder(path, clip_length=16, test=True, target_repr="6d")
        assert feeder.source_repr == "quat"
        assert feeder[0][0].shape == (6, 16, V)

    def test_source_repr_read_from_dataset_not_assumed(self, tmp_path):
        """A 6D-preprocessed dataset must not be re-interpreted as quaternions."""
        rng = np.random.RandomState(1)
        sixd = rng.randn(30, J, 6).astype(np.float64)
        path, _ = _write_npz(tmp_path / "sixd.npz", F=30,
                             repr_token="6d", joint_rot=sixd)
        feeder = Feeder(path, clip_length=16, test=True, target_repr="6d")
        assert feeder.source_repr == "6d"
        tensor, _, _ = feeder[0]
        assert tensor.shape == (6, 16, V)
        # No conversion ran: the packed tensor is exactly the stored 6D data
        # under the deterministic test-mode temporal sampling.
        expected = pybvh_ml.pack_to_ctv(
            pybvh_ml.MotionArrays(root_pos=feeder.clips[0]["root_pos"],
                                  joint_rot=sixd),
            center_root=False)
        idx = pybvh_ml.uniform_temporal_sample(30, 16, mode="test") % 30
        assert np.allclose(tensor.numpy(), expected[:, idx, :], atol=1e-5)


# ---------------------------------------------------------------------------
# Angle units at the config boundary
# ---------------------------------------------------------------------------

class TestDegreesAtTheConfigBoundary:
    """The YAML is degrees; pybvh-ml is radians-first with a ``degrees=`` opt-in.

    emo_mocap declares the unit at the call site rather than converting with
    ``math.radians`` on the way in — one convention for both angle-taking
    steps. Dropping the flag is the failure this guards: a ``noise_sigma``
    of 1.5 would become 1.5 *radians*, ~57x the intended jitter, and nothing
    downstream would look wrong until the accuracy did.
    """

    _SKELETON = (
        "skeleton: {num_nodes: 25, inward_edges: [[0, 1]],\n"
        "           lr_joint_pairs: [], up_axis: '+z', lateral_axis: '+x'}\n"
    )
    _HEAD = (
        "data: {data_path: x.npz}\n"
        "model: {type: stgcn, num_class: 7}\n"
    )

    def test_noise_sigma_passes_through_in_degrees(self, tmp_path):
        cfg = _config(tmp_path, self._HEAD + self._SKELETON +
                      "augmentation: {enabled: true, noise_sigma: 90.0}\n")
        step = _build_pipeline(cfg).augmentations[0]
        # pybvh-ml >= 0.5 split the fused add_joint_noise; this config
        # surface only ever set the rotation sigma.
        assert step.fn is pybvh_ml.add_joint_rotation_noise
        assert "sigma_deg" not in step.kwargs
        assert step.kwargs["sigma"] == pytest.approx(90.0)
        assert step.kwargs["degrees"] is True

    def test_rotate_range_passes_through_in_degrees(self, tmp_path):
        cfg = _config(tmp_path, self._HEAD + self._SKELETON +
                      "augmentation: {enabled: true, rotate: true,\n"
                      "               rotate_range: [-180, 180]}\n")
        kwargs = _build_pipeline(cfg).augmentations[0].kwargs
        assert "angle_deg" not in kwargs
        assert kwargs["degrees"] is True
        drawn = [kwargs["angle"](np.random.default_rng(s)) for s in range(50)]
        assert min(drawn) >= -180.0 and max(drawn) <= 180.0
        # A radians-shaped range would never leave [-pi, pi].
        assert max(abs(a) for a in drawn) > math.pi

    def test_degrees_flag_reaches_the_library_through_the_pipeline(self, tmp_path):
        """End-to-end: ``degrees=True`` must survive the pipeline dispatch.

        Asserting on ``step.kwargs`` alone would still pass if the pipeline
        dropped the flag before calling the step, so compare the built
        pipeline against an explicitly-radians one on the same draws.
        """
        cfg = _config(tmp_path, self._HEAD + self._SKELETON +
                      "augmentation: {enabled: true, noise_sigma: 90.0}\n")
        rng = np.random.RandomState(11)
        quats = rng.randn(24, J, 4)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        arrays = pybvh_ml.MotionArrays(root_pos=rng.randn(24, 3),
                                       joint_rot=quats)

        radians_pipeline = pybvh_ml.AugmentationPipeline(
            [(pybvh_ml.add_joint_rotation_noise, 1.0,
              {"sigma": math.radians(90.0)})],
            representation="quat",
        )
        from_config = _build_pipeline(cfg)(arrays, rng=np.random.default_rng(3))
        explicit = radians_pipeline(arrays, rng=np.random.default_rng(3))
        assert np.allclose(from_config.joint_rot, explicit.joint_rot)

    def test_representation_declared_once_on_the_pipeline(self, tmp_path):
        """The token lives on the pipeline, not repeated on every step.

        pybvh-ml >= 0.5 takes a pipeline-level ``representation``, and a
        per-step copy is the copy-paste surface that lets one step disagree
        with the rest. Assert both halves: the pipeline declares it, and no
        step carries its own.
        """
        cfg = _five_transform_config()
        pipeline = _build_pipeline(cfg)
        assert pipeline.representation == "quat"
        assert len(pipeline.augmentations) > 1
        for step in pipeline.augmentations:
            assert "representation" not in step.kwargs

    def test_pipeline_actually_runs_on_quaternion_data(self, tmp_path):
        """End-to-end: a full five-transform pipeline must execute."""
        cfg = _five_transform_config()
        pipeline = _build_pipeline(cfg)
        rng = np.random.RandomState(0)
        root_pos = rng.randn(30, 3)
        quats = rng.randn(30, J, 4)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        out = pipeline(
            pybvh_ml.MotionArrays(root_pos=root_pos, joint_rot=quats),
            rng=np.random.default_rng(7),
        )
        # Speed perturbation resamples the time axis, so the frame count may
        # shift — but root and joints must stay in lockstep, and the joint
        # layout (J joints x 4 quaternion channels) must survive.
        assert out.root_pos.shape[0] == out.joint_rot.shape[0]
        assert out.root_pos.shape[1:] == (3,)
        assert out.joint_rot.shape[1:] == (J, 4)
        assert np.isfinite(out.joint_rot).all()


# ---------------------------------------------------------------------------
# Skeleton axis resolution
# ---------------------------------------------------------------------------

def _five_transform_config():
    """The recipe config with every augmentation switched back on.

    Several tests below exercise pybvh-ml pipeline plumbing that only shows up
    with more than one step. The shipped recipe is mirror-only by design, so
    they build the multi-step case explicitly rather than depending on a
    config that recommends something this repo does not.
    """
    return load_config_with_overrides("configs/diema7_stgcn_recipe.yaml", [
        "augmentation.rotate=true", "augmentation.speed=true",
        "augmentation.noise_sigma=1.5", "augmentation.dropout=true",
    ])


class TestAxisResolution:
    def test_diema_configs_declare_z_up(self):
        """DIEMA BVH files are Z-up. A '+y' here silently tips the performer."""
        for name in ("diema7_stgcn_recipe", "diema13_stgcn_recipe"):
            cfg = load_config(f"configs/{name}.yaml")
            assert cfg.skeleton.up_axis == "+z", name
            assert cfg.skeleton.lateral_axis == "+x", name
            assert not hasattr(cfg.skeleton, "up_idx"), name

    def test_dataset_axis_used_when_config_silent(self):
        assert _resolve_axis("up_axis", None, "+z", "+y") == "+z"

    def test_config_axis_used_when_dataset_silent(self):
        assert _resolve_axis("up_axis", "+z", None, "+y") == "+z"

    def test_disagreement_raises(self):
        with pytest.raises(ValueError, match="silently corrupts"):
            _resolve_axis("up_axis", "+y", "+z", "+y")

    def test_fallback_warns(self):
        with pytest.warns(RuntimeWarning, match="falling back"):
            assert _resolve_axis("up_axis", None, None, "+y") == "+y"

    def test_lateral_axis_derived_from_up_and_forward(self):
        assert _third_axis("+z", "+y") == "+x"
        assert _third_axis("+y", "+z") == "+x"
        assert _third_axis("-y", "+x") == "+z"
        assert _third_axis("+z", None) is None
        assert _third_axis("+z", "+z") is None

    def test_lateral_axis_rejects_a_malformed_axis(self):
        """Parsing goes through pybvh.parse_axis, so garbage raises.

        The old hand-rolled version sliced the last character, so 'bogus'
        produced a silent None that fell through to the '+x' default.
        """
        with pytest.raises(ValueError, match="Axis must be one of"):
            _third_axis("+z", "bogus")

    def test_build_pipeline_rejects_stale_config_axis(self, tmp_path):
        cfg = _config(
            tmp_path,
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]],\n"
            "           lr_joint_pairs: [], up_axis: '+y'}\n"
            "augmentation: {enabled: true, rotate: true}\n"
        )
        with pytest.raises(ValueError, match="up_axis"):
            _build_pipeline(cfg, {"world_up": "+z", "rest_forward": "+y"})

    def test_vertical_rotation_preserves_height_on_z_up_data(self):
        """The bug this whole axis machinery exists to prevent.

        A yaw about the true up axis leaves the up coordinate untouched;
        about the wrong one it swings the performer's height around.
        """
        rng = np.random.RandomState(3)
        root_pos = rng.randn(20, 3)
        quats = rng.randn(20, J, 4)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)

        arrays = pybvh_ml.MotionArrays(root_pos=root_pos, joint_rot=quats)
        right = pybvh_ml.rotate_vertical(
            arrays, angle=math.pi / 2, up_axis="+z", representation="quat")
        wrong = pybvh_ml.rotate_vertical(
            arrays, angle=math.pi / 2, up_axis="+y", representation="quat")

        assert np.allclose(right.root_pos[:, 2], root_pos[:, 2])
        assert not np.allclose(wrong.root_pos[:, 2], root_pos[:, 2])


# ---------------------------------------------------------------------------
# MotionArrays: container-level conversion, read-only fields, dtype
# ---------------------------------------------------------------------------

class TestMotionArraysContract:
    """The guarantees the Feeder's per-sample fast path relies on.

    `Feeder.__getitem__` builds a container straight over the cached clip
    arrays with no defensive copy. That is only safe because the fields are
    read-only views, so nothing downstream can write back into `self.clips`.
    If a future pybvh-ml made the fields writable again, the copy would have
    to come back — these tests are what would say so.
    """

    def test_fields_are_read_only(self):
        arrays = pybvh_ml.MotionArrays(root_pos=np.zeros((4, 3)),
                                       joint_rot=np.zeros((4, 2, 4)))
        with pytest.raises(ValueError, match="read-only"):
            arrays.root_pos[0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            arrays.joint_rot[0] = 1.0

    def test_feeder_reads_do_not_disturb_the_cached_clip(self, tmp_path):
        """A full augmented read must leave the Feeder's own clip untouched."""
        path, _ = _write_npz(tmp_path / "cache.npz", F=60)
        feeder = Feeder(path, clip_length=16, test=False, target_repr="6d",
                        augmentation_pipeline=_noise_pipeline())
        feeder.set_epoch(0)
        before_root = np.array(feeder.clips[0]["root_pos"])
        before_rot = np.array(feeder.clips[0]["joint_rot"])
        for _ in range(3):
            feeder[0]
        assert np.array_equal(feeder.clips[0]["root_pos"], before_root)
        assert np.array_equal(feeder.clips[0]["joint_rot"], before_rot)

    def test_convert_arrays_works_on_the_container(self):
        """Conversion carries the root stream instead of being unpacked.

        The Feeder used to reassemble the container around a bare-array
        conversion; `convert_arrays` is container-level, and the rotation-only
        primitive is `convert_rotations`.
        """
        rng = np.random.RandomState(5)
        q = rng.randn(12, J, 4)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        root = rng.randn(12, 3)
        arrays = pybvh_ml.MotionArrays(root_pos=root, joint_rot=q)

        out = pybvh_ml.convert_arrays(arrays, "quat", "6d")
        assert isinstance(out, pybvh_ml.MotionArrays)
        assert out.joint_rot.shape == (12, J, 6)
        # root_pos has no rotation representation, so it rides through as-is.
        assert np.array_equal(out.root_pos, root)
        # Same numbers as the rotation-level primitive.
        assert np.allclose(out.joint_rot,
                           pybvh_ml.convert_rotations(q, "quat", "6d"))

    def test_container_is_not_unpackable(self):
        """`root_pos, joint_rot = ...` was the pre-0.5 shape and must not work.

        Silently yielding two fields is the failure this prevents: pybvh-ml
        adds per-joint position streams in 0.6, and a tuple-like container
        would drop the new stream rather than announce it.
        """
        arrays = pybvh_ml.MotionArrays(root_pos=np.zeros((4, 3)),
                                       joint_rot=np.zeros((4, 2, 4)))
        with pytest.raises(TypeError, match="not iterable"):
            _a, _b = arrays


# ---------------------------------------------------------------------------
# center_root metadata
# ---------------------------------------------------------------------------

class TestCenterRootMetadata:
    def test_already_centered_data_is_not_recentered(self, tmp_path):
        path, root_pos = _write_npz(tmp_path / "c.npz", center_root=True)
        feeder = Feeder(path, clip_length=8, test=True, target_repr="quat")
        assert feeder.stored_center_root
        tensor, _, _ = feeder[0]
        # Root vertex channel 0:3 must still carry the stored (uncentered-by-us)
        # trajectory, offset only by the temporal sampling.
        assert not np.allclose(tensor[:3, 0, 0].numpy(), 0.0)

    def test_uncentered_dataset_gets_centered(self, tmp_path):
        path, _ = _write_npz(tmp_path / "u.npz", center_root=False)
        feeder = Feeder(path, clip_length=8, test=True, target_repr="quat")
        assert feeder.stored_center_root is False
        packed = pybvh_ml.pack_to_ctv(
            pybvh_ml.MotionArrays(root_pos=feeder.clips[0]["root_pos"],
                                  joint_rot=feeder.clips[0]["joint_rot"]),
            center_root=True)
        assert np.allclose(packed[:3, 0, 0], 0.0)

    def test_missing_center_root_does_not_double_center(self, tmp_path):
        """A hand-built dataset without the flag must not be re-centered."""
        path, _ = _write_npz(tmp_path / "nokey.npz")
        feeder = Feeder(path, clip_length=8, test=True, target_repr="quat")
        assert feeder.stored_center_root is None
        feeder[0]  # must not raise, must not double-center


# ---------------------------------------------------------------------------
# Seeding / epoch contract
# ---------------------------------------------------------------------------

def _noise_pipeline():
    return pybvh_ml.AugmentationPipeline([
        (pybvh_ml.add_joint_rotation_noise, 1.0,
         {"sigma": math.radians(20.0), "representation": "quat"}),
    ])


class TestSeeding:
    def _feeder(self, tmp_path, name="s.npz", seed=255):
        path, _ = _write_npz(tmp_path / name, F=60)
        return Feeder(path, clip_length=16, test=False, seed=seed,
                      augmentation_pipeline=_noise_pipeline())

    def test_same_epoch_is_reproducible(self, tmp_path):
        f1 = self._feeder(tmp_path, "a.npz")
        f2 = self._feeder(tmp_path, "a.npz")
        f1.set_epoch(3)
        f2.set_epoch(3)
        assert torch.allclose(f1[0][0], f2[0][0])

    def test_different_epochs_differ(self, tmp_path):
        feeder = self._feeder(tmp_path, "b.npz")
        feeder.set_epoch(0)
        first = feeder[0][0].clone()
        feeder.set_epoch(1)
        assert not torch.allclose(first, feeder[0][0])

    def test_draw_is_independent_of_access_order(self, tmp_path):
        """Sample 0's augmentation must not depend on what was read before it."""
        rng = np.random.RandomState(11)
        arrays = {}
        for i in range(3):
            quats = rng.randn(50, J, 4)
            quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
            arrays[f"clip_{i}_root_pos"] = rng.randn(50, 3)
            arrays[f"clip_{i}_joint_rot"] = quats
        skel = {"num_joints": J, "euler_orders": ["ZYX"] * J,
                "joint_names": [f"j{i}" for i in range(J)],
                "edges": [], "lr_pairs": []}
        path = tmp_path / "multi.npz"
        np.savez(path, num_clips=np.array(3),
                 labels=np.zeros(3, dtype=np.int64),
                 filenames=np.array(["a", "b", "c"]),
                 representation=np.array("quat"),
                 skeleton_info_json=np.array(json.dumps(skel)),
                 mean=np.zeros(3 + J * 4), std=np.ones(3 + J * 4),
                 **arrays)

        def read(order):
            f = Feeder(str(path), clip_length=16, test=False, seed=7,
                       augmentation_pipeline=_noise_pipeline())
            f.set_epoch(2)
            return {i: f[i][0] for i in order}

        forward = read([0, 1, 2])
        backward = read([2, 1, 0])
        for i in range(3):
            assert torch.allclose(forward[i], backward[i])

    def test_set_epoch_rejects_negative(self, tmp_path):
        feeder = self._feeder(tmp_path, "neg.npz")
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            feeder.set_epoch(-1)

    def test_missing_set_epoch_warns_once(self, tmp_path):
        feeder = self._feeder(tmp_path, "warn.npz")
        with pytest.warns(RuntimeWarning, match="set_epoch"):
            feeder[0]
        # Second read is silent — one warning per process, not per sample.
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            feeder[0]
        assert not [c for c in caught if "set_epoch" in str(c.message)]

    def test_test_mode_temporal_sampling_stays_deterministic(self, tmp_path):
        """pybvh-ml 0.5 honours a caller rng in test mode; we must pass None."""
        path, _ = _write_npz(tmp_path / "t.npz", F=200)
        feeder = Feeder(path, clip_length=16, test=True, target_repr="quat")
        assert torch.allclose(feeder[0][0], feeder[0][0])

    def test_negative_index_matches_positive(self, tmp_path):
        path, _ = _write_npz(tmp_path / "neg2.npz")
        feeder = Feeder(path, clip_length=8, test=True, target_repr="quat")
        assert torch.allclose(feeder[-1][0], feeder[len(feeder) - 1][0])
        with pytest.raises(IndexError):
            feeder[5]


class TestEpochSeedCallback:
    """The callback is the only thing telling the Feeder the epoch changed."""

    def test_callback_forwards_the_epoch(self, tmp_path):
        from types import SimpleNamespace
        from emo_mocap.data.loader import EpochSeedCallback

        path, _ = _write_npz(tmp_path / "cb.npz")
        feeder = Feeder(path, clip_length=8, test=False, seed=1,
                        augmentation_pipeline=_noise_pipeline())
        trainer = SimpleNamespace(
            datamodule=SimpleNamespace(dataset_train=feeder), current_epoch=4,
        )
        EpochSeedCallback().on_train_epoch_start(trainer, None)
        assert feeder._epoch_state.current == 4

    def test_callback_tolerates_a_datamodule_without_a_train_set(self):
        from types import SimpleNamespace
        from emo_mocap.data.loader import EpochSeedCallback

        trainer = SimpleNamespace(datamodule=SimpleNamespace(), current_epoch=0)
        EpochSeedCallback().on_train_epoch_start(trainer, None)  # no raise

    def test_train_cli_registers_the_callback(self):
        """A refactor that drops it silently freezes augmentation at epoch 0."""
        import inspect
        from emo_mocap.cli import train as train_cli

        source = inspect.getsource(train_cli.main)
        assert "EpochSeedCallback()" in source

    def test_setup_claims_epoch_zero_before_workers_fork(self, tmp_path):
        """Lightning prefetches training batches before any per-epoch hook.

        ``on_train_epoch_start`` fires *after* the DataLoader has been built
        and its workers forked, so without this the first batches of epoch 0
        are drawn from a Feeder in the never-set state and every worker emits
        the missing-set_epoch warning on every run. ``Callback.setup`` runs
        before dataloader construction, which is early enough.
        """
        from types import SimpleNamespace
        from emo_mocap.data.loader import EpochSeedCallback

        path, _ = _write_npz(tmp_path / "cb_setup.npz")
        feeder = Feeder(path, clip_length=8, test=False, seed=1,
                        augmentation_pipeline=_noise_pipeline())
        assert feeder._epoch_state._raw() == -1, "precondition: never set"

        trainer = SimpleNamespace(
            datamodule=SimpleNamespace(dataset_train=feeder), current_epoch=0,
        )
        EpochSeedCallback().setup(trainer, None, "fit")
        assert feeder._epoch_state._raw() == 0

        # ...and having been set, drawing a sample must not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            feeder[0]

    def test_setup_ignores_non_fit_stages(self, tmp_path):
        """`test`/`predict` have no train set to advance; don't touch it."""
        from types import SimpleNamespace
        from emo_mocap.data.loader import EpochSeedCallback

        path, _ = _write_npz(tmp_path / "cb_stage.npz")
        feeder = Feeder(path, clip_length=8, test=False, seed=1,
                        augmentation_pipeline=_noise_pipeline())
        trainer = SimpleNamespace(
            datamodule=SimpleNamespace(dataset_train=feeder), current_epoch=0,
        )
        EpochSeedCallback().setup(trainer, None, "test")
        assert feeder._epoch_state._raw() == -1


class TestGlobalSeeding:
    """data.seed must reach every RNG, not just the Feeder's augmentation."""

    def test_train_cli_seeds_globals(self):
        """Without this, weight init and shuffle order ignore data.seed, so a
        multi-seed sweep is not reproducible and deterministic=True pins nothing."""
        import inspect
        from emo_mocap.cli import train as train_cli

        source = inspect.getsource(train_cli.main)
        assert "seed_everything(cfg.data.seed" in source

    def test_compute_line_reports_completed_epochs(self):
        """trainer.current_epoch is already the completed count after fit."""
        import inspect
        from emo_mocap.cli import train as train_cli

        source = inspect.getsource(train_cli.main)
        assert "epochs={epochs_run}" in source
        assert "trainer.current_epoch + 1" not in source


class TestUpstreamPrimitivesNotReimplemented:
    """Each of these replaced a local copy of something pybvh-ml now exports.

    The migration that produced ``upstream_feedback.md`` had to hand-roll all
    of them; 0.8.1 / 0.5 made them public. A test per primitive so a later
    refactor can't quietly grow the copy back.
    """

    def test_feeder_uses_the_library_seeding_scheme(self, tmp_path):
        """``(seed, epoch, idx)`` draws must be pybvh-ml's, not a local copy."""
        from pybvh_ml.torch import rng_for

        path, _ = _write_npz(tmp_path / "seed.npz")
        feeder = Feeder(path, clip_length=8, test=False, seed=7)
        feeder.set_epoch(3)
        mine = feeder._rng_for(2).standard_normal(5)
        theirs = rng_for(7, 3, 2).standard_normal(5)
        np.testing.assert_array_equal(mine, theirs)

    def test_epoch_state_is_the_library_class(self, tmp_path):
        """A plain attribute here means persistent workers replay epoch 0."""
        from pybvh_ml.torch import EpochState

        path, _ = _write_npz(tmp_path / "epoch.npz")
        feeder = Feeder(path, clip_length=8, test=False)
        assert isinstance(feeder._epoch_state, EpochState)

    def test_missing_set_epoch_still_warns(self, tmp_path):
        """The warning survives the switch to EpochState.

        ``EpochState.current`` reports 0 both for "never set" and for
        "epoch 0", so the Feeder tracks the distinction itself. If that
        bookkeeping is lost the warning goes silent and every epoch replays
        the same augmentation with nothing to say so.
        """
        path, _ = _write_npz(tmp_path / "warn.npz", F=60)
        feeder = Feeder(path, clip_length=16, test=False,
                        augmentation_pipeline=_noise_pipeline())
        with pytest.warns(RuntimeWarning, match="set_epoch"):
            feeder[0]

    def test_steps_are_named_tuples(self):
        """``step.kwargs`` reads better than ``step[2]`` and can't drift."""
        cfg = _five_transform_config()
        step = _build_pipeline(cfg).augmentations[0]
        assert isinstance(step, pybvh_ml.AugmentationStep)
        # A NamedTuple is still a tuple, so old positional code keeps working.
        assert tuple(step) == (step.fn, step.prob, step.kwargs)

    def test_preprocess_forwards_target_fps(self, tmp_path, monkeypatch):
        """Resampling happens inside pybvh-ml, before rotations are extracted.

        The CLI used to slice the finished .npz, which cannot produce correct
        velocities: a finite difference's stencil baseline is the *original*
        frame_time. Assert the kwarg reaches the library and that no
        post-processing pass survives.
        """
        from emo_mocap.cli import preprocess as preprocess_cli

        (tmp_path / "JP_06_anger_1_H.bvh").write_text("HIERARCHY\n")
        emo2idx = tmp_path / "emo.txt"
        emo2idx.write_text("anger 0\n")
        out = tmp_path / "ds.npz"

        seen = {}

        def fake_preprocess_directory(**kwargs):
            seen.update(kwargs)
            return {"num_clips": 1, "representation": "quat",
                    "skeleton_info": {}, "uniformity": {"fps": {120.0: ["x"]}}}

        monkeypatch.setattr(preprocess_cli.pybvh_ml, "preprocess_directory",
                            fake_preprocess_directory)
        monkeypatch.setattr("sys.argv", [
            "emo-preprocess", "--input", str(tmp_path), "--output", str(out),
            "--emo2idx", str(emo2idx), "--target-fps", "30",
        ])
        preprocess_cli.main()

        assert seen["target_fps"] == 30.0
        assert not hasattr(preprocess_cli, "_decimate_npz")
