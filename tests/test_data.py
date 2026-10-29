"""Data pipeline tests.

Uses synthetic mock data in pybvh-ml npz format (no real dataset files needed).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers to create mock npz data (pybvh-ml format)
# ---------------------------------------------------------------------------

def _make_mock_npz(path, n_samples=10, J=24, min_frames=20, max_frames=200,
                   num_classes=7, seed=42):
    """Create a mock .npz file mimicking pybvh-ml's preprocess_directory output.

    Stores quaternion data: root_pos (F,3) + joint_rot (F,J,4) per clip.
    Must match the format that pybvh_ml.load_preprocessed() expects.
    """
    rng = np.random.RandomState(seed)
    arrays = {}
    filenames = []
    labels = []

    for i in range(n_samples):
        F = rng.randint(min_frames, max_frames + 1)
        arrays[f"clip_{i}_root_pos"] = rng.randn(F, 3).astype(np.float64)
        # Quaternions: random unit quaternions
        quats = rng.randn(F, J, 4).astype(np.float64)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        arrays[f"clip_{i}_joint_rot"] = quats
        filenames.append(f"test_sample_{i}")
        labels.append(rng.randint(0, num_classes))

    arrays["labels"] = np.array(labels, dtype=np.int64)
    arrays["filenames"] = np.array(filenames)
    arrays["num_clips"] = np.array(n_samples)
    arrays["representation"] = np.array("quat")

    # Skeleton info as JSON (matches pybvh-ml's _save_npz format)
    skel_info = {
        "num_joints": J,
        "euler_orders": ["ZYX"] * J,
        "joint_names": [f"joint_{i}" for i in range(J)],
        "edges": [[i + 1, i] for i in range(J - 1)],
        "lr_pairs": [],
    }
    arrays["skeleton_info_json"] = np.array(json.dumps(skel_info))

    # Normalization stats (dummy)
    arrays["mean"] = np.zeros(3 + J * 4, dtype=np.float64)
    arrays["std"] = np.ones(3 + J * 4, dtype=np.float64)

    np.savez(path, **arrays)
    return filenames, labels


def _make_mock_split(n_samples, train_frac=0.6, val_frac=0.2):
    """Create a split dict from sample count."""
    n_train = max(1, int(n_samples * train_frac))
    n_val = max(1, int(n_samples * val_frac))
    filenames = [f"test_sample_{i}" for i in range(n_samples)]

    split = {
        "train": [(filenames[i], i) for i in range(n_train)],
        "val": [(filenames[i], i) for i in range(n_train, n_train + n_val)],
        "test": [(filenames[i], i) for i in range(n_train + n_val, n_samples)],
    }
    return split


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create temporary mock npz data and split pickle, return paths."""
    n_samples = 10
    data_path = tmp_path / "mock_data.npz"
    split_path = tmp_path / "mock_split_dict.pkl"

    filenames, labels = _make_mock_npz(data_path, n_samples=n_samples)
    split = _make_mock_split(n_samples)

    with open(split_path, "wb") as f:
        pickle.dump(split, f)

    return {
        "data_path": str(data_path),
        "split_path": str(split_path),
        "n_samples": n_samples,
        "split": split,
        "filenames": filenames,
        "labels": labels,
    }


from emo_mocap.data.feeder import Feeder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeederBasic:
    """Basic Feeder functionality."""

    def test_correct_sample_count_no_filter(self, mock_data_dir):
        feeder = Feeder(mock_data_dir["data_path"], clip_length=64, test=True)
        assert len(feeder) == 10

    def test_correct_sample_count_with_indices(self, mock_data_dir):
        indices = [0, 2, 4]
        feeder = Feeder(mock_data_dir["data_path"], indices=indices, clip_length=64, test=True)
        assert len(feeder) == 3

    def test_output_tensor_shape_euler(self, mock_data_dir):
        feeder = Feeder(mock_data_dir["data_path"], clip_length=64,
                        target_repr="euler", test=True,
                        euler_orders=["ZYX"] * 24)
        tensor, label, filename = feeder[0]
        # Euler: C=3, clip_length=64, V=1(root)+24(joints)=25
        assert tensor.shape == (3, 64, 25)

    def test_output_tensor_shape_quaternion(self, mock_data_dir):
        feeder = Feeder(mock_data_dir["data_path"], clip_length=64,
                        target_repr="quat", test=True)
        tensor, label, filename = feeder[0]
        # Quaternion: C=4, clip_length=64, V=25
        assert tensor.shape == (4, 64, 25)

    def test_output_types(self, mock_data_dir):
        feeder = Feeder(mock_data_dir["data_path"], clip_length=64, test=True,
                        target_repr="quat")
        tensor, label, filename = feeder[0]
        assert tensor.dtype == torch.float32
        assert label.dtype == torch.long
        assert isinstance(filename, str)

    def test_labels_in_range(self, mock_data_dir):
        feeder = Feeder(mock_data_dir["data_path"], clip_length=64, test=True,
                        target_repr="quat")
        for i in range(len(feeder)):
            _, label, _ = feeder[i]
            assert 0 <= label.item() < 7

    def test_filenames_match(self, mock_data_dir):
        feeder = Feeder(mock_data_dir["data_path"], clip_length=64, test=True,
                        target_repr="quat")
        _, _, filename = feeder[0]
        assert filename == "test_sample_0"


class TestFeederEdgeCases:
    """Edge cases: very short, exact-length, and very long sequences."""

    def _make_single_clip_npz(self, tmp_path, name, F, J=24):
        """Helper to create a single-clip npz file in pybvh-ml format."""
        path = tmp_path / f"{name}.npz"
        rng = np.random.RandomState(42)
        root_pos = rng.randn(F, 3).astype(np.float64)
        quats = rng.randn(F, J, 4).astype(np.float64)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        skel_info = {"num_joints": J, "euler_orders": ["ZYX"] * J,
                     "joint_names": [f"j{i}" for i in range(J)],
                     "edges": [], "lr_pairs": []}
        np.savez(path,
                 clip_0_root_pos=root_pos,
                 clip_0_joint_rot=quats,
                 labels=np.array([0], dtype=np.int64),
                 filenames=np.array(["test"]),
                 num_clips=np.array(1),
                 representation=np.array("quat"),
                 skeleton_info_json=np.array(json.dumps(skel_info)),
                 mean=np.zeros(3 + J * 4), std=np.ones(3 + J * 4))
        return str(path)

    def test_very_short_sequence(self, tmp_path):
        """3 frames with clip_length=64 should not crash (wraparound)."""
        path = self._make_single_clip_npz(tmp_path, "short", F=3)
        feeder = Feeder(path, clip_length=64, test=True, target_repr="quat")
        tensor, label, _ = feeder[0]
        assert tensor.shape == (4, 64, 25)
        assert not torch.isnan(tensor).any()

    def test_exact_length_sequence(self, tmp_path):
        """Sequence exactly equal to clip_length."""
        path = self._make_single_clip_npz(tmp_path, "exact", F=64)
        feeder = Feeder(path, clip_length=64, test=True, target_repr="quat")
        tensor, label, _ = feeder[0]
        assert tensor.shape == (4, 64, 25)

    def test_very_long_sequence(self, tmp_path):
        """1000 frames with clip_length=64."""
        path = self._make_single_clip_npz(tmp_path, "long", F=1000)
        feeder = Feeder(path, clip_length=64, test=True, target_repr="quat")
        tensor, label, _ = feeder[0]
        assert tensor.shape == (4, 64, 25)
        assert not torch.isnan(tensor).any()

    def test_single_entry(self, tmp_path):
        """Single sample in the dataset."""
        path = self._make_single_clip_npz(tmp_path, "single", F=100)
        feeder = Feeder(path, clip_length=64, test=True, target_repr="quat")
        assert len(feeder) == 1
        tensor, label, _ = feeder[0]
        assert tensor.shape == (4, 64, 25)


class TestSplitIntegrity:
    """Verify split dict properties."""

    def test_no_filename_overlap(self, mock_data_dir):
        split = mock_data_dir["split"]
        train_names = {name for name, _ in split["train"]}
        val_names = {name for name, _ in split["val"]}
        test_names = {name for name, _ in split["test"]}

        assert train_names.isdisjoint(val_names), "Train/val overlap"
        assert train_names.isdisjoint(test_names), "Train/test overlap"
        assert val_names.isdisjoint(test_names), "Val/test overlap"

    def test_all_ids_valid(self, mock_data_dir):
        split = mock_data_dir["split"]
        n = mock_data_dir["n_samples"]
        for partition in ("train", "val", "test"):
            for _, idx in split[partition]:
                assert 0 <= idx < n, f"Invalid index {idx} in {partition} split"

    def test_splits_cover_all_data(self, mock_data_dir):
        split = mock_data_dir["split"]
        all_ids = set()
        for partition in ("train", "val", "test"):
            for _, idx in split[partition]:
                all_ids.add(idx)
        assert all_ids == set(range(mock_data_dir["n_samples"]))


class TestFeederWithSplitIndices:
    """Test Feeder when filtered by split indices."""

    def test_train_split_size(self, mock_data_dir):
        split = mock_data_dir["split"]
        train_indices = [idx for _, idx in split["train"]]
        feeder = Feeder(mock_data_dir["data_path"], indices=train_indices,
                        clip_length=64, target_repr="quat")
        assert len(feeder) == len(split["train"])

    def test_val_split_deterministic(self, mock_data_dir):
        split = mock_data_dir["split"]
        val_indices = [idx for _, idx in split["val"]]
        feeder = Feeder(
            mock_data_dir["data_path"], indices=val_indices,
            clip_length=64, test=True, target_repr="quat",
        )
        t1, _, _ = feeder[0]
        t2, _, _ = feeder[0]
        assert torch.allclose(t1, t2), "Test mode should be deterministic"


class TestLabelsFile:
    """``labels_path``: a CSV of ``filename,label`` that replaces the labels
    stored in the dataset, so one npz serves any per-clip target (intensity,
    a property of the performer, ...) without being rebuilt."""

    def _write(self, path, rows):
        path.write_text("filename,label\n" + "".join(f"{f},{l}\n" for f, l in rows))
        return path

    def test_replaces_the_stored_labels_by_filename(self, mock_data_dir, tmp_path):
        # Rows deliberately in reverse file order: the match is by name.
        names = mock_data_dir["filenames"]
        path = self._write(tmp_path / "l.csv",
                           [(f, i % 2) for i, f in reversed(list(enumerate(names)))])
        feeder = Feeder(mock_data_dir["data_path"], labels_path=path,
                        clip_length=16, test=True, target_repr="quat")
        assert [feeder[i][1].item() for i in range(len(names))] == [0, 1] * 5

    def test_follows_the_index_subset(self, mock_data_dir, tmp_path):
        names = mock_data_dir["filenames"]
        path = self._write(tmp_path / "l.csv", [(f, i) for i, f in enumerate(names)])
        feeder = Feeder(mock_data_dir["data_path"], labels_path=path, indices=[7, 2],
                        clip_length=16, test=True, target_repr="quat")
        assert [feeder[0][1].item(), feeder[1][1].item()] == [7, 2]

    def test_a_clip_missing_from_the_file_raises(self, mock_data_dir, tmp_path):
        names = mock_data_dir["filenames"]
        path = self._write(tmp_path / "l.csv", [(f, 0) for f in names[1:]])
        with pytest.raises(ValueError, match="test_sample_0"):
            Feeder(mock_data_dir["data_path"], labels_path=path,
                   clip_length=16, test=True, target_repr="quat")

    def test_the_loader_passes_it_to_every_split(self, mock_data_dir, tmp_path):
        from emo_mocap.data.loader import Loader
        names = mock_data_dir["filenames"]
        path = self._write(tmp_path / "l.csv", [(f, 1) for f in names])
        loader = Loader(mock_data_dir["data_path"], split_dict=mock_data_dir["split"],
                        labels_path=path, num_workers=0, target_repr="quat")
        loader.setup("fit")
        loader.setup("test")
        for ds in (loader.dataset_train, loader.dataset_val, loader.dataset_test):
            assert set(ds.labels.tolist()) == {1}

    def test_the_config_default_keeps_the_stored_labels(self, tmp_path):
        from emo_mocap.tools.config import load_config
        path = tmp_path / "c.yaml"
        path.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7, in_channels: 3}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]]}\n"
        )
        assert load_config(path).data.labels_path is None
