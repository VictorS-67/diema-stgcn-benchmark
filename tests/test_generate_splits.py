"""Tests for the generate_splits CLI (optional export tool)."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from emo_mocap.cli.generate_splits import main


def _make_mock_npz(path, actor_clips, J=24, seed=42):
    """Create a mock npz with DIEMA-format filenames for specific actors.

    Args:
        path: output npz path
        actor_clips: dict of {actor_id: num_clips}, e.g. {"JP_01": 3}
    """
    rng = np.random.RandomState(seed)
    arrays = {}
    filenames = []
    labels = []
    clip_idx = 0

    for actor, n in actor_clips.items():
        nat, pid = actor.split("_")
        for i in range(n):
            F = rng.randint(20, 100)
            arrays[f"clip_{clip_idx}_root_pos"] = rng.randn(F, 3).astype(np.float64)
            quats = rng.randn(F, J, 4).astype(np.float64)
            quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
            arrays[f"clip_{clip_idx}_joint_rot"] = quats
            filenames.append(f"{nat}_{pid}_anger_{i}_H")
            labels.append(0)
            clip_idx += 1

    arrays["labels"] = np.array(labels, dtype=np.int64)
    arrays["filenames"] = np.array(filenames)
    arrays["num_clips"] = np.array(clip_idx)
    arrays["representation"] = np.array("quat")

    skel_info = {
        "num_joints": J,
        "euler_orders": ["ZYX"] * J,
        "joint_names": [f"joint_{i}" for i in range(J)],
        "edges": [[i + 1, i] for i in range(J - 1)],
        "lr_pairs": [],
    }
    arrays["skeleton_info_json"] = np.array(json.dumps(skel_info))
    arrays["mean"] = np.zeros(3 + J * 4, dtype=np.float64)
    arrays["std"] = np.ones(3 + J * 4, dtype=np.float64)

    np.savez(path, **arrays)


class TestGenerateSplitsCLI:
    def test_creates_fold_files(self, tmp_path, monkeypatch):
        """CLI creates the expected number of fold pkl files."""
        npz_path = tmp_path / "data.npz"
        output_dir = tmp_path / "splits"
        _make_mock_npz(npz_path, {"JP_01": 3, "JP_02": 4, "TW_01": 5, "TW_02": 2})

        monkeypatch.setattr(sys, "argv", [
            "emo-generate-splits",
            "--data-path", str(npz_path),
            "--output-dir", str(output_dir),
            "--num-folds", "3",   # rotating-val LPO requires >= 3
        ])
        main()

        fold_files = sorted(output_dir.glob("fold_*.pkl"))
        assert len(fold_files) == 3
        assert fold_files[0].name == "fold_01.pkl"
        assert fold_files[-1].name == "fold_03.pkl"

    def test_pkl_structure(self, tmp_path, monkeypatch):
        """Each pkl has the correct keys and format for Loader (val/test disjoint)."""
        npz_path = tmp_path / "data.npz"
        output_dir = tmp_path / "splits"
        _make_mock_npz(npz_path, {"JP_01": 3, "JP_02": 4, "JP_03": 5})

        monkeypatch.setattr(sys, "argv", [
            "emo-generate-splits",
            "--data-path", str(npz_path),
            "--output-dir", str(output_dir),
            "--num-folds", "3",
        ])
        main()

        with open(output_dir / "fold_01.pkl", "rb") as f:
            split = pickle.load(f)

        assert set(split.keys()) == {"train", "val", "test"}
        # New contract: val and test are disjoint.
        val_names = {f for f, _ in split["val"]}
        test_names = {f for f, _ in split["test"]}
        assert val_names.isdisjoint(test_names)
        # Every clip (3 + 4 + 5 = 12) must fall into exactly one of the three splits.
        total = len(split["train"]) + len(split["val"]) + len(split["test"])
        assert total == 12

        # Entries are (str, int) tuples
        for fname, idx in split["train"]:
            assert isinstance(fname, str)
            assert isinstance(idx, int)
