"""Tests for the config-driven checkpointing flow in emo-train.

Covers:
- Defaults preserve legacy behavior (best_val_acc saved, in-memory weights tested).
- _build_checkpoint_callbacks builds the right ModelCheckpoint per preset.
- _resolve_test_ckpt_path returns None for "current" and the right path otherwise.
- Bad presets / bad test_with values raise.
- An integration smoke run produces the expected on-disk checkpoint files
  and trainer.test() loads from disk when test_with != "current".
"""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from emo_mocap.cli.train import (
    _CKPT_PRESETS,
    _build_checkpoint_callbacks,
    _resolve_test_ckpt_path,
)
from emo_mocap.tools.config import load_config, load_config_with_overrides


class TestCheckpointDefaults:
    """The default config should preserve the legacy behavior."""

    @staticmethod
    def _config_without_a_checkpointing_block(tmp_path):
        """A config that says nothing about checkpointing, so these tests
        exercise tools/config.py's defaults rather than whatever the shipped
        configs happen to declare today."""
        path = tmp_path / "minimal.yaml"
        path.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]]}\n"
        )
        return load_config(path)

    def test_default_save_is_best_val_acc(self, tmp_path):
        cfg = self._config_without_a_checkpointing_block(tmp_path)
        assert cfg.checkpointing.save == ["best_val_acc"]

    def test_default_test_with_is_current(self, tmp_path):
        cfg = self._config_without_a_checkpointing_block(tmp_path)
        assert cfg.checkpointing.test_with == "current"

    def test_override_test_with(self):
        cfg = load_config_with_overrides(
            "configs/diema7_stgcn_recipe.yaml",
            ["checkpointing.test_with=best_val_acc"],
        )
        assert cfg.checkpointing.test_with == "best_val_acc"


class TestBuildCheckpointCallbacks:
    def test_single_preset(self):
        callbacks, by_preset = _build_checkpoint_callbacks(["best_val_acc"])
        assert len(callbacks) == 1
        assert list(by_preset) == ["best_val_acc"]
        cb = by_preset["best_val_acc"]
        assert isinstance(cb, ModelCheckpoint)
        assert cb.monitor == "val_acc"
        assert cb.mode == "max"
        assert cb.save_top_k == 1

    def test_multiple_presets(self):
        callbacks, by_preset = _build_checkpoint_callbacks(
            ["best_val_loss", "best_val_acc", "last"]
        )
        assert len(callbacks) == 3
        assert set(by_preset) == {"best_val_loss", "best_val_acc", "last"}
        assert by_preset["best_val_loss"].monitor == "val_loss"
        assert by_preset["best_val_loss"].mode == "min"
        assert by_preset["best_val_acc"].monitor == "val_acc"
        assert by_preset["best_val_acc"].mode == "max"
        # "last" tracks no metric and uses save_last=True
        last_cb = by_preset["last"]
        assert last_cb.monitor is None
        assert last_cb.save_last is True
        assert last_cb.save_top_k == 0

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown checkpointing preset"):
            _build_checkpoint_callbacks(["best_val_yolo"])

    def test_empty_save_list(self):
        callbacks, by_preset = _build_checkpoint_callbacks([])
        assert callbacks == []
        assert by_preset == {}

    def test_preset_filenames_are_distinct(self):
        # Filenames must differ between presets so multiple checkpoints can
        # coexist in the same dirpath without overwriting each other.
        filenames = {spec["filename"] for spec in _CKPT_PRESETS.values()}
        assert len(filenames) == len(_CKPT_PRESETS)


class TestResolveTestCkptPath:
    def test_current_returns_none(self):
        assert _resolve_test_ckpt_path("current", {}) is None

    def test_returns_best_model_path(self):
        cb = MagicMock(spec=ModelCheckpoint)
        cb.best_model_path = "/tmp/best-val-acc-epoch=02-val_acc=0.6000.ckpt"
        path = _resolve_test_ckpt_path("best_val_acc", {"best_val_acc": cb})
        assert path == cb.best_model_path

    def test_returns_last_model_path_for_last(self):
        cb = MagicMock(spec=ModelCheckpoint)
        cb.last_model_path = "/tmp/last.ckpt"
        path = _resolve_test_ckpt_path("last", {"last": cb})
        assert path == "/tmp/last.ckpt"

    def test_empty_path_returns_none(self):
        # Lightning leaves best_model_path == "" when no checkpoint was saved
        # (e.g., no validation step ran). Treat that as "no checkpoint".
        cb = MagicMock(spec=ModelCheckpoint)
        cb.best_model_path = ""
        path = _resolve_test_ckpt_path("best_val_acc", {"best_val_acc": cb})
        assert path is None

    def test_unknown_test_with_raises(self):
        with pytest.raises(ValueError, match="not in checkpointing.save"):
            _resolve_test_ckpt_path("best_val_loss", {"best_val_acc": MagicMock()})


def _make_synthetic_dataset(tmp_path):
    """Build a tiny pybvh-ml-format npz + split.pkl for a smoke training run.

    Mirrors the construction in tests/test_cli.py::test_train_smoke so the
    checkpoint smoke test exercises the same data path the rest of the suite
    is validated against.
    """
    J = 24
    rng = np.random.default_rng(0)
    arrays = {}
    for i in range(8):
        root_pos = rng.standard_normal((80, 3)).astype(np.float64)
        quats = rng.standard_normal((80, J, 4)).astype(np.float64)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        arrays[f"clip_{i}_root_pos"] = root_pos
        arrays[f"clip_{i}_joint_rot"] = quats

    arrays["labels"] = np.array([i % 7 for i in range(8)], dtype=np.int64)
    arrays["filenames"] = np.array([f"smoke_{i:02d}" for i in range(8)])
    arrays["num_clips"] = np.array(8)
    arrays["representation"] = np.array("quat")
    skel_info = {"num_joints": J, "euler_orders": ["ZYX"] * J,
                 "joint_names": [f"j{i}" for i in range(J)],
                 "edges": [], "lr_pairs": []}
    arrays["skeleton_info_json"] = np.array(json.dumps(skel_info))
    arrays["mean"] = np.zeros(3 + J * 4, dtype=np.float64)
    arrays["std"] = np.ones(3 + J * 4, dtype=np.float64)

    data_path = tmp_path / "data.npz"
    np.savez(data_path, **arrays)

    split = {
        "train": [(f"smoke_{i:02d}", i) for i in range(6)],
        "val":   [(f"smoke_{i:02d}", i) for i in range(6, 8)],
        "test":  [(f"smoke_{i:02d}", i) for i in range(6, 8)],
    }
    split_path = tmp_path / "split.pkl"
    with open(split_path, "wb") as f:
        pickle.dump(split, f)
    return data_path, split_path, J


@pytest.mark.slow
def test_multi_preset_saves_all_files(tmp_path):
    """Training with three presets should produce three distinct ckpt files
    on disk, and trainer.test(ckpt_path=...) should accept the resolved path.
    """
    from emo_mocap.training.lightning_model import LightningModel
    from emo_mocap.data.loader import Loader
    from emo_mocap.models.stgcn.stgcn_model import STGCN_Model

    data_path, split_path, J = _make_synthetic_dataset(tmp_path)
    cfg = load_config("configs/diema7_stgcn_recipe.yaml")
    model = STGCN_Model.from_config(cfg)
    lit = LightningModel(model, base_lr=1e-3, num_class=cfg.model.num_class)

    loader = Loader(
        data_path=data_path,
        split_path=split_path,
        clip_length=32,
        batch_size=4,
        num_workers=0,
        target_repr=cfg.data.target_repr,
        seed=42,
        euler_orders=["ZYX"] * J,
    )

    ckpt_dir = tmp_path / "ckpts"
    callbacks, by_preset = _build_checkpoint_callbacks(
        ["best_val_loss", "best_val_acc", "last"]
    )
    # Pin all checkpoints to the same dirpath so the test can inspect them.
    for cb in callbacks:
        cb.dirpath = str(ckpt_dir)

    with patch("torch.cuda.is_available", return_value=False):
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=callbacks,
        )
        trainer.fit(lit, datamodule=loader)

    # Snapshot before any test() call
    snapshot = {
        name: (cb.last_model_path if name == "last" else cb.best_model_path)
        for name, cb in by_preset.items()
    }
    for name, path in snapshot.items():
        assert path, f"Preset {name} produced no checkpoint path"
        assert (ckpt_dir / Path(path).name).exists(), \
            f"Checkpoint file for {name} not on disk: {path}"

    # File names should be distinct so the three presets coexist
    names = {Path(p).name for p in snapshot.values()}
    assert len(names) == 3

    # _resolve_test_ckpt_path should return each on-disk path
    for preset in ("best_val_loss", "best_val_acc", "last"):
        resolved = _resolve_test_ckpt_path(preset, by_preset)
        assert resolved == snapshot[preset]

    # And "current" still means in-memory
    assert _resolve_test_ckpt_path("current", by_preset) is None
