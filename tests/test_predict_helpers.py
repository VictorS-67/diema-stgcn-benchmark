"""Tests for the checkpoint-discovery helper and predict.py CSV output format.

These are lightweight: the helpers operate on directory layouts and CSV rows,
so we fake the inputs rather than running a real training job.
"""

import csv
import io

import numpy as np
import pytest
import torch

from emo_mocap.cli.predict import _save_interpretability, _write_predictions
from emo_mocap.tools.checkpoints import find_best_checkpoint


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


class TestFindBestCheckpoint:
    def test_picks_latest_version(self, tmp_path):
        log_dir = tmp_path / "logs"
        exp = "stgcn_fold03"
        # Two versions exist; the helper should pick version_1 over version_0.
        _touch(log_dir / exp / "version_0" / "checkpoints" / "best-val-acc-epoch=02-val_acc=0.5000.ckpt")
        _touch(log_dir / exp / "version_1" / "checkpoints" / "best-val-acc-epoch=04-val_acc=0.6000.ckpt")
        found = find_best_checkpoint(log_dir, exp, "best_val_acc")
        assert "version_1" in str(found)
        assert found.name.endswith(".ckpt")

    def test_falls_back_to_older_version_if_latest_empty(self, tmp_path):
        log_dir = tmp_path / "logs"
        exp = "stgcn_fold03"
        _touch(log_dir / exp / "version_0" / "checkpoints" / "best-val-acc-epoch=02-val_acc=0.5000.ckpt")
        # version_1 has no checkpoints dir
        (log_dir / exp / "version_1").mkdir(parents=True)
        found = find_best_checkpoint(log_dir, exp, "best_val_acc")
        assert "version_0" in str(found)

    def test_different_preset(self, tmp_path):
        log_dir = tmp_path / "logs"
        exp = "stgcn_fold03"
        _touch(log_dir / exp / "version_0" / "checkpoints" / "best-val-loss-epoch=01-val_loss=1.2000.ckpt")
        _touch(log_dir / exp / "version_0" / "checkpoints" / "best-val-acc-epoch=02-val_acc=0.5000.ckpt")
        loss_ckpt = find_best_checkpoint(log_dir, exp, "best_val_loss")
        acc_ckpt = find_best_checkpoint(log_dir, exp, "best_val_acc")
        assert "val-loss" in loss_ckpt.name
        assert "val-acc" in acc_ckpt.name

    def test_last_preset(self, tmp_path):
        log_dir = tmp_path / "logs"
        exp = "stgcn_fold03"
        _touch(log_dir / exp / "version_0" / "checkpoints" / "last.ckpt")
        found = find_best_checkpoint(log_dir, exp, "last")
        assert found.name == "last.ckpt"

    def test_missing_experiment_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Experiment directory"):
            find_best_checkpoint(tmp_path, "nonexistent_fold01")

    def test_no_matching_ckpt_raises(self, tmp_path):
        log_dir = tmp_path / "logs"
        exp = "stgcn_fold03"
        _touch(log_dir / exp / "version_0" / "checkpoints" / "best-val-loss-epoch=01-val_loss=1.2000.ckpt")
        with pytest.raises(FileNotFoundError, match="best-val-acc"):
            find_best_checkpoint(log_dir, exp, "best_val_acc")


def _make_batch(num_class, batch_size, start_idx):
    """Build one predict_step output tuple."""
    predicted = torch.tensor([i % num_class for i in range(batch_size)])
    proba = torch.rand(batch_size, num_class)
    proba = proba / proba.sum(dim=1, keepdim=True)
    labels = torch.tensor([(i + 1) % num_class for i in range(batch_size)])
    sample_names = [f"clip_{start_idx + i:02d}" for i in range(batch_size)]
    return (predicted, proba, labels, sample_names)


class TestWritePredictions:
    def test_legacy_format_has_single_probabilities_column(self):
        batches = [_make_batch(num_class=5, batch_size=2, start_idx=0)]
        buf = io.StringIO()
        _write_predictions(batches, num_class=5, out_file=buf, per_class_columns=False)
        rows = list(csv.reader(io.StringIO(buf.getvalue())))
        assert rows[0] == ["sample_name", "true_label", "predicted_label", "probabilities"]
        assert len(rows) == 3  # header + 2 rows
        # Probabilities column is a space-separated string of N floats
        proba_strs = rows[1][3].split()
        assert len(proba_strs) == 5

    def test_per_class_columns_format(self):
        batches = [_make_batch(num_class=5, batch_size=2, start_idx=0)]
        buf = io.StringIO()
        _write_predictions(batches, num_class=5, out_file=buf, per_class_columns=True)
        rows = list(csv.reader(io.StringIO(buf.getvalue())))
        assert rows[0] == [
            "sample_name", "true_label", "predicted_label",
            "proba_0", "proba_1", "proba_2", "proba_3", "proba_4",
        ]
        assert len(rows) == 3
        # Each proba column is a single float string
        for row in rows[1:]:
            assert len(row) == 8  # 3 meta + 5 probas
            for p in row[3:]:
                float(p)  # parses as float

    def test_legacy_format_accepts_5tuple(self):
        # predict_step now returns a 5-tuple; _write_predictions must slice cleanly.
        pred, proba, labels, names = _make_batch(num_class=3, batch_size=2, start_idx=0)
        batch5 = (pred, proba, labels, names, None)
        buf = io.StringIO()
        _write_predictions([batch5], num_class=3, out_file=buf, per_class_columns=False)
        rows = list(csv.reader(io.StringIO(buf.getvalue())))
        assert len(rows) == 3

    def test_multiple_batches_concatenate(self):
        batches = [
            _make_batch(num_class=3, batch_size=2, start_idx=0),
            _make_batch(num_class=3, batch_size=3, start_idx=2),
        ]
        buf = io.StringIO()
        _write_predictions(batches, num_class=3, out_file=buf, per_class_columns=True)
        rows = list(csv.reader(io.StringIO(buf.getvalue())))
        assert len(rows) == 1 + 5  # header + 2+3 sample rows
        names = [r[0] for r in rows[1:]]
        assert names == [f"clip_{i:02d}" for i in range(5)]


class TestSaveInterpretability:
    def _make_batch_with_attention(self, batch_size, num_class, V, n_pro, start_idx):
        predicted = torch.tensor([i % num_class for i in range(batch_size)])
        proba = torch.rand(batch_size, num_class)
        proba = proba / proba.sum(dim=1, keepdim=True)
        labels = torch.tensor([(i + 1) % num_class for i in range(batch_size)])
        names = [f"clip_{start_idx + i:02d}" for i in range(batch_size)]
        attention = {
            "topology": torch.randn(batch_size, V, V),
            "prototype_response": torch.randn(batch_size, V * V, n_pro),
            "joint_saliency": torch.randn(batch_size, V),
        }
        return (predicted, proba, labels, names, attention)

    def test_writes_one_npz_per_sample(self, tmp_path):
        batch = self._make_batch_with_attention(
            batch_size=3, num_class=7, V=25, n_pro=16, start_idx=0
        )
        _save_interpretability([batch], tmp_path)
        files = sorted(p.name for p in tmp_path.glob("*.npz"))
        assert files == ["clip_00.npz", "clip_01.npz", "clip_02.npz"]

    def test_npz_contents_have_expected_keys_and_shapes(self, tmp_path):
        batch = self._make_batch_with_attention(
            batch_size=2, num_class=7, V=25, n_pro=16, start_idx=0
        )
        _save_interpretability([batch], tmp_path)
        with np.load(tmp_path / "clip_00.npz") as payload:
            assert payload["topology"].shape == (25, 25)
            assert payload["prototype_response"].shape == (25 * 25, 16)
            assert payload["joint_saliency"].shape == (25,)
            assert payload["proba"].shape == (7,)
            assert "predicted" in payload.files
            assert "true_label" in payload.files

    def test_none_attention_skipped(self, tmp_path):
        # STGCN-style batch: attention=None. Nothing should be written.
        pred, proba, labels, names = _make_batch(num_class=3, batch_size=2, start_idx=0)
        batch = (pred, proba, labels, names, None)
        _save_interpretability([batch], tmp_path)
        assert list(tmp_path.glob("*.npz")) == []

    def test_list_attention_stacked(self, tmp_path):
        """topology_all_layers comes through as a list of tensors and should
        be stacked into a (L, V, V) array."""
        predicted = torch.tensor([0, 1])
        proba = torch.rand(2, 3)
        labels = torch.tensor([0, 1])
        names = ["a", "b"]
        attention = {
            "topology": torch.randn(2, 5, 5),
            "topology_all_layers": [torch.randn(2, 5, 5) for _ in range(4)],
        }
        _save_interpretability(
            [(predicted, proba, labels, names, attention)], tmp_path
        )
        with np.load(tmp_path / "a.npz") as payload:
            assert payload["topology_all_layers"].shape == (4, 5, 5)
