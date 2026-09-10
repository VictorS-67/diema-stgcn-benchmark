"""The two plotting helpers must return a figure and not mutate their input."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from emo_mocap.tools.visualization import plot_CSVLogger


class TestTrainingCurve:
    def _metrics(self, tmp_path, rows=6):
        csv = tmp_path / "metrics.csv"
        pd.DataFrame({
            "epoch": np.arange(rows),
            "train_loss": np.linspace(2.0, 0.1, rows),
            "val_loss": np.linspace(2.0, 1.2, rows),
            "val_acc": np.linspace(0.15, 0.42, rows),
        }).to_csv(csv, index=False)
        return csv

    def test_returns_a_loss_axis_and_an_accuracy_axis(self, tmp_path):
        loss_ax, acc_ax = plot_CSVLogger(self._metrics(tmp_path))
        assert isinstance(loss_ax, Axes) and isinstance(acc_ax, Axes)
        assert loss_ax.get_ylabel() == "Loss" and acc_ax.get_ylabel() == "Acc"

    def test_accepts_a_string_path(self, tmp_path):
        loss_ax, _ = plot_CSVLogger(str(self._metrics(tmp_path)))
        assert isinstance(loss_ax, Axes)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            plot_CSVLogger(tmp_path / "nope.csv")
