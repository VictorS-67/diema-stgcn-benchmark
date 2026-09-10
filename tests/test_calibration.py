"""Tests for temperature scaling and the per-epoch calibration diagnostic.

The properties that matter here are the ones the report leans on:

- temperature scaling never changes a prediction, only its confidence;
- the fitted temperature moves the right way (>1 for an overconfident model,
  <1 for an underconfident one) and lowers cross-entropy;
- the training loop logs ``val_loss_tempered`` / ``val_temperature`` each
  epoch, and does not log them during Lightning's sanity-check pass, where
  the fit would run on two batches.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from emo_mocap.models.base import BaseModel
from emo_mocap.tools.calibration import (
    expected_calibration_error,
    fit_temperature,
)
from emo_mocap.training.lightning_model import LightningModel


def _separable_logits(n=400, num_class=7, scale=1.0, seed=0):
    """Logits that rank correctly ~most of the time, at a chosen confidence.

    ``scale`` is the knob under test: large means overconfident (the fit
    should return T > 1), small means underconfident (T < 1).
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_class, (n,), generator=g)
    logits = torch.randn(n, num_class, generator=g)
    # Push the true class up by a margin, then rescale the whole vector.
    logits[torch.arange(n), labels] += 2.0
    return logits * scale, labels


class TestTemperatureScaling:
    def test_predictions_are_unchanged(self):
        """The load-bearing property: T cannot alter argmax, so accuracy is fixed."""
        logits, labels = _separable_logits(scale=4.0)
        temperature = fit_temperature(logits, labels)

        before = logits.argmax(dim=1)
        after = (logits / temperature).argmax(dim=1)
        assert torch.equal(before, after)

    def test_overconfident_model_gets_temperature_above_one(self):
        logits, labels = _separable_logits(scale=5.0)
        assert fit_temperature(logits, labels) > 1.0

    def test_underconfident_model_gets_temperature_below_one(self):
        logits, labels = _separable_logits(scale=0.2)
        assert fit_temperature(logits, labels) < 1.0

    def test_fit_never_increases_cross_entropy(self):
        for scale in (0.2, 1.0, 5.0):
            logits, labels = _separable_logits(scale=scale)
            temperature = fit_temperature(logits, labels)
            raw = F.cross_entropy(logits, labels).item()
            tempered = F.cross_entropy(logits / temperature, labels).item()
            assert tempered <= raw + 1e-6, f"scale={scale}"

    def test_degenerate_input_returns_identity(self):
        assert fit_temperature(torch.empty(0, 7), torch.empty(0)) == 1.0

    def test_runs_under_inference_mode(self):
        """Validation runs in inference_mode; the fit must still work there.

        Inference tensors cannot join an autograd graph, so this guards the
        numpy round-trip in the Lightning hook against a silent regression.
        """
        logits, labels = _separable_logits(scale=4.0)
        with torch.inference_mode():
            arr, lab = logits.numpy(), labels.numpy()
        temperature = fit_temperature(torch.from_numpy(arr), torch.from_numpy(lab))
        assert temperature > 1.0


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_is_near_zero(self):
        """A model right exactly as often as it claims scores ~0."""
        n = 3000
        g = torch.Generator().manual_seed(1)
        conf = torch.full((n,), 0.8)
        correct = (torch.rand(n, generator=g) < 0.8).long()
        probs = torch.stack([1 - conf, conf], dim=1)
        labels = correct  # class 1 is the confident one
        assert expected_calibration_error(probs, labels) < 0.03

    def test_overconfident_model_scores_high(self):
        n = 1000
        probs = torch.stack([torch.full((n,), 0.01), torch.full((n,), 0.99)], dim=1)
        labels = torch.zeros(n, dtype=torch.long)  # always wrong
        assert expected_calibration_error(probs, labels) > 0.9

    def test_temperature_scaling_improves_ece(self):
        logits, labels = _separable_logits(scale=5.0)
        temperature = fit_temperature(logits, labels)
        raw = expected_calibration_error(logits.softmax(dim=1), labels)
        tempered = expected_calibration_error(
            (logits / temperature).softmax(dim=1), labels)
        assert tempered < raw


class _TinyModel(BaseModel):
    """Smallest thing that satisfies the protocol, so the test is about logging."""

    def __init__(self, num_class=7):
        super().__init__()
        self.num_class = num_class
        self.fc = nn.Linear(4, num_class)

    def forward(self, x):
        # (N, C, T, V) -> (N, C)
        return {"logits": self.fc(x.mean(dim=(2, 3)))}

    def output_dim(self):
        return self.num_class


def _lit_with_recorder(sanity_checking=False):
    """LightningModel wired to a fake trainer, recording what it logs."""
    lit = LightningModel(_TinyModel(), base_lr=0.1, num_class=7)
    logged = {}

    def record(name, value, **kw):
        # torchmetrics objects are logged by reference (val_acc); only the
        # scalars this test is about get unwrapped.
        if isinstance(value, (int, float, torch.Tensor)):
            logged[name] = float(value)

    lit.log = record

    trainer = MagicMock()
    trainer.sanity_checking = sanity_checking
    trainer.world_size = 1
    lit._trainer = trainer
    return lit, logged


class TestValidationCalibrationLogging:
    def _feed(self, lit, n_batches=6, batch_size=32):
        g = torch.Generator().manual_seed(3)
        for _ in range(n_batches):
            x = torch.randn(batch_size, 4, 8, 5, generator=g)
            y = torch.randint(0, 7, (batch_size,), generator=g)
            with torch.no_grad():
                lit.validation_step((x, y, ["n"] * batch_size), 0)

    def test_logs_tempered_loss_and_temperature(self):
        lit, logged = _lit_with_recorder()
        self._feed(lit)
        lit.on_validation_epoch_end()

        assert "val_loss_tempered" in logged
        assert "val_temperature" in logged
        assert "val_ece" in logged
        assert "val_ece_tempered" in logged
        assert logged["val_temperature"] > 0

    def test_tempered_loss_is_at_most_raw_loss(self):
        lit, logged = _lit_with_recorder()
        self._feed(lit)

        logits = torch.from_numpy(np.concatenate(lit._val_logits))
        labels = torch.from_numpy(np.concatenate(lit._val_labels)).long()
        raw = F.cross_entropy(logits, labels).item()

        lit.on_validation_epoch_end()
        assert logged["val_loss_tempered"] <= raw + 1e-6

    def test_sanity_check_pass_logs_no_calibration(self):
        """Two batches is not a split to fit a temperature on.

        ``val_loss`` is still logged — that comes from validation_step and is
        per-batch — but nothing calibration-related should appear.
        """
        lit, logged = _lit_with_recorder(sanity_checking=True)
        self._feed(lit, n_batches=2)
        lit.on_validation_epoch_end()
        assert not [k for k in logged if "temperature" in k or "tempered" in k
                    or "ece" in k]
        assert lit._val_logits == []

    def test_buffers_are_cleared_between_epochs(self):
        """Otherwise memory grows and epoch N is fitted on epochs 1..N."""
        lit, _ = _lit_with_recorder()
        self._feed(lit)
        assert lit._val_logits
        lit.on_validation_epoch_end()
        assert lit._val_logits == []
        assert lit._val_labels == []

    def test_distributed_is_refused_not_guessed(self):
        lit, logged = _lit_with_recorder()
        lit._trainer.world_size = 2
        self._feed(lit)
        with pytest.warns(RuntimeWarning, match="distributed"):
            lit.on_validation_epoch_end()
        assert "val_loss_tempered" not in logged
        assert lit._val_logits == []
