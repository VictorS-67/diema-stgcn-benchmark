"""Tests for the emo-ensemble CLI."""

import csv
import io
import math
import subprocess
import sys
from pathlib import Path

import pytest

from emo_mocap.cli.ensemble import (
    _entropy,
    _load_class_names,
    _load_predictions,
    ensemble_predictions,
    write_ensemble_csv,
)


def _write_csv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _write_per_class_csv(path, num_class, rows):
    """Rows are (sample_name, true_label, predicted_label, proba_list)."""
    header = ["sample_name", "true_label", "predicted_label",
              *[f"proba_{i}" for i in range(num_class)]]
    csv_rows = []
    for name, tl, pl, probas in rows:
        csv_rows.append([name, tl, pl, *[f"{p:.6f}" for p in probas]])
    _write_csv(path, csv_rows, header)


class TestLoadPredictions:
    def test_parses_per_class_columns(self, tmp_path):
        p = tmp_path / "preds.csv"
        _write_per_class_csv(p, 3, [
            ("clip_00", 0, 0, [0.8, 0.1, 0.1]),
            ("clip_01", 1, 2, [0.2, 0.3, 0.5]),
        ])
        rows, nc = _load_predictions(p)
        assert nc == 3
        assert rows[0]["sample_name"] == "clip_00"
        assert rows[0]["true_label"] == 0
        assert rows[0]["proba"] == pytest.approx([0.8, 0.1, 0.1])

    def test_proba_columns_sorted_numerically(self, tmp_path):
        # 12 classes: "proba_10" must come after "proba_9", not between 1 and 2.
        p = tmp_path / "preds.csv"
        probas = [i / 100.0 for i in range(12)]  # arbitrary, but distinct per index
        _write_per_class_csv(p, 12, [("clip_00", 0, 0, probas)])
        rows, nc = _load_predictions(p)
        assert nc == 12
        assert rows[0]["proba"] == pytest.approx(probas)

    def test_rejects_legacy_format(self, tmp_path):
        # The legacy 'probabilities' (space-separated) format is not accepted.
        p = tmp_path / "preds.csv"
        _write_csv(p, [["clip_00", 0, 0, "0.8 0.1 0.1"]],
                   header=["sample_name", "true_label", "predicted_label", "probabilities"])
        with pytest.raises(ValueError, match="no proba_"):
            _load_predictions(p)


class TestLoadClassNames:
    def test_none_returns_integer_names(self):
        assert _load_class_names(None, 3) == ["0", "1", "2"]

    def test_one_per_line(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("anger\njoy\nsadness\n")
        assert _load_class_names(p, 3) == ["anger", "joy", "sadness"]

    def test_name_idx_pairs(self, tmp_path):
        # Sorted by idx, so out-of-order lines are handled.
        p = tmp_path / "names.txt"
        p.write_text("sadness 2\nanger 0\njoy 1\n")
        assert _load_class_names(p, 3) == ["anger", "joy", "sadness"]

    def test_reads_real_emo_to_idx(self):
        # Uses the repo's canonical label file; first 7 names should be the
        # basic-emotion set (matches diema7 configs).
        names = _load_class_names(Path("configs/emo_to_idx.txt"), 7)
        assert names == ["anger", "contempt", "disgust", "fear", "joy", "sadness", "surprise"]

    def test_too_few_names_raises(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("anger\njoy\n")
        with pytest.raises(ValueError, match="has 2 names"):
            _load_class_names(p, 5)

    def test_truncates_when_more_names_than_classes(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("anger\njoy\nsadness\nfear\n")
        assert _load_class_names(p, 2) == ["anger", "joy"]


class TestEntropy:
    def test_uniform_distribution(self):
        p = [0.25, 0.25, 0.25, 0.25]
        assert _entropy(p) == pytest.approx(math.log(4))

    def test_degenerate_distribution(self):
        assert _entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_zeros_are_ignored(self):
        # Must not blow up on log(0); should return the same value as nonzero-only.
        assert _entropy([0.5, 0.5, 0.0]) == pytest.approx(math.log(2))


class TestEnsemblePredictions:
    def _make_rows(self, samples):
        """samples is list of (name, true_label, proba_list)."""
        return [
            {"sample_name": n, "true_label": tl, "proba": p}
            for n, tl, p in samples
        ]

    def test_mean_is_arithmetic_mean(self):
        inputs = [
            self._make_rows([("clip_00", 0, [1.0, 0.0, 0.0]),
                             ("clip_01", 1, [0.0, 1.0, 0.0])]),
            self._make_rows([("clip_00", 0, [0.0, 0.0, 1.0]),
                             ("clip_01", 1, [0.0, 0.0, 1.0])]),
        ]
        results = ensemble_predictions(inputs, num_class=3)
        assert results[0]["mean"] == pytest.approx([0.5, 0.0, 0.5])
        assert results[1]["mean"] == pytest.approx([0.0, 0.5, 0.5])

    def test_ensemble_argmax(self):
        # Predictor 1 and 2 disagree; ensemble averages and argmaxes.
        inputs = [
            self._make_rows([("clip_00", 0, [0.6, 0.4])]),
            self._make_rows([("clip_00", 0, [0.3, 0.7])]),
        ]
        results = ensemble_predictions(inputs, num_class=2)
        # Mean is [0.45, 0.55] → argmax = 1
        assert results[0]["ensemble_pred"] == 1

    def test_agreement_count(self):
        # Three predictors: two pick class 0, one picks class 1. Ensemble mean:
        # [0.5, 0.5] — a tie. argmax picks class 0 (first max). 2 predictors agree.
        inputs = [
            self._make_rows([("clip_00", 0, [0.9, 0.1])]),
            self._make_rows([("clip_00", 0, [0.6, 0.4])]),
            self._make_rows([("clip_00", 0, [0.0, 1.0])]),
        ]
        results = ensemble_predictions(inputs, num_class=2)
        assert results[0]["ensemble_pred"] == 0
        assert results[0]["agreement_count"] == 2

    def test_entropy_of_mean_vs_mean_entropy(self):
        # Two confident but disagreeing predictors:
        #   mean_entropy  = mean of two ~0 entropies = ~0
        #   entropy_of_mean = H([0.5, 0.5]) = log(2)
        inputs = [
            self._make_rows([("clip_00", 0, [1.0, 0.0])]),
            self._make_rows([("clip_00", 0, [0.0, 1.0])]),
        ]
        results = ensemble_predictions(inputs, num_class=2)
        assert results[0]["mean_entropy"] == pytest.approx(0.0)
        assert results[0]["entropy_of_mean"] == pytest.approx(math.log(2))

    def test_std_for_identical_inputs_is_zero(self):
        inputs = [
            self._make_rows([("clip_00", 0, [0.7, 0.3])]),
            self._make_rows([("clip_00", 0, [0.7, 0.3])]),
        ]
        results = ensemble_predictions(inputs, num_class=2)
        assert results[0]["std"] == pytest.approx([0.0, 0.0])

    def test_order_follows_first_input(self):
        # Second input provides samples in reverse order; output order should
        # match the first input's order.
        inputs = [
            self._make_rows([("a", 0, [1.0, 0.0]), ("b", 1, [0.0, 1.0])]),
            self._make_rows([("b", 1, [0.0, 1.0]), ("a", 0, [1.0, 0.0])]),
        ]
        results = ensemble_predictions(inputs, num_class=2)
        assert [r["sample_name"] for r in results] == ["a", "b"]

    def test_mismatched_sample_set_raises(self):
        inputs = [
            self._make_rows([("a", 0, [1.0, 0.0])]),
            self._make_rows([("b", 0, [1.0, 0.0])]),
        ]
        with pytest.raises(ValueError, match="disagree on sample set"):
            ensemble_predictions(inputs, num_class=2)

    def test_mismatched_true_label_raises(self):
        inputs = [
            self._make_rows([("a", 0, [1.0, 0.0])]),
            self._make_rows([("a", 1, [1.0, 0.0])]),
        ]
        with pytest.raises(ValueError, match="disagree on true_label"):
            ensemble_predictions(inputs, num_class=2)


class TestWriteEnsembleCSV:
    def test_header_uses_class_names(self):
        results = [{
            "sample_name": "clip_00", "true_label": 0, "ensemble_pred": 1,
            "agreement_count": 2, "entropy_of_mean": 0.5, "mean_entropy": 0.3,
            "mean": [0.1, 0.9], "std": [0.01, 0.01],
        }]
        buf = io.StringIO()
        write_ensemble_csv(results, ["anger", "joy"], buf)
        rows = list(csv.reader(io.StringIO(buf.getvalue())))
        assert rows[0] == [
            "sample_name", "true_label", "ensemble_pred", "agreement_count",
            "entropy_of_mean", "mean_entropy",
            "mean_anger", "mean_joy", "std_anger", "std_joy",
        ]

    def test_integer_class_names(self):
        results = [{
            "sample_name": "clip_00", "true_label": 0, "ensemble_pred": 0,
            "agreement_count": 1, "entropy_of_mean": 0.0, "mean_entropy": 0.0,
            "mean": [1.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0],
        }]
        buf = io.StringIO()
        write_ensemble_csv(results, ["0", "1", "2"], buf)
        rows = list(csv.reader(io.StringIO(buf.getvalue())))
        assert rows[0][6:] == ["mean_0", "mean_1", "mean_2", "std_0", "std_1", "std_2"]


class TestEnsembleCLIEndToEnd:
    """Exercise the CLI: build two per-class CSVs, run emo-ensemble, inspect output."""

    def test_cli_produces_expected_output(self, tmp_path):
        in_a = tmp_path / "seed_a.csv"
        in_b = tmp_path / "seed_b.csv"
        _write_per_class_csv(in_a, 3, [
            ("clip_00", 0, 0, [0.9, 0.05, 0.05]),
            ("clip_01", 2, 2, [0.1, 0.2, 0.7]),
        ])
        _write_per_class_csv(in_b, 3, [
            ("clip_00", 0, 0, [0.7, 0.2, 0.1]),
            ("clip_01", 2, 1, [0.3, 0.5, 0.2]),
        ])
        out = tmp_path / "ensemble.csv"

        result = subprocess.run(
            [sys.executable, "-m", "emo_mocap.cli.ensemble",
             "--inputs", str(in_a), str(in_b), "--output", str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Accuracy:" in result.stdout

        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        # clip_00: mean = [0.8, 0.125, 0.075] → argmax 0
        assert rows[0]["ensemble_pred"] == "0"
        # clip_01: mean = [0.2, 0.35, 0.45] → argmax 2
        assert rows[1]["ensemble_pred"] == "2"

    def test_cli_rejects_single_input(self, tmp_path):
        in_a = tmp_path / "seed_a.csv"
        _write_per_class_csv(in_a, 2, [("clip_00", 0, 0, [0.8, 0.2])])
        out = tmp_path / "ensemble.csv"
        result = subprocess.run(
            [sys.executable, "-m", "emo_mocap.cli.ensemble",
             "--inputs", str(in_a), "--output", str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "at least 2" in result.stderr
