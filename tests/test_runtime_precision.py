"""Numeric policy is shared by every entry point, not set per-command.

`emo-train` used to configure TF32 and pass `cfg.training.precision` to its
Trainer while `emo-evaluate` / `emo-predict` built a bare
`pl.Trainer(deterministic=True)` — so the same checkpoint scored differently
depending on which command ran it. These tests pin the contract that fixes
it: one helper, one answer, and evaluation is full fp32 with TF32 off.
"""

import ast
from pathlib import Path

import pytest
import torch

from emo_mocap.tools.config import load_config
from emo_mocap.tools.runtime import (
    configure_eval_runtime,
    configure_training_runtime,
    resolve_precision,
)


CLI_DIR = Path(__file__).resolve().parent.parent / "emo_mocap" / "cli"
CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml"))


class TestEvalPrecisionContract:
    def test_default_eval_precision_is_full_fp32(self):
        """The default must not inherit training's mixed precision."""
        cfg = load_config(CONFIGS[0])
        assert cfg.training.eval_precision == "32-true"

    @pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.stem)
    def test_shipped_configs_evaluate_in_fp32(self, cfg_path):
        """Every shipped config trains mixed but evaluates exact."""
        cfg = load_config(cfg_path)
        assert configure_eval_runtime(cfg, verbose=False) == "32-true"

    def test_eval_disables_tf32_on_both_switches(self):
        """PyTorch has two TF32 flags with opposite defaults; fp32 means both off.

        Setting only `float32_matmul_precision` would leave every TCN
        convolution running in TF32 — most of the compute in an ST-GCN — so
        an "fp32 evaluation" that moved one flag would not be one.
        """
        if not torch.cuda.is_available():
            pytest.skip("TF32 flags are CUDA-only")
        cfg = load_config(CONFIGS[0])
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True

        configure_eval_runtime(cfg, verbose=False)

        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cudnn.allow_tf32 is False

    def test_training_and_eval_precision_are_independent(self, tmp_path):
        """A config may train in bf16 and still evaluate exactly."""
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]], lr_joint_pairs: []}\n"
            "training: {precision: bf16-mixed}\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.training.precision == "bf16-mixed"
        assert configure_eval_runtime(cfg, verbose=False) == "32-true"


class TestPrecisionFallback:
    def test_mixed_precision_falls_back_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_precision("bf16-mixed", context="t", verbose=False) == "32-true"

    def test_pure_float_modes_need_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        for p in ("32-true", "64-true"):
            assert resolve_precision(p, context="t", verbose=False) == p

    def test_training_runtime_returns_config_precision(self, tmp_path):
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]], lr_joint_pairs: []}\n"
            "training: {precision: 32-true}\n"
        )
        assert configure_training_runtime(
            load_config(cfg_file), verbose=False) == "32-true"


class TestNoEntryPointSetsPrecisionItself:
    """The regression that started this: policy set in one CLI and nowhere else."""

    @pytest.mark.parametrize("cli", ["train.py", "evaluate.py", "predict.py"])
    def test_cli_routes_precision_through_the_shared_helper(self, cli):
        src = (CLI_DIR / cli).read_text()
        assert "configure_eval_runtime" in src, (
            f"{cli} must resolve evaluation precision via tools/runtime.py")
        # No CLI may reach for the global TF32 knob on its own.
        assert "set_float32_matmul_precision" not in src, (
            f"{cli} sets matmul precision directly; that policy belongs in "
            f"tools/runtime.py so every entry point agrees")

    @pytest.mark.parametrize("cli", ["train.py", "evaluate.py", "predict.py"])
    def test_every_trainer_is_given_an_explicit_precision(self, cli):
        """A bare pl.Trainer() silently means 32-true — that was the bug."""
        tree = ast.parse((CLI_DIR / cli).read_text())
        trainers = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Trainer"
        ]
        assert trainers, f"no pl.Trainer(...) found in {cli}"
        for call in trainers:
            kwargs = {kw.arg for kw in call.keywords}
            assert "precision" in kwargs, (
                f"{cli}:{call.lineno} builds a Trainer without an explicit "
                f"precision=; it would default to 32-true and disagree with "
                f"whichever command set one")


class TestTestAfterUsesEvalPrecision:
    """`emo-train --test-after` must score the checkpoint like `emo-evaluate` does.

    A Lightning Trainer carries one precision, so the fit trainer cannot also
    run the test phase at a different one — train.py builds a second Trainer
    for it. Nothing else exercises that branch, so this runs the real CLI.
    """

    @pytest.mark.slow
    def test_test_phase_runs_at_32_true(self, tmp_path, monkeypatch):
        import json
        import pickle

        import numpy as np
        import pytorch_lightning as pl

        from emo_mocap.cli import train as train_cli

        J, N = 24, 8
        rng = np.random.default_rng(0)
        arrays = {}
        for i in range(N):
            q = rng.standard_normal((80, J, 4))
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            arrays[f"clip_{i}_root_pos"] = rng.standard_normal((80, 3))
            arrays[f"clip_{i}_joint_rot"] = q
        arrays["labels"] = np.array([i % 7 for i in range(N)], dtype=np.int64)
        arrays["filenames"] = np.array([f"c_{i:02d}" for i in range(N)])
        arrays["num_clips"] = np.array(N)
        arrays["representation"] = np.array("quat")
        arrays["skeleton_info_json"] = np.array(json.dumps(
            {"num_joints": J, "euler_orders": ["ZYX"] * J,
             "joint_names": [f"j{i}" for i in range(J)],
             "edges": [], "lr_pairs": []}))
        arrays["mean"] = np.zeros(3 + J * 4)
        arrays["std"] = np.ones(3 + J * 4)
        npz = tmp_path / "d.npz"
        np.savez(npz, **arrays)

        split = {"train": [(f"c_{i:02d}", i) for i in range(4)],
                 "val":   [(f"c_{i:02d}", i) for i in range(4, 6)],
                 "test":  [(f"c_{i:02d}", i) for i in range(6, 8)]}
        split_path = tmp_path / "s.pkl"
        split_path.write_bytes(pickle.dumps(split))

        # Record every Trainer the CLI builds, then delegate to the real one.
        built = []
        real_trainer = pl.Trainer

        def recording_trainer(**kwargs):
            built.append(kwargs)
            return real_trainer(**kwargs, enable_progress_bar=False,
                                enable_model_summary=False)

        monkeypatch.setattr(train_cli.pl, "Trainer", recording_trainer)
        # Lightning's isolate_rng() touches CUDA RNG state; keep this portable.
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("sys.argv", [
            "emo-train", "--config", "configs/diema7_stgcn_recipe.yaml",
            "--test-after", "--override",
            # The synthetic corpus above carries rotations only, so the
            # recipe's 15-channel position streams are switched off here.
            # This test is about the precision policy, not the recipe.
            "data.streams=null", "model.in_channels=6",
            f"data.data_path={npz}", f"data.split_path={split_path}",
            "training.max_epochs=1", "training.batch_size=2",
            "data.num_workers=0", "training.accelerator=cpu",
            "training.early_stopping=false", "augmentation.enabled=false",
            f"logging.log_dir={tmp_path / 'logs'}",
        ])

        train_cli.main()

        assert len(built) == 2, (
            f"expected a fit Trainer and a separate test Trainer, got {len(built)}")
        fit_kwargs, test_kwargs = built
        assert test_kwargs["precision"] == "32-true"
        # And the test Trainer must not carry checkpoint callbacks, whose
        # in-memory bookkeeping the test loop can otherwise reset.
        assert "callbacks" not in test_kwargs
