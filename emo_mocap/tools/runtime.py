"""Process-wide numeric policy, shared by every CLI entry point.

Training and evaluation used to configure the runtime *differently*, and
only by accident: ``emo-train`` set the matmul precision and passed
``cfg.training.precision`` to its Trainer, while ``emo-evaluate`` and
``emo-predict`` built a bare ``pl.Trainer(deterministic=True)`` and so
silently ran at Lightning's ``32-true`` default with TF32 matmuls off.
The same checkpoint on the same data therefore scored differently
depending on which command you ran — measured on the STGCN config, a
``max|logit diff|`` of 5.1e-01 for bf16-mixed vs 32-true and 3.0e-02 for
TF32 vs fp32, either of which can tip a clip sitting near a decision
boundary.

This module is the single place that policy lives. The rule it encodes:

* **Training** runs fast — ``cfg.training.precision`` (bf16-mixed on the
  4090) with TF32 matmuls enabled.
* **Evaluation** runs *precise* — full fp32 with TF32 off everywhere.
  Evaluation is a measurement, and bf16's ~4e-3 relative error is noise
  injected into a reported metric to speed up a pass over a few hundred
  clips. fp32 also makes the number hardware-independent, so a
  collaborator on a non-Ampere GPU reproduces the table.

A note on why there are two TF32 switches below: PyTorch keeps separate
flags for matmuls and for convolutions, **with opposite defaults**
(``cuda.matmul.allow_tf32`` is False, ``cudnn.allow_tf32`` is True), and
``set_float32_matmul_precision`` moves only the first. Setting just that
one would leave every TCN convolution running in TF32 — most of the
compute in an ST-GCN — so "fp32 evaluation" has to say both.
"""

import torch


def _needs_cuda(precision: str) -> bool:
    """Whether a Lightning precision string requires a CUDA device.

    The pure-float modes run anywhere; every mixed / reduced mode needs one.
    """
    return precision not in ("32-true", "64-true")


def resolve_precision(requested: str, *, context: str, verbose: bool = True) -> str:
    """Fall back to ``32-true`` when a mixed precision was asked for without CUDA.

    Keeps CPU-only machines (and the test suite) working with a config that
    names ``bf16-mixed``, instead of failing at Trainer construction.
    """
    if _needs_cuda(requested) and not torch.cuda.is_available():
        if verbose:
            print(f"Warning: {context} precision={requested!r} requested but "
                  f"CUDA unavailable; falling back to 32-true.")
        return "32-true"
    return requested


def set_tf32(enabled: bool) -> None:
    """Turn TF32 on or off for **both** matmuls and convolutions.

    TF32 keeps fp32's 8-bit exponent but truncates the mantissa to 10 bits,
    accumulating in fp32 — same dynamic range, ~5e-4 relative precision
    instead of ~1e-7. Worth it for training throughput on Tensor Cores,
    not worth it for a measurement.
    """
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    torch.backends.cudnn.allow_tf32 = enabled


def configure_training_runtime(cfg, *, verbose: bool = True) -> str:
    """Set the numeric policy for a training run; return its Lightning precision.

    TF32 is enabled only on compute capability >= 8 (Ampere and later), which
    is where Tensor Cores support it; on older cards the flag is a no-op but
    setting it is still pointless noise.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        set_tf32(major >= 8)
    requested = getattr(cfg.training, "precision", "32-true")
    return resolve_precision(requested, context="training", verbose=verbose)


def configure_eval_runtime(cfg, *, verbose: bool = True) -> str:
    """Set the numeric policy for evaluation; return its Lightning precision.

    Called by ``emo-evaluate``, ``emo-predict``, and by ``emo-train`` before
    its ``--test-after`` phase — so a checkpoint scores the same however you
    reach it. When ``eval_precision`` is the default ``32-true`` this also
    disables TF32, which is what makes the number reproducible off this GPU;
    a config that deliberately evaluates in mixed precision keeps TF32 on,
    since it has already opted out of exactness.
    """
    requested = getattr(cfg.training, "eval_precision", "32-true")
    precision = resolve_precision(requested, context="eval", verbose=verbose)
    set_tf32(precision != "32-true")
    return precision
