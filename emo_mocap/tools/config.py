"""YAML configuration loader with validation.

Loads experiment configs from YAML files and provides structured access
to settings with validation for required fields and sensible defaults.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_namespace(d):
    """Recursively convert a dict to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(item) for item in d]
    return d


_REQUIRED_FIELDS = [
    ("data", "data_path"),
    ("model", "type"),
    ("model", "num_class"),
    ("skeleton", "num_nodes"),
    ("skeleton", "inward_edges"),
]

_DEFAULTS = {
    "model": {
        "in_channels": 3,
        "dropout": 0.5,
        "dual_loss": False,
    },
    "training": {
        "base_lr": 0.1,
        "optimizer": "SGD",
        "scheduler_type": "cosine",
        "scheduler_params": [],
        "weight_decay": 0.0001,
        # Softens the one-hot target: the true class gets 1 - eps + eps/K and
        # every other class eps/K. Judged on test accuracy, not on calibration
        # — temperature scaling already fixes calibration after the fact at no
        # accuracy risk, so smoothing has to earn its place as a regulariser.
        "label_smoothing": 0.0,
        "max_epochs": 100,
        "batch_size": 32,
        "clip_length": 64,
        "aux_loss_weights": {},
        "accelerator": "auto",
        "devices": "auto",
        # Mixed-precision training. "32-true" preserves legacy float32 behavior.
        # RTX 4090 and other Ampere+ cards benefit from "bf16-mixed" (~1.5-2x
        # speedup, lower VRAM, no loss scaling needed). "16-mixed" uses float16
        # + gradient scaler — older cards.
        "precision": "32-true",
        # Precision for *evaluation* — emo-evaluate, emo-predict, and
        # emo-train's --test-after phase, which all route through
        # tools/runtime.py so one checkpoint scores the same however you
        # reach it. Deliberately independent of `precision` and defaulting
        # to full fp32 (TF32 off too): a metric is a measurement, and
        # bf16's ~4e-3 relative error buys speed on a pass over a few
        # hundred clips that nobody needs, at the cost of a number that
        # only reproduces on Ampere+.
        "eval_precision": "32-true",
        # Global L2 norm clip on gradients (0 disables). Stabilises bf16-mixed
        # training — Adaptive GCNs with high LR can produce transient inf in
        # intermediate activations, which become NaN after backward.
        "gradient_clip_val": 0.0,
        "early_stopping": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": 10,
    },
    "data": {
        "split_path": None,
        "num_workers": None,  # None = auto-detect (half CPU cores, capped at 8)
        "seed": 255,
        "target_repr": "euler",
        # Fraction of *training performers* to keep (1.0 = all). Subsamples
        # by actor, never by clip, so a smaller value means "fewer people",
        # not "less data" — see splits.subsample_train_performers. Val and
        # test are never touched. For the Track D learning curve.
        "train_performer_fraction": 1.0,
        # Which streams the packed (C, T, V) tensor carries, in channel order
        # (pybvh-ml >= 0.6). None keeps the historical default —
        # root translation as vertex 0 plus joint rotations, V = 1 + J.
        # Naming position/derived streams changes the vertex space: without
        # "root_pos" in the list V = J and skeleton.inward_edges must be the
        # joint-space edge list (skeleton_info["edges"]), not the 25-vertex
        # root-prefixed one. E.g. ["joint_pos"] -> (3, T, 24) NTU-style input;
        # ["joint_pos", "joint_vel", "joint_acc"] -> (9, T, 24).
        "streams": None,
        # Divide every position stream by the performer's own skeleton size
        # (sum of bone lengths), so body size stops riding in every channel.
        # Only meaningful with position streams — rotations are already
        # scale-free. See Feeder.scale_normalize; on this corpus dividing body
        # size out helps position-only input and hurts the mixed input the
        # recommended config uses, so it ships off.
        "scale_normalize": False,
    },
    "augmentation": {
        "enabled": False,
        "rotate": False,
        "rotate_prob": 1.0,
        "rotate_range": [-180, 180],
        "mirror": False,
        "mirror_prob": 0.5,
        "speed": False,
        "speed_prob": 1.0,
        "speed_range": [0.8, 1.2],
        "noise_sigma": 0.0,
        # Frame dropout: randomly drop frames and SLERP-fill. Good occlusion
        # robustness. Shape is preserved so no downstream changes needed.
        "dropout": False,
        "dropout_prob": 1.0,
        "dropout_rate": 0.1,
    },
    "logging": {
        "experiment_name": None,
        "log_dir": "logs/",
    },
    # Checkpointing defaults preserve the legacy behavior: save the best
    # val_acc checkpoint, but test the in-memory (final-epoch) weights.
    # Set ``test_with`` to a preset name (e.g. ``best_val_acc``) to load a
    # checkpoint from disk before testing.
    "checkpointing": {
        "save": ["best_val_acc"],
        "test_with": "current",
    },
}


def _apply_defaults(raw: dict) -> dict:
    """Apply default values to a raw config dict (non-destructive)."""
    for section, defaults in _DEFAULTS.items():
        if section not in raw:
            raw[section] = {}
        for key, value in defaults.items():
            if key not in raw[section]:
                raw[section][key] = value
    return raw


def _validate(raw: dict, path: str) -> None:
    """Validate that all required fields are present."""
    for section, field in _REQUIRED_FIELDS:
        if section not in raw:
            raise ValueError(
                f"Config {path}: missing required section '{section}'"
            )
        if field not in raw[section]:
            raise ValueError(
                f"Config {path}: missing required field '{section}.{field}'"
            )


def load_config(path: str | Path) -> SimpleNamespace:
    """Load and validate a YAML config file.

    Args:
        path: path to the YAML config file

    Returns:
        A nested SimpleNamespace with config values accessible as attributes
        (e.g., config.model.type, config.training.base_lr)

    Raises:
        FileNotFoundError: if the config file doesn't exist
        ValueError: if required fields are missing
        yaml.YAMLError: if the YAML is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path}: expected a YAML mapping, got {type(raw).__name__}")

    _validate(raw, str(path))
    raw = _apply_defaults(raw)

    return _to_namespace(raw)


# YAML's null spellings. Deliberately excludes "none", which is a legitimate
# string value in this schema (skeleton graph mode `edge_weighting: none`).
_NULL_TOKENS = {"null", "~"}


def _coerce_value(value_str):
    """Coerce a string value to a list, dict, None, int, float, bool, or str.

    Bracketed values are parsed as YAML so list- and mapping-valued fields are
    reachable from the CLI: ``checkpointing.save='[every_10_epochs, last]'``.
    Without this they arrived as a bare string, and a consumer that iterates
    the field walked it character by character — ``save=last`` failed with
    "Unknown checkpointing preset: 'l'", which reads like a typo in the value
    rather than a type error in the override.

    Scalars keep the hand-rolled ladder rather than going through YAML too:
    YAML 1.1 reads ``no`` / ``off`` / ``y`` as booleans, which would quietly
    turn a legitimate short string value into ``False``.
    """
    stripped = value_str.strip()
    if stripped[:1] in ("[", "{"):
        try:
            return yaml.safe_load(stripped)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Override value {value_str!r} starts like a list or mapping "
                f"but is not valid YAML: {exc}"
            ) from exc
    if stripped.lower() in _NULL_TOKENS:
        return None
    if stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return value_str


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply dot-separated key=value overrides to a config dict.

    Args:
        raw: the raw config dict (modified in-place)
        overrides: list of strings like 'training.max_epochs=200'

    Returns:
        The modified config dict
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (no '='): {override}")
        key, value = override.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Override key must be section.field (got {key})"
            )
        section, field = parts
        if section not in raw:
            raw[section] = {}
        raw[section][field] = _coerce_value(value)
    return raw


def interpolate_log_dir(log_dir: str, seed: int, fold: int | None = None) -> str:
    """Substitute ``{seed}`` and ``{fold}`` placeholders in a log_dir string.

    Lets a single config drive multi-seed sweeps and per-fold output trees
    without YAML duplication. Examples::

        "logs/"                       -> "logs/"
        "logs/seed{seed}/"            -> "logs/seed42/"
        "logs/seed{seed}/fold{fold}/" -> "logs/seed42/fold03/"

    Fold is formatted as a zero-padded 2-digit integer to match the
    ``{model}_fold{NN}`` experiment-name convention used elsewhere.

    Raises ValueError if ``{fold}`` is referenced but no fold was supplied.
    """
    if "{fold}" in log_dir and fold is None:
        raise ValueError(
            f"logging.log_dir={log_dir!r} references {{fold}} but no --fold was given. "
            "Use --fold/--num-folds for LPO runs, or drop {fold} from log_dir."
        )
    fold_str = f"{fold:02d}" if fold is not None else ""
    return log_dir.format(seed=seed, fold=fold_str)


def load_config_with_overrides(path: str | Path, overrides: list[str] | None = None) -> SimpleNamespace:
    """Load a YAML config, apply overrides, validate, and return namespace.

    Convenience wrapper combining load_config and apply_overrides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path}: expected a YAML mapping, got {type(raw).__name__}")

    if overrides:
        raw = apply_overrides(raw, overrides)

    _validate(raw, str(path))
    raw = _apply_defaults(raw)

    return _to_namespace(raw)
