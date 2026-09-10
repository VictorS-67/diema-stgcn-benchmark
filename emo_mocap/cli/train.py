"""Train an emotion recognition model.

Usage:
    # Single split (from config or override)
    emo-train --config configs/diema7_stgcn.yaml
    emo-train --config configs/diema7_stgcn.yaml --override training.max_epochs=50

    # LPO cross-validation (on-the-fly split, no pkl files needed)
    emo-train --config configs/diema12_stgcn.yaml --fold 3 --num-folds 10
"""

import argparse
import hashlib
import platform
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import pybvh
import pybvh_ml

from emo_mocap.tools.config import interpolate_log_dir, load_config_with_overrides
from emo_mocap.tools.runtime import configure_eval_runtime, configure_training_runtime
from emo_mocap.models.registry import get_model
from emo_mocap.data.loader import EpochSeedCallback, Loader
from emo_mocap.data.splits import (
    build_lpo_split,
    parse_diema_actor,
    subsample_train_performers,
)
from emo_mocap.training.lightning_model import LightningModel


_AXIS_STRINGS = {0: "+x", 1: "+y", 2: "+z"}


# Each preset maps to a single ModelCheckpoint configuration. ``last`` is
# special-cased: it uses Lightning's ``save_last=True`` and tracks no metric.
_CKPT_PRESETS = {
    "best_val_loss": {
        "monitor": "val_loss",
        "mode": "min",
        "filename": "best-val-loss-{epoch:02d}-{val_loss:.4f}",
    },
    "best_val_acc": {
        "monitor": "val_acc",
        "mode": "max",
        "filename": "best-val-acc-{epoch:02d}-{val_acc:.4f}",
    },
    "best_train_loss": {
        "monitor": "train_loss",
        "mode": "min",
        "filename": "best-train-loss-{epoch:02d}-{train_loss:.4f}",
    },
}


_EVERY_N_RE = re.compile(r"^every_(\d+)_epochs$")


def _build_checkpoint_callbacks(save_list):
    """Build one ModelCheckpoint per preset name in ``save_list``.

    Returns ``(callbacks, by_preset)`` where ``by_preset`` maps the preset
    name to its ModelCheckpoint, used later to resolve test-time loading.

    In addition to the named presets, accepts the pattern
    ``every_{N}_epochs`` (e.g. ``every_10_epochs``), which keeps all
    checkpoints saved every N epochs. Used for learning-curve diagnostics.
    """
    if isinstance(save_list, str):
        # A bare string is iterable, so this used to walk it character by
        # character and report "Unknown checkpointing preset: 'l'".
        raise TypeError(
            f"checkpointing.save must be a list, got the string "
            f"{save_list!r}. From the CLI, quote it as a YAML list: "
            f"--override \"checkpointing.save=[{save_list}]\"."
        )
    callbacks = []
    by_preset = {}
    for preset in save_list:
        if preset == "last":
            cb = ModelCheckpoint(save_top_k=0, save_last=True)
        elif preset in _CKPT_PRESETS:
            spec = _CKPT_PRESETS[preset]
            cb = ModelCheckpoint(
                monitor=spec["monitor"],
                mode=spec["mode"],
                save_top_k=1,
                filename=spec["filename"],
            )
        elif (match := _EVERY_N_RE.match(preset)) is not None:
            n = int(match.group(1))
            cb = ModelCheckpoint(
                every_n_epochs=n,
                save_top_k=-1,
                filename="epoch-{epoch:03d}",
            )
        else:
            available = sorted(list(_CKPT_PRESETS) + ["last", "every_N_epochs"])
            raise ValueError(
                f"Unknown checkpointing preset: {preset!r}. Available: {available}"
            )
        callbacks.append(cb)
        by_preset[preset] = cb
    return callbacks, by_preset


def _resolve_test_ckpt_path(test_with, by_preset):
    """Return the on-disk checkpoint path to load for ``trainer.test()``.

    Returns ``None`` for the sentinel ``"current"``, which means "test the
    in-memory model state" (the legacy final-epoch behavior).
    """
    if test_with == "current":
        return None
    if test_with not in by_preset:
        raise ValueError(
            f"checkpointing.test_with={test_with!r} is not in checkpointing.save "
            f"({sorted(by_preset)}). Add it to save, or use 'current'."
        )
    cb = by_preset[test_with]
    path = cb.last_model_path if test_with == "last" else cb.best_model_path
    return path or None


def _config_axis(skeleton, axis_attr, idx_attr):
    """Read a signed-axis string from skeleton config, or None if unset.

    Accepts either ``*_axis`` (signed string like ``"+z"``) or the legacy
    ``*_idx`` (integer 0/1/2 for X/Y/Z, positive only).
    """
    axis = getattr(skeleton, axis_attr, None)
    if isinstance(axis, str):
        return axis
    idx = getattr(skeleton, idx_attr, None)
    if idx is not None:
        return _AXIS_STRINGS[int(idx)]
    return None


def _third_axis(up, forward):
    """The axis left over once up and forward are spoken for — the lateral one.

    pybvh-ml records ``world_up`` and ``rest_forward`` in the dataset but not
    the lateral axis, and the array-level ``pybvh_ml.mirror`` needs one
    explicitly: the rest-pose measurement that lets ``pybvh.transforms.mirror``
    auto-detect it (averaging left-minus-right bone offsets) requires a
    skeleton, which the packed arrays no longer carry. So it is derived by
    elimination. Mirroring is sign-invariant, so the returned sign is arbitrary.

    Parsing goes through ``pybvh.parse_axis`` (public since pybvh 0.8.1)
    instead of slicing the strings by hand. That also makes a malformed axis
    raise here — where the offending string is named — rather than becoming a
    silent ``None`` that falls through to the ``+x`` default.

    Returns None when either axis is absent, or when the two name the same
    axis (a dataset that can't distinguish forward from up leaves no leftover).
    """
    if not up or not forward:
        return None
    remaining = {0, 1, 2} - {pybvh.parse_axis(up).index,
                             pybvh.parse_axis(forward).index}
    if len(remaining) != 1:
        return None
    return _AXIS_STRINGS[remaining.pop()]


def _resolve_axis(name, from_config, from_dataset, default):
    """Reconcile a skeleton axis declared in the config with the dataset's own.

    pybvh-ml >= 0.5 records ``world_up`` / ``rest_forward`` in the saved
    ``skeleton_info``, so the dataset can answer this question itself. When
    the config also answers it and the two disagree, one of them is wrong and
    the augmentation is silently ruined — a "vertical" rotation about a
    horizontal axis tips the performer over instead of yawing them, which
    destroys exactly the gravity-relative posture cues emotion recognition
    reads. So: raise, don't pick a winner.
    """
    if from_config and from_dataset and from_config != from_dataset:
        raise ValueError(
            f"skeleton.{name} in the config is {from_config!r} but the "
            f"preprocessed dataset reports {from_dataset!r}. Augmentation "
            f"about the wrong axis silently corrupts the data — fix the "
            f"config (or re-preprocess), don't guess."
        )
    resolved = from_config or from_dataset
    if resolved is None:
        warnings.warn(
            f"No skeleton.{name} in the config and none recorded in the "
            f"dataset (preprocessed by pybvh-ml < 0.5?); falling back to "
            f"{default!r}. Verify this matches your data.",
            RuntimeWarning, stacklevel=2,
        )
        return default
    return resolved


def _apply_performer_fraction(cfg, split_dict):
    """Shrink the training cohort if ``data.train_performer_fraction`` says so.

    A no-op at the default 1.0. Announces itself when it does anything,
    because a silently smaller training set is the kind of thing that gets
    mistaken for a bad result three experiments later.
    """
    fraction = getattr(cfg.data, "train_performer_fraction", 1.0)
    if fraction == 1.0:
        return split_dict

    before = len({parse_diema_actor(f) for f, _ in split_dict["train"]})
    n_clips_before = len(split_dict["train"])
    split_dict = subsample_train_performers(
        split_dict, fraction, seed=cfg.data.seed)
    after = len({parse_diema_actor(f) for f, _ in split_dict["train"]})
    print(f"train_performer_fraction={fraction}: {before} -> {after} performers, "
          f"{n_clips_before} -> {len(split_dict['train'])} clips "
          f"(val/test untouched)")
    return split_dict


def _needed_position_fields(streams):
    """Position fields the requested streams put into the sample container.

    Mirrors the Feeder's own resolution: a derived stream (``joint_vel``)
    needs its base (``joint_pos``) present in the container, so the pipeline
    must be built to handle that stream even though it is never packed as-is.
    """
    derived = {"joint_vel": "joint_pos", "joint_acc": "joint_pos",
               "node_vel": "node_pos", "node_acc": "node_pos"}
    return {derived.get(s, s) for s in (streams or ())} & {"joint_pos", "node_pos"}


def _build_pipeline(cfg, skeleton_info=None, streams=None):
    """Build a pybvh_ml AugmentationPipeline from config.

    Angles in the YAML are in **degrees** — that is the unit a researcher
    thinks in when writing a config. pybvh-ml is radians-first, but every
    angle-taking augmentation accepts ``degrees=True``, so the config's unit
    is declared at the call site and the library converts. That is one
    convention for both angle steps; the earlier ``math.radians()`` at this
    boundary meant rotation and noise were spelled two different ways.

    ``skeleton_info`` is the dataset's own skeleton metadata (from
    ``pybvh_ml.load_preprocessed``); when supplied it is cross-checked against
    the config's axis declarations.

    ``streams`` is ``cfg.data.streams``: when it pulls a position stream into
    the sample container, two steps need extra wiring that only the dataset's
    own metadata can provide (pybvh-ml >= 0.6) — the rotation-noise step
    refreshes positions by forward kinematics and needs ``fk_topology``, and
    a node-space mirror needs the node left/right pairing.

    Returns None if augmentation is disabled.
    """
    aug_cfg = cfg.augmentation
    if not aug_cfg.enabled:
        return None

    skeleton = cfg.skeleton
    skeleton_info = skeleton_info or {}
    position_fields = _needed_position_fields(streams)

    if "node_pos" in position_fields and skeleton_info.get(
            "mismatched_end_site_pairs"):
        # pybvh's node_lr_pairs *drops* a pair whose two sides carry
        # different end-site counts; mirroring with a dropped tip produces
        # a half-swapped skeleton. The check ran at preprocessing time and
        # was recorded — refuse to augment node positions on such a rig.
        raise ValueError(
            f"Dataset records mismatched end-site pairs "
            f"{skeleton_info['mismatched_end_site_pairs']}; node-space "
            f"mirror would half-swap those tips. Fix the rig or drop the "
            f"node streams."
        )
    up_axis = _resolve_axis(
        "up_axis",
        _config_axis(skeleton, "up_axis", "up_idx"),
        skeleton_info.get("world_up"),
        "+y",
    )
    lateral_axis = _resolve_axis(
        "lateral_axis",
        _config_axis(skeleton, "lateral_axis", "lateral_idx"),
        _third_axis(skeleton_info.get("world_up"),
                    skeleton_info.get("rest_forward")),
        "+x",
    )
    steps = []

    if getattr(aug_cfg, "rotate", False):
        lo, hi = aug_cfg.rotate_range
        steps.append((
            pybvh_ml.rotate_vertical, getattr(aug_cfg, "rotate_prob", 1.0),
            {"angle": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
             "up_axis": up_axis, "degrees": True},
        ))

    if getattr(aug_cfg, "mirror", False):
        lr_pairs = [tuple(p) for p in skeleton.lr_joint_pairs]
        mirror_kwargs = {"lr_joint_pairs": lr_pairs,
                         "lateral_axis": lateral_axis}
        if "node_pos" in position_fields:
            # Node space includes end sites, so it has its own left/right
            # pairing; the dataset recorded it at preprocessing time.
            mirror_kwargs["lr_node_pairs"] = [
                tuple(p) for p in skeleton_info["node_lr_pairs"]]
        steps.append((
            pybvh_ml.mirror, aug_cfg.mirror_prob,
            mirror_kwargs,
        ))

    if getattr(aug_cfg, "speed", False):
        lo, hi = aug_cfg.speed_range
        steps.append((
            pybvh_ml.speed_perturbation_arrays, getattr(aug_cfg, "speed_prob", 1.0),
            {"factor": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi)},
        ))

    noise_sigma = getattr(aug_cfg, "noise_sigma", 0.0)
    if noise_sigma > 0:
        # Rotation noise only. pybvh-ml >= 0.5 split the old fused
        # add_joint_noise into a rotation form and a root-position form
        # because the two sigmas are in different units (radians vs. the
        # data's length unit). This config surface only ever set the
        # rotation sigma, so the split is a rename here — root jitter would
        # be a new config key, not a silent side effect of this one.
        noise_kwargs = {"sigma": noise_sigma, "degrees": True}
        if position_fields:
            # With positions in the sample, this step refreshes them by
            # forward kinematics from the noised rotations (pybvh-ml >= 0.6)
            # — that keeps rotation-noise semantics identical between
            # rotation-stream and position-stream trainings, rather than
            # substituting an unrelated keypoint jitter. The topology was
            # recorded at preprocessing time, so no BVH file is needed here.
            noise_kwargs["fk_topology"] = pybvh_ml.build_fk_topology(
                skeleton_info)
            if skeleton_info.get("world_up"):
                # Only consulted under "first" centering, harmless otherwise.
                noise_kwargs["world_up"] = skeleton_info["world_up"]
        steps.append((
            pybvh_ml.add_joint_rotation_noise, 1.0,
            noise_kwargs,
        ))

    if getattr(aug_cfg, "dropout", False):
        steps.append((
            pybvh_ml.dropout_arrays, aug_cfg.dropout_prob,
            {"drop_rate": aug_cfg.dropout_rate},
        ))

    if not steps:
        return None
    # Preprocessing stores data as quaternions, and every augmentation runs in
    # that space. Declared once on the pipeline (pybvh-ml >= 0.5) rather than
    # repeated on all five steps: five copies of the same token is a
    # copy-paste surface where one step ends up disagreeing with the rest,
    # and a disagreement is a silent mid-pipeline representation change.
    return pybvh_ml.AugmentationPipeline(steps, representation="quat")


def _check_streams_shape(cfg, streams, target_repr, skeleton_info=None):
    """Cross-check data.streams against model.in_channels / skeleton.num_nodes.

    A stream list fully determines the packed tensor's channel count (each
    positional/derived stream is 3 channels, ``joint_rot`` is the target
    representation's width) and, when the dataset's joint count is known,
    its vertex count too. A mismatch otherwise surfaces as a shape error in
    the middle of the first forward pass, far from the config that caused it.
    """
    repr_channels = {"euler": 3, "quat": 4, "6d": 6, "axisangle": 3,
                     "rotmat": 9}
    effective = tuple(streams) if streams else ("root_pos", "joint_rot")
    # "root_pos" becomes vertex 0 (zero-padded on the channel axis when the
    # other streams are wider), so it adds a vertex, never channels.
    expected_c = sum(repr_channels[target_repr] if s == "joint_rot" else 3
                     for s in effective if s != "root_pos")
    if expected_c != cfg.model.in_channels:
        raise ValueError(
            f"data.streams={list(effective)} with target_repr="
            f"{target_repr!r} packs {expected_c} channels, but "
            f"model.in_channels is {cfg.model.in_channels}. Fix one of them."
        )
    num_joints = (skeleton_info or {}).get("num_joints")
    if num_joints:
        vertex_streams = [s for s in effective if s != "root_pos"]
        space = "node" if any(s.startswith("node") for s in vertex_streams) \
            else "joint"
        base_v = (skeleton_info or {}).get("num_nodes") \
            if space == "node" else num_joints
        if base_v:
            expected_v = base_v + ("root_pos" in effective)
            if expected_v != cfg.skeleton.num_nodes:
                raise ValueError(
                    f"data.streams={list(effective)} packs {expected_v} "
                    f"vertices ({space} space"
                    f"{' + root vertex' if 'root_pos' in effective else ''}) "
                    f"but skeleton.num_nodes is {cfg.skeleton.num_nodes}. "
                    f"The skeleton block must describe the packed vertex "
                    f"space — without 'root_pos' in the streams there is no "
                    f"root vertex and inward_edges must be the joint-space "
                    f"edge list (skeleton_info['edges'])."
                )


def _file_digest(path, chunk=1 << 23):
    """Cheap fingerprint of a dataset file: size plus three sampled chunks.

    Not a full hash. These datasets are 1-2 GB and this runs once per fold, so
    digesting every byte would add roughly a minute per cross-validation sweep
    to answer a question that does not need cryptographic strength. The job here
    is only to tell two corpus builds apart — the 2026-08-06 DIEM-A re-export
    changed 65 clip lengths and a whole nationality's rotations, which any of
    head, middle or tail will catch. Two files that collide on all three and on
    total size are the same build for our purposes.

    Returns None rather than raising: provenance is a convenience, and a missing
    or unreadable dataset will fail loudly a moment later in the loader.
    """
    try:
        p = Path(path)
        size = p.stat().st_size
        h = hashlib.sha256(str(size).encode())
        with p.open("rb") as f:
            for offset in (0, max(0, size // 2 - chunk // 2), max(0, size - chunk)):
                f.seek(offset)
                h.update(f.read(chunk))
        return f"{h.hexdigest()[:16]}:{size}"
    except OSError:
        return None


def resolve_init_path(path, fold):
    """Fill the `{fold}` placeholder in an --init-from path, zero-padded.

    Warm restarts are per-fold: fold 3 must resume from the checkpoint that
    never saw fold 3's performers during training. One path with a placeholder
    expresses that; ten paths on the command line invite an off-by-one.
    """
    if "{fold}" not in path:
        return path
    if fold is None:
        raise ValueError("--init-from uses {fold} but --fold was not given")
    return path.format(fold=f"{fold:02d}")


def load_init_weights(lit_model, path, fold=None):
    """Copy a checkpoint's weights into a freshly built model, nothing else.

    Returns the resolved path. Everything the checkpoint knows about the
    *optimisation* — momentum buffers, LR scheduler position, epoch counter —
    is deliberately discarded, which is what separates a warm restart from a
    resume. A resumed cosine that has already annealed sits at ~1e-6 and the
    run does nothing.
    """
    resolved = resolve_init_path(path, fold)
    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    missing, unexpected = lit_model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        # Loud, not fatal: metric-buffer keys legitimately differ across
        # versions, but a silent architecture mismatch would quietly train a
        # half-initialised network and report the result as a warm restart.
        print(f"[init-from] missing keys: {missing}")
        print(f"[init-from] unexpected keys: {unexpected}")
    print(f"[init-from] loaded weights from {resolved} "
          f"(epoch {ckpt.get('epoch', '?')}); optimizer and schedule are fresh")
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Train an emotion recognition model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument("--test-after", action="store_true", help="Run test after training")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold number for LPO cross-validation (1-indexed)")
    parser.add_argument("--num-folds", type=int, default=None,
                        help="Total number of LPO folds")
    parser.add_argument("--init-from", default=None, metavar="CKPT",
                        help="Start from the WEIGHTS of a checkpoint, but train as a "
                             "fresh run: new optimizer, new LR schedule, epoch counter "
                             "back to zero. Use {fold} as a placeholder for the fold "
                             "number, zero-padded to two digits. This is deliberately "
                             "NOT Lightning's --resume: resuming restores the optimizer "
                             "and scheduler state, which would continue the old cosine "
                             "instead of starting a new one.")
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)

    # Seed every global RNG before anything is constructed. Without this,
    # data.seed only ever reached the Feeder's augmentation: weight init,
    # dropout masks and the DataLoader's shuffle order came from whatever
    # entropy torch picked at import, so a "seed 42 vs seed 255" sweep was
    # not reproducible and `deterministic=True` had nothing to pin down.
    # workers=True makes Lightning seed each DataLoader worker too.
    pl.seed_everything(cfg.data.seed, workers=True)

    # Determine split: on-the-fly LPO or from config
    use_lpo = args.fold is not None or args.num_folds is not None
    if use_lpo:
        if args.fold is None or args.num_folds is None:
            parser.error("--fold and --num-folds must be used together")
        if args.fold < 1 or args.fold > args.num_folds:
            parser.error(f"--fold must be between 1 and {args.num_folds}")
        split_dict = build_lpo_split(cfg.data.data_path, args.fold, args.num_folds)
        split_dict = _apply_performer_fraction(cfg, split_dict)
        split_path = None
    else:
        split_dict = None
        split_path = cfg.data.split_path

    # Build model
    model_cls = get_model(cfg.model.type)
    model = model_cls.from_config(cfg)

    # Build augmentation pipeline. The dataset's own skeleton metadata
    # (pybvh-ml >= 0.5 records world_up / rest_forward) cross-checks the
    # config's axis declarations, so a stale config can't silently rotate
    # the skeleton about the wrong axis.
    streams = getattr(cfg.data, "streams", None)
    target_repr = getattr(cfg.data, "target_repr", "euler")
    skeleton_info = pybvh_ml.load_preprocessed(
        cfg.data.data_path
    )["skeleton_info"] if cfg.augmentation.enabled else {}
    pipeline = _build_pipeline(cfg, skeleton_info, streams=streams)

    # Fail before the first forward pass, naming the config keys, when the
    # requested streams cannot produce the tensor the model was built for.
    # The vertex space is the trap: no "root_pos" in the streams means no
    # root vertex, V = J, and a 25-node skeleton block silently indexes the
    # wrong joints. The channel count is arithmetic over the stream list.
    _check_streams_shape(cfg, streams, target_repr, skeleton_info)

    # Build data module
    loader = Loader(
        data_path=cfg.data.data_path,
        split_path=split_path,
        split_dict=split_dict,
        clip_length=cfg.training.clip_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        target_repr=target_repr,
        seed=cfg.data.seed,
        augmentation_pipeline=pipeline,
        streams=streams,
        scale_normalize=cfg.data.scale_normalize,
    )

    # Build Lightning model
    aux_loss_weights = cfg.training.aux_loss_weights
    if isinstance(aux_loss_weights, dict):
        pass  # already a dict
    else:
        aux_loss_weights = vars(aux_loss_weights) if hasattr(aux_loss_weights, "__dict__") else {}

    lit_model = LightningModel(
        model=model,
        base_lr=cfg.training.base_lr,
        num_class=cfg.model.num_class,
        optimizer=cfg.training.optimizer,
        scheduler_type=cfg.training.scheduler_type,
        scheduler_params=cfg.training.scheduler_params,
        weight_decay=cfg.training.weight_decay,
        aux_loss_weights=aux_loss_weights,
        label_smoothing=cfg.training.label_smoothing,
        # Which corpus, which representation, which streams. Two DIEM-A builds
        # exist on disk and differ by about a point of accuracy; a checkpoint
        # that does not say which one it consumed cannot be compared to another
        # without reconstructing the command that produced it.
        provenance={
            "data_path": str(cfg.data.data_path),
            "data_sha256": _file_digest(cfg.data.data_path),
            "target_repr": str(target_repr),
            "streams": list(streams) if streams else None,
            "scale_normalize": bool(cfg.data.scale_normalize),
            "seed": cfg.data.seed,
        },
    )

    # Warm start: take the weights, leave everything else behind.
    #
    # The point of this flag is to ask what a *converged* model does when it is
    # given a second training cycle. That question only makes sense if the new
    # cycle is genuinely new -- fresh momentum buffers, a fresh cosine that
    # starts at base_lr and anneals to zero over the new max_epochs, epoch
    # counter at zero. Lightning's `fit(ckpt_path=...)` does the opposite: it
    # restores the optimizer and scheduler so the old schedule picks up where
    # it stopped, which for a finished cosine means training at ~1e-6 forever.
    if args.init_from:
        load_init_weights(lit_model, args.init_from, args.fold)

    # Experiment name
    log_cfg = cfg.logging
    experiment_name = log_cfg.experiment_name
    if experiment_name is None:
        if use_lpo:
            experiment_name = f"{cfg.model.type}_fold{args.fold:02d}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{cfg.model.type}_{timestamp}"

    # Resolve {seed}/{fold} placeholders in log_dir so a single config can
    # drive multi-seed sweeps and per-fold output trees without YAML duplication.
    resolved_log_dir = interpolate_log_dir(log_cfg.log_dir, cfg.data.seed, args.fold)

    # Loggers and callbacks
    # Compute the next version ourselves so both loggers share one directory
    # (otherwise CSV and TensorBoard each auto-increment independently).
    log_base = Path(resolved_log_dir) / experiment_name
    existing_versions = [
        int(m.group(1))
        for d in (log_base.iterdir() if log_base.exists() else [])
        if (m := re.match(r"^version_(\d+)$", d.name))
    ]
    version = max(existing_versions, default=-1) + 1

    csv_logger = CSVLogger(save_dir=resolved_log_dir, name=experiment_name, version=version)
    tb_logger = TensorBoardLogger(save_dir=resolved_log_dir, name=experiment_name, version=version)

    ckpt_cfg = cfg.checkpointing
    ckpt_callbacks, ckpt_by_preset = _build_checkpoint_callbacks(ckpt_cfg.save)
    # EpochSeedCallback tells the train Feeder which epoch it is, so the
    # (seed, epoch, idx) augmentation draws actually vary across epochs.
    callbacks = [EpochSeedCallback(), *ckpt_callbacks]
    if cfg.training.early_stopping:
        es_monitor = cfg.training.early_stopping_monitor
        # Infer mode from metric name: anything containing "loss" → minimize,
        # anything containing "acc" or similar → maximize.
        es_mode = "min" if "loss" in es_monitor else "max"
        callbacks.append(EarlyStopping(
            monitor=es_monitor,
            mode=es_mode,
            patience=cfg.training.early_stopping_patience,
            verbose=True,
        ))

    # Numeric policy (TF32 + mixed precision) lives in tools/runtime.py, which
    # emo-evaluate and emo-predict call too. It used to be set here and nowhere
    # else, so the same checkpoint scored differently depending on which
    # command you ran it through.
    precision = configure_training_runtime(cfg)

    grad_clip = getattr(cfg.training, "gradient_clip_val", 0.0) or None
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        deterministic=True,
        accelerator=cfg.training.accelerator,
        devices=getattr(cfg.training, "devices", "auto"),
        precision=precision,
        gradient_clip_val=grad_clip,
    )

    fit_start = time.time()
    trainer.fit(lit_model, datamodule=loader)
    fit_duration = time.time() - fit_start
    # Lightning increments current_epoch at the end of each epoch, so after
    # fit it already *is* the number of completed epochs. Snapshot it here:
    # trainer.test() below runs its own loop and moves the counter.
    epochs_run = trainer.current_epoch

    # Snapshot checkpoint paths/scores BEFORE trainer.test(): Lightning's
    # test phase can reset these in-memory attributes on ModelCheckpoint
    # (the files on disk are intact, but best_model_path can be cleared).
    snapshot_paths = {
        name: (cb.last_model_path if name == "last" else cb.best_model_path)
        for name, cb in ckpt_by_preset.items()
    }
    snapshot_scores = {
        name: (None if name == "last" else cb.best_model_score)
        for name, cb in ckpt_by_preset.items()
    }

    if args.test_after:
        # Test on a *separate* Trainer at the evaluation precision, not the
        # training one. A Trainer carries a single precision, so reusing the
        # fit trainer would score this checkpoint in bf16-mixed while
        # emo-evaluate scores it in fp32 — the two commands would disagree on
        # the same file. Carries no ModelCheckpoint callbacks either, so the
        # test loop cannot touch the checkpoint bookkeeping — strictly safer
        # than the snapshot dance above, which exists because the fit trainer
        # *could*. Lightning warns that the callbacks used to create the
        # checkpoint are absent; that only affects restoring *callback state*,
        # which a test-only pass has no use for, so the warning is expected
        # here rather than a sign of something missing.
        eval_precision = configure_eval_runtime(cfg)
        test_trainer = pl.Trainer(
            logger=[csv_logger, tb_logger],
            deterministic=True,
            accelerator=cfg.training.accelerator,
            devices=getattr(cfg.training, "devices", "auto"),
            precision=eval_precision,
        )
        test_ckpt = _resolve_test_ckpt_path(ckpt_cfg.test_with, ckpt_by_preset)
        if test_ckpt:
            print(f"Testing on checkpoint ({ckpt_cfg.test_with}, "
                  f"precision={eval_precision}): {test_ckpt}")
            test_trainer.test(lit_model, datamodule=loader, ckpt_path=test_ckpt)
        else:
            test_trainer.test(lit_model, datamodule=loader)

    for name, path in snapshot_paths.items():
        score = snapshot_scores[name]
        score_str = f" (score={float(score):.4f})" if score is not None else ""
        print(f"Saved checkpoint [{name}]: {path}{score_str}")

    # Compute-report line in a parseable format so multi-seed aggregators can
    # ingest it straight from the per-job log.
    if torch.cuda.is_available():
        device_str = torch.cuda.get_device_name(0)
    else:
        device_str = f"CPU ({platform.processor() or platform.machine()})"
    print(
        f"COMPUTE: fit_seconds={fit_duration:.1f} "
        f"epochs={epochs_run} "
        f"device={device_str!r} "
        f"precision={precision} "
        f"batch_size={cfg.training.batch_size}"
    )


if __name__ == "__main__":
    main()
