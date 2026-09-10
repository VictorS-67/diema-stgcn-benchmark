"""Feeder dataset for loading preprocessed motion capture data.

Loads rotation data from pybvh-ml's npz/hdf5 format, applies augmentation
in the stored representation, converts to the target representation, and
packs to (C, T, V) tensors for the model.

Representation vocabulary follows pybvh 0.8 / pybvh-ml 0.5: the short
tokens ``euler`` / ``quat`` / ``6d`` / ``axisangle`` / ``rotmat``. There is
no alias layer for the retired ``"quaternion"`` spelling — a dataset still
carrying it predates the 0.8 rotation-math consolidation and should be
regenerated with ``emo-preprocess`` (see ``check_repr``).
"""

import warnings

import numpy as np
import torch
import torch.utils.data

import pybvh_ml
from pybvh_ml.torch import EpochState, rng_for


# pybvh 0.8 shortened the representation vocabulary; there is no alias layer.
# Datasets and configs written against the old vocabulary are rejected rather
# than silently translated — a dataset stamped "quaternion" predates the
# 0.8 rotation-math consolidation, and re-preprocessing is the honest fix.
_RETIRED_REPR_TOKENS = {"quaternion": "quat"}

# Derived temporal streams (pybvh-ml >= 0.6) are computed by the packer from
# an already-augmented position stream — they are never stored per clip, so
# requesting one means the clip must carry its *base* stream.
_DERIVED_BASES = {"joint_vel": "joint_pos", "joint_acc": "joint_pos",
                  "node_vel": "node_pos", "node_acc": "node_pos"}
_POSITION_FIELDS = ("joint_pos", "node_pos")

# Which edge list measures each position field's skeleton. Bone lengths are
# what `scale_normalize` divides by, and the two position spaces have
# different topologies (node space adds the end sites).
_EDGE_KEY = {"joint_pos": "edges", "node_pos": "node_edges"}


def skeleton_scale(positions, edges):
    """Anatomical size of one performer: the sum of their bone lengths.

    Read off frame 0 of a position stream — bone lengths are fixed by the
    BVH offsets, so any frame gives the same answer and no averaging is
    needed. Deliberately *not* a motion statistic (spread of the joints,
    range of the trajectory): those grow when a performer moves expansively,
    and expansive movement is exactly the emotion signal we are trying to
    classify. Dividing by a motion statistic would normalise away the label.

    Args:
        positions: (F, V, 3) position stream, any centering — a per-frame
            translation cancels in the parent-child difference.
        edges: (E, 2) array of (child, parent) vertex indices.

    Returns:
        float: total bone length, in the dataset's own units.
    """
    p = positions[0]                                    # (V, 3)
    child, parent = edges[:, 0], edges[:, 1]
    return float(np.linalg.norm(p[child] - p[parent], axis=-1).sum())


def check_repr(repr_name, source):
    """Reject a retired representation token, naming the fix.

    ``"quaternion"`` became ``"quat"`` in pybvh 0.8 / pybvh-ml 0.5. Rather
    than map it, we fail: a dataset still stamped with the old token was
    written by an older pipeline, and re-running ``emo-preprocess`` is what
    the caller actually wants. Every current token passes through untouched
    (pybvh-ml validates the vocabulary itself).
    """
    if repr_name in _RETIRED_REPR_TOKENS:
        raise ValueError(
            f"{source} uses the retired representation token "
            f"{repr_name!r} (renamed to {_RETIRED_REPR_TOKENS[repr_name]!r} "
            f"in pybvh 0.8 / pybvh-ml 0.5). Re-run emo-preprocess to "
            f"regenerate the dataset, or update the config."
        )
    return repr_name


class Feeder(torch.utils.data.Dataset):
    """PyTorch Dataset that loads preprocessed skeleton sequences.

    The pipeline per sample:
    1. Load rotation data from the dataset into a ``pybvh_ml.MotionArrays``
       (root_pos + joint_rot)
    2. Augment in the stored representation (train mode only)
    3. Convert to target representation (Euler, 6D, ...)
    4. Pack to (C, T, V) layout
    5. Temporal sample to fixed clip_length

    Augmentation draws come from a ``(seed, epoch, idx)`` seed sequence, so a
    given sample's augmentation is reproducible and independent of batch
    order and worker count. Call :meth:`set_epoch` at the start of each
    training epoch (the ``EpochSeedCallback`` in ``emo_mocap.data.loader``
    does this) so the draws change from epoch to epoch.

    Args:
        data_path: path to the .npz / .hdf5 file (pybvh-ml format)
        indices: optional list of clip indices to select a subset
        clip_length: number of frames to sample per sequence (default: 64)
        target_repr: target rotation representation for model input
            ('euler', '6d', 'quat', 'axisangle', 'rotmat')
        test: if True, use deterministic sampling and skip augmentation
        seed: base seed for reproducible augmentation (default: 255)
        augmentation_pipeline: optional pybvh_ml.AugmentationPipeline
            (only applied in training mode, unless ``augment_in_test``)
        augment_in_test: apply the pipeline even when ``test=True``, while
            keeping temporal sampling deterministic. For test-time
            augmentation only — see ``scripts/tta_eval.py``. Leave False for
            any number that is going to be reported as a plain metric.
        stochastic_crop_in_test: draw a random temporal window when
            ``test=True`` instead of the deterministic centre-anchored one.
            The other half of test-time augmentation: vary it with
            ``set_epoch(k)`` to get a different crop per pass. Leave False for
            reported metrics — evaluation must be reproducible.
        euler_orders: per-joint Euler orders (required when converting to or
            from 'euler'; defaults to the dataset's stored orders)
        streams: which streams the packed tensor carries, in channel order
            (pybvh-ml >= 0.6), e.g. ``("joint_pos",)`` for the NTU-style
            ``(3, T, J)`` input or ``("joint_pos", "joint_vel", "joint_acc")``
            for a 9-channel one. ``None`` keeps the historical default —
            root translation as vertex 0 plus joint rotations, ``V = 1 + J``.
            Without ``"root_pos"`` in the tuple there is no root vertex:
            ``V = J`` and the model's graph must use the joint-space edge
            list (``skeleton_info["edges"]``). Position streams require a
            dataset preprocessed with ``--include-positions``; the derived
            ``*_vel`` / ``*_acc`` streams are differenced by the packer from
            the *augmented* positions (never stored), which is what makes a
            speed-perturbed sample's velocity come out rescaled for free.
        scale_normalize: divide every position stream by the performer's own
            skeleton size, so a tall and a short performer producing the same
            gesture produce the same numbers. Positions carry body size in
            every channel; rotations do not, which is one candidate
            explanation for why position input trails rotation input by ~10
            points (Track E). Ignored when no position stream is packed —
            there is nothing to rescale.
    """

    def __init__(self, data_path, indices=None, clip_length=64,
                 target_repr="euler", test=False, seed=255,
                 augmentation_pipeline=None, euler_orders=None,
                 augment_in_test=False, stochastic_crop_in_test=False,
                 streams=None, scale_normalize=False):
        preprocessed = pybvh_ml.load_preprocessed(data_path)
        self.clips = preprocessed["clips"]
        self.labels = preprocessed.get("labels")
        self.filenames = preprocessed.get("filenames", [])
        self.skeleton_info = preprocessed.get("skeleton_info", {})

        # The representation the file was written in. pybvh-ml stores it in
        # the dataset metadata, so we don't have to assume quaternions —
        # a 6D-preprocessed dataset now feeds the Feeder unchanged.
        self.source_repr = check_repr(
            preprocessed.get("representation") or "quat", f"Dataset {data_path}"
        )

        # Whether the stored root_pos is already first-frame-centered.
        # pybvh-ml >= 0.5 records this; a missing flag means the dataset
        # predates that and needs regenerating, which check_repr already
        # catches for any dataset this project produced.
        self.stored_center_root = preprocessed.get("center_root")

        # Which frame the stored positions are expressed in ("world" /
        # "skeleton" / "first"); recorded by pybvh-ml >= 0.6 and threaded
        # onto every container we build, because augmentation correctness
        # depends on it (root jitter is a genuine no-op under "skeleton").
        self.position_centering = preprocessed.get("position_centering")

        self.streams = tuple(streams) if streams else None
        # The container only carries the position fields the requested
        # streams actually need: a rotation-only run on a positions-carrying
        # dataset should not pay the FK-refresh cost in the noise step, and
        # a derived stream (joint_vel) needs its base present, not packed.
        needed = {_DERIVED_BASES.get(s, s) for s in (self.streams or ())}
        self._position_fields = tuple(f for f in _POSITION_FIELDS if f in needed)
        missing = [f for f in self._position_fields
                   if self.clips and f not in self.clips[0]]
        if missing:
            raise ValueError(
                f"data.streams={self.streams} needs {missing} but dataset "
                f"{data_path} does not carry them. Re-run emo-preprocess "
                f"with --include-positions (and --position-space "
                f"{'node' if 'node_pos' in missing else 'joint'})."
            )

        # Per-clip rescaling factors, computed over *all* clips before the
        # subset below. The reference is the corpus-wide mean skeleton, so
        # the packed numbers keep their original order of magnitude and the
        # train and test Feeders divide by the same constant — a reference
        # taken from each Feeder's own subset would silently give the two
        # splits different units.
        self._scale_factors = None
        if scale_normalize and self._position_fields:
            field = self._position_fields[0]
            edges = np.asarray(self.skeleton_info[_EDGE_KEY[field]])
            sizes = np.array([skeleton_scale(c[field], edges) for c in self.clips])
            self._scale_factors = sizes.mean() / sizes

        if indices is not None:
            self.clips = [self.clips[i] for i in indices]
            if self._scale_factors is not None:
                self._scale_factors = self._scale_factors[indices]
            if self.labels is not None:
                self.labels = self.labels[indices]
            if self.filenames:
                self.filenames = [self.filenames[i] for i in indices]

        self.clip_length = clip_length
        self.target_repr = check_repr(target_repr, "data.target_repr")
        self.mode = "test" if test else "train"
        self.pipeline = augmentation_pipeline
        self.augment_in_test = augment_in_test
        self.stochastic_crop_in_test = stochastic_crop_in_test
        self.seed = seed
        # pybvh-ml >= 0.5 exposes the epoch counter and the (seed, epoch, idx)
        # generator publicly, so the Feeder holds the library's own rather
        # than a copy. The counter lives in shared memory: a plain attribute
        # is pickled once when DataLoader workers start, so a later
        # set_epoch() in the main process would be invisible to them and
        # persistent workers would replay epoch 0's augmentation forever.
        self._epoch_state = EpochState()
        # Warn-once bookkeeping, deliberately a plain attribute: each
        # DataLoader worker inherits False and gets its own warning, and a
        # flag pickled as "already warned" is how this class of bug stays
        # silent. Whether set_epoch() was *ever* called is NOT tracked here —
        # see _rng_for.
        self._warned_no_epoch = False

        # Euler orders from skeleton_info (needed for quat<->Euler conversion)
        self.euler_orders = euler_orders or self.skeleton_info.get("euler_orders")

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation epoch.

        Mirrors ``DistributedSampler.set_epoch`` / ``MotionDataset.set_epoch``:
        call it at the start of every training epoch so the same sample index
        draws a different augmentation each epoch. Without it, all epochs
        replay epoch -1's draws — useful for debugging, harmful for training.
        """
        self._epoch_state.set(epoch)

    def _rng_for(self, idx):
        """Per-sample RNG derived from ``(seed, epoch, idx)``.

        Deriving from the index rather than advancing one shared stream makes
        a sample's augmentation independent of the order the DataLoader
        happens to visit it in, and of how many workers are running.
        ``pybvh_ml.torch.rng_for`` is that scheme; the Feeder only supplies
        the epoch and the missing-``set_epoch`` warning.

        "Was ``set_epoch`` ever called" has to be read from ``EpochState``'s
        shared ``-1`` sentinel, not from a plain attribute on this object.
        Lightning spawns DataLoader workers *before* ``on_train_epoch_start``
        fires, so every worker would pickle a not-yet-set flag and warn on
        every single run — the augmentation meanwhile being perfectly correct,
        because the epoch counter itself is shared memory and the workers do
        observe it. A warning that cries wolf on every run is how a real
        occurrence becomes invisible.

        ``_raw()`` is pybvh-ml private API; the public ``current`` collapses
        "never set" and "epoch 0" to the same 0. Worth asking upstream for a
        public ``is_set`` — this is the only thing standing between here and
        a clean read.
        """
        if self._epoch_state._raw() < 0:
            if (self.pipeline is not None and self.mode == "train"
                    and not self._warned_no_epoch):
                self._warned_no_epoch = True
                warnings.warn(
                    "Feeder.set_epoch() was never called; every epoch will "
                    "replay the same augmentation per sample. Add "
                    "emo_mocap.data.loader.EpochSeedCallback to the Trainer "
                    "(emo-train does this) or call set_epoch() yourself.",
                    RuntimeWarning, stacklevel=3,
                )
        return rng_for(self.seed, self._epoch_state.current, idx)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self.clips)
        if not 0 <= idx < len(self.clips):
            raise IndexError(f"index {idx} out of range for {len(self.clips)} clips")

        clip = self.clips[idx]
        rng = self._rng_for(idx)

        # The streams travel together as a MotionArrays (pybvh-ml >= 0.5):
        # one clip's root translation plus its joint rotations — and, when
        # the requested streams need them, the position streams with their
        # recorded centering. Rotations ride along even for position-only
        # streams: augmenting them and letting the noise step's FK refresh
        # regenerate the positions keeps the augmentation protocol identical
        # between rotation and position trainings, which is what makes the
        # two comparable.
        #
        # No defensive copy of the cached clip: the container exposes its
        # fields as *read-only* views, so nothing downstream can write back
        # through them into `self.clips`. Construction allocates nothing,
        # which matters here — this runs once per sample per epoch.
        fields = {f: clip[f] for f in self._position_fields}
        if fields:
            fields["position_centering"] = self.position_centering
        arrays = pybvh_ml.MotionArrays(
            root_pos=clip["root_pos"],      # (F, 3)
            joint_rot=clip["joint_rot"],    # (F, J, C_source)
            **fields,                       # joint_pos/node_pos: (F, V, 3)
        )

        # 1. Augment in the stored representation.
        #
        # Normally train-mode only — evaluation must see the data as it is.
        # ``augment_in_test`` is the one sanctioned exception: test-time
        # augmentation, where the *same* clip is scored under several fixed
        # transforms and the predictions averaged. That needs the pipeline to
        # run while temporal sampling stays deterministic, so the only thing
        # varying between passes is the transform under study. It is opt-in
        # precisely because silently augmenting an evaluation split is how a
        # metric stops meaning what it says.
        if self.pipeline is not None and (self.mode == "train"
                                          or self.augment_in_test):
            arrays = self.pipeline(arrays, rng=rng)

        # 1b. Divide out the performer's body size. This has to happen *after*
        # augmentation, not before: the noise step perturbs rotations and
        # regenerates the positions by forward kinematics from the unscaled
        # offsets, which would undo the rescaling. Every augmentation step
        # preserves bone lengths, so the factor computed at init is still the
        # right one here. The root trajectory is scaled alongside the joints —
        # stride length is a body-size quantity too, and leaving it in the
        # original units would put two streams on two different scales.
        if self._scale_factors is not None:
            f = float(self._scale_factors[idx])
            scaled = {name: getattr(arrays, name) * f
                      for name in self._position_fields}
            arrays = arrays.replace(root_pos=arrays.root_pos * f, **scaled)

        # 2. Convert to target representation — only when a rotation stream
        # is actually packed: positions have no representation, and skipping
        # the conversion avoids demanding euler_orders from a run that never
        # looks at the rotations again. Works on the container, so the other
        # streams are carried through rather than unpacked and reassembled
        # (`convert_rotations` is the bare-array form, for data with no root).
        packs_rotations = self.streams is None or "joint_rot" in self.streams
        if packs_rotations and self.target_repr != self.source_repr:
            arrays = pybvh_ml.convert_arrays(
                arrays, self.source_repr, self.target_repr,
                euler_orders=self.euler_orders,
            )  # joint_rot: (F, J, C_target)

        # 3. Pack to CTV. Re-centering an already-centered clip is what we
        # must avoid: pybvh-ml records whether the stored root_pos was
        # centered at preprocessing time, so only center when it wasn't.
        # streams=None keeps 0.5.0's exact default packing (root vertex 0 +
        # rotations), byte-identically — so it is omitted, not spelled out.
        pack_kwargs = {} if self.streams is None else {"streams": self.streams}
        data_ctv = pybvh_ml.pack_to_ctv(
            arrays, center_root=self.stored_center_root is False,
            **pack_kwargs,
        )

        # 4. Temporal sample. In test mode pybvh-ml honours a caller-supplied
        # rng (it silently ignored it before 0.5), so pass None to keep the
        # deterministic centre-anchored draw that evaluation depends on —
        # unless a caller has explicitly asked for stochastic crops, which is
        # temporal test-time augmentation and varies with set_epoch().
        num_frames = data_ctv.shape[1]
        deterministic_crop = self.mode == "test" and not self.stochastic_crop_in_test
        frame_indices = pybvh_ml.uniform_temporal_sample(
            num_frames, self.clip_length, mode=self.mode,
            rng=None if deterministic_crop else rng,
        )
        frame_indices = frame_indices % num_frames
        data_ctv = data_ctv[:, frame_indices, :]  # (C, clip_length, V)

        # 5. Return
        label = int(self.labels[idx]) if self.labels is not None else -1
        filename = self.filenames[idx] if self.filenames else ""

        return (
            torch.tensor(data_ctv, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            filename,
        )
