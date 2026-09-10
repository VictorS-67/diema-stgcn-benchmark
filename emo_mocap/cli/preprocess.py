"""Preprocess BVH files into npz format for training.

Stores data as quaternions (pybvh's ``"quat"`` representation token) — the
canonical intermediate representation. At training time, the Feeder converts
to the target representation (Euler, 6D, etc.) specified in the config.

Only BVH files whose emotion appears in the emo2idx mapping are included.
This means the same raw BVH directory can produce different datasets
(e.g., 7-class or 13-class) by using different emo2idx files.

Usage:
    python -m emo_mocap.cli.preprocess --input data/raw/diema_bvh/ \
        --output data/processed/diema7_quat.npz \
        --emo2idx configs/emo_to_idx_7.txt

    With Bvh-level augmentation:
    python -m emo_mocap.cli.preprocess --input data/raw/diema_bvh/ \
        --output data/processed/diema7_quat.npz \
        --emo2idx configs/emo_to_idx_7.txt \
        --augment-copies 3 --augment-speed-range 0.8 1.2 --augment-dropout-rate 0.1
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pybvh_ml


def _load_emo2idx(path):
    """Load emotion-to-index mapping from a text file.

    Expected format: one 'emotion index' pair per line.
    """
    emo2idx = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                emo2idx[parts[0]] = int(parts[1])
    return emo2idx


def _parse_diema_emotion(filename_stem):
    """Extract emotion string from DIEMA filename.

    Format: {nationality}_{performerID}_{emotion}_{scenario}_{intensity}
    Example: JP_06_anger_2_M → 'anger'
    """
    parts = filename_stem.split("_")
    return parts[2]


def _diema_label_fn(filename_stem, emo2idx):
    """Extract emotion label index from DIEMA filename."""
    emotion = _parse_diema_emotion(filename_stem)
    return emo2idx[emotion]


def _diema_filter_fn(filename_stem, emo2idx):
    """Return True if the file's emotion is in the mapping."""
    emotion = _parse_diema_emotion(filename_stem)
    return emotion in emo2idx


def _augment_and_preprocess(input_dir, output_path, emo2idx, copies,
                            speed_range, dropout_rate, seed=0,
                            file_pattern="*.bvh", **preprocess_kwargs):
    """Preprocess with Bvh-level augmentation (speed perturbation, frame dropout).

    Filters, then generates augmented copies of included BVH files into a temp
    directory, and preprocesses everything at once.

    Augmentation draws come from one seeded ``np.random.Generator`` threaded
    through both the speed factor and pybvh's ``drop_frames``, so re-running
    the command reproduces the dataset bit for bit.
    """
    import shutil
    import pybvh
    from pybvh.transforms import perturb_speed, drop_frames

    rng = np.random.default_rng(seed)

    # Collect valid files
    valid_paths = [
        p for p in sorted(input_dir.glob(file_pattern))
        if _diema_filter_fn(p.stem, emo2idx)
    ]

    with tempfile.TemporaryDirectory() as aug_dir:
        aug_path = Path(aug_dir)

        for bvh_path in valid_paths:
            # Copy original. The staging directory is flat even when the input
            # tree is nested — DIEMA filenames are globally unique, and the
            # downstream preprocess_directory call globs it non-recursively.
            shutil.copy2(bvh_path, aug_path / bvh_path.name)

            # Generate augmented copies
            for i in range(copies):
                bvh = pybvh.read_bvh_file(bvh_path)

                if speed_range is not None:
                    factor = rng.uniform(*speed_range)
                    bvh = perturb_speed(bvh, factor)

                if dropout_rate is not None and dropout_rate > 0:
                    bvh = drop_frames(bvh, dropout_rate, rng=rng)

                # overwrite=False (pybvh >= 0.8.1) turns a stem collision into
                # an error instead of a silently smaller dataset: the staging
                # directory is flat, so two source files sharing a stem under
                # --recursive would otherwise clobber each other's copies.
                aug_stem = f"{bvh_path.stem}_aug{i:02d}"
                bvh.write(str(aug_path / f"{aug_stem}.bvh"),
                          overwrite=False, verbose=False)

        label_fn = lambda stem: _diema_label_fn(stem, emo2idx)
        return pybvh_ml.preprocess_directory(
            bvh_dir=aug_path,
            output_path=output_path,
            representation="quat",
            center_root=True,
            label_fn=label_fn,
            **preprocess_kwargs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BVH files into npz format (quaternion representation)"
    )
    parser.add_argument("--input", required=True, help="Directory containing BVH files")
    parser.add_argument("--recursive", action="store_true",
                        help="Search --input recursively. DIEMA ships one "
                             "subdirectory per performer (data/raw/bvh/JP_06/...), "
                             "which the default flat glob would miss entirely.")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--emo2idx", required=True,
                        help="Path to emotion-to-index mapping file")
    parser.add_argument("--augment-copies", type=int, default=0,
                        help="Number of augmented copies per sample (0 = disabled)")
    parser.add_argument("--augment-speed-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Speed perturbation range (e.g., 0.8 1.2)")
    parser.add_argument("--augment-dropout-rate", type=float, default=None,
                        help="Frame dropout rate for augmentation (e.g., 0.1)")
    parser.add_argument("--augment-seed", type=int, default=0,
                        help="Seed for the Bvh-level augmentation draws "
                             "(speed factor, frame dropout). Same seed = same "
                             "augmented dataset.")
    parser.add_argument("--target-fps", type=float, default=None, metavar="HZ",
                        help="Resample every clip to HZ before extraction "
                             "(DIEMA is 120 Hz; training runs at 30). Absolute "
                             "rate, not a factor, so a mixed-rate directory "
                             "lands on one rate rather than N different ones.")
    parser.add_argument("--target-world-up", default=None, metavar="AXIS",
                        help="Signed axis string (e.g. '+z'). When set, pybvh-ml "
                             "reorients every clip's world up to AXIS before "
                             "extracting rotations, fixing axis drift in outlier "
                             "files. DIEMA: '+z'.")
    parser.add_argument("--target-rest-forward", default=None, metavar="AXIS",
                        help="Signed axis string (e.g. '+y'). Matches "
                             "target-world-up — both should be set together.")
    parser.add_argument("--target-rest-up", default=None, metavar="AXIS",
                        help="Signed axis string (e.g. '+z'). Reorients rest "
                             "pose up so that quaternions extracted from outlier "
                             "files match topology-identical ones.")
    parser.add_argument("--include-positions", action="store_true",
                        help="Also store per-vertex positions (pybvh-ml >= 0.6) "
                             "computed by forward kinematics at preprocessing "
                             "time. Required for position-stream training "
                             "(data.streams with joint_pos / node_pos / "
                             "joint_vel / ...).")
    parser.add_argument("--position-space", default="joint",
                        choices=["joint", "node"],
                        help="Vertex space for the stored positions: 'joint' "
                             "shares its vertex axis with the rotations (24 on "
                             "DIEMA); 'node' includes end sites — fingertips, "
                             "toe tips, head top — and pairs with the dataset's "
                             "node_edges.")
    parser.add_argument("--position-centering", default="skeleton",
                        choices=["world", "skeleton", "first"],
                        help="Frame the stored positions are expressed in. "
                             "'skeleton' (root at the origin every frame) is "
                             "what NTU-style models are fed; 'world' keeps them "
                             "coherent with root_pos for geometry checks. Note "
                             "'first' is rejected alongside center_root=True, "
                             "which this CLI always sets.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    emo2idx = _load_emo2idx(args.emo2idx)
    file_pattern = "**/*.bvh" if args.recursive else "*.bvh"

    if not any(input_dir.glob(file_pattern)):
        hint = "" if args.recursive else " (try --recursive)"
        print(f"No BVH files found in {input_dir}{hint}")
        return

    # Axis harmonisation and resampling apply to both paths — a reorientation
    # that only ran on the un-augmented dataset would silently produce two
    # datasets with different world frames.
    #
    # target_fps resamples inside pybvh-ml, *before* rotations are extracted
    # (SLERP for rotations, linear for root position). That ordering is what
    # makes it correct rather than merely convenient: velocities and foot
    # contacts are derived from the resampled clip, so they describe the
    # motion at the target rate. Decimating the finished .npz — what this CLI
    # used to do — cannot reproduce that, because a finite-difference
    # velocity's stencil baseline is set by the original frame_time.
    common_kwargs = dict(
        target_fps=args.target_fps,
        target_world_up=args.target_world_up,
        target_rest_forward=args.target_rest_forward,
        target_rest_up=args.target_rest_up,
        include_positions=args.include_positions,
        position_space=args.position_space,
        position_centering=args.position_centering,
    )

    if args.augment_copies > 0:
        result = _augment_and_preprocess(
            input_dir, output_path, emo2idx,
            args.augment_copies, args.augment_speed_range,
            args.augment_dropout_rate, seed=args.augment_seed,
            file_pattern=file_pattern, **common_kwargs,
        )
    else:
        label_fn = lambda stem: _diema_label_fn(stem, emo2idx)
        filter_fn = lambda stem: _diema_filter_fn(stem, emo2idx)
        result = pybvh_ml.preprocess_directory(
            bvh_dir=input_dir,
            output_path=output_path,
            representation="quat",
            center_root=True,
            label_fn=label_fn,
            filter_fn=filter_fn,
            file_pattern=file_pattern,
            **common_kwargs,
        )

    print(f"Saved {result['num_clips']} clips to {output_path}")
    print(f"Representation: {result['representation']}")
    # The skeleton axes are what the training-time augmentation reads to know
    # which way is up; print them so a mismatch is visible at build time.
    skel = result["skeleton_info"]
    print(f"Skeleton axes: world_up={skel.get('world_up')} "
          f"rest_forward={skel.get('rest_forward')} "
          f"rest_up={skel.get('rest_up')}")
    # The uniformity audit reports the *source* frame rates, so a directory
    # that wasn't the uniform 120 Hz it was assumed to be is visible here
    # rather than only in the training curve.
    source_fps = sorted((result.get("uniformity") or {}).get("fps", {}))
    if source_fps:
        print(f"Source frame rate(s): {', '.join(f'{f} Hz' for f in source_fps)}")
    if args.target_fps is not None:
        print(f"Resampled to: {args.target_fps} Hz")
    if args.include_positions:
        print(f"Positions: space={skel.get('position_space')} "
              f"centering={result.get('position_centering')} "
              f"num_joints={skel.get('num_joints')} "
              f"num_nodes={skel.get('num_nodes')}")


if __name__ == "__main__":
    main()
