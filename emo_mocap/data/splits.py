"""LPO (Leave-Performers-Out) cross-validation split generation.

Generates K-fold splits with THREE disjoint actor groups per fold:

    test  = actors in the fold-k group (rotating)
    val   = actors in the fold-(k+1 mod K) group (rotating, shifted by one)
    train = everything else

This is stricter than the original design (which had ``val == test``): it
means val metrics are reported on actors the training never saw AND that
aren't in the test set, so checkpoint selection on val_acc can't
over-fit to the test population.

Every actor serves as test in exactly one fold and as val in exactly one
other fold. ``clip_actors_map`` can be consulted to discover which
fold's test set contains a given clip (e.g. the 69 PLD clips used for
the ACII 2026 interpretability analysis).

DIEMA filename format:
    {nationality}_{performerID}_{emotion}_{scenarioId}_{intensity}
    Example: JP_06_anger_1_H  →  actor ID = JP_06

Augmented filenames (e.g. JP_06_anger_1_H_aug00) are handled correctly:
the actor ID is always the first two underscore-separated parts.
"""

from collections import defaultdict

import numpy as np
import pybvh_ml


def parse_diema_actor(filename_stem: str) -> str:
    """Extract actor ID from a DIEMA filename stem.

    Args:
        filename_stem: filename without extension, e.g. 'JP_06_anger_1_H'
            or 'TW_30_joy_2_M_aug02'

    Returns:
        Actor ID string, e.g. 'JP_06' or 'TW_30'
    """
    parts = filename_stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def generate_lpo_splits(
    filenames: list[str],
    num_folds: int,
    actor_fn=parse_diema_actor,
) -> list[dict]:
    """Generate K-fold LPO splits with disjoint train / val / test.

    Steps:
    1. Extract actor IDs from filenames via actor_fn
    2. Sort unique actors alphabetically
    3. Distribute actors round-robin into num_folds groups
    4. For fold k: test = group_k, val = group_{(k+1) mod K}, train = rest

    The rotating val is shifted by exactly one group so that every actor
    serves as test in one fold and as val in exactly one other — no
    privileged val or test actors across the sweep.

    Round-robin ensures balanced folds. With 92 actors and 10 folds,
    folds 0-1 get 10 actors each, folds 2-9 get 9 each.

    Args:
        filenames: list of filename stems (one per clip in the dataset)
        num_folds: number of folds (K)
        actor_fn: callable that extracts an actor ID from a filename stem.
            Defaults to parse_diema_actor (DIEMA convention).

    Returns:
        List of K split dicts. Each dict has keys 'train', 'val', 'test'.
        Each value is a list of (filename, original_index) tuples,
        matching the format expected by Loader. The three lists are
        mutually disjoint by actor.
    """
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")

    # Group clip indices by actor
    actor_to_indices = defaultdict(list)
    for idx, fname in enumerate(filenames):
        actor_id = actor_fn(fname)
        actor_to_indices[actor_id].append(idx)

    # Sort actors alphabetically for determinism
    sorted_actors = sorted(actor_to_indices.keys())

    # val and test each consume one actor-group per fold, so we need at
    # least 3 groups (otherwise train is empty or val == test). This is
    # a stricter lower bound than the old val==test design.
    if num_folds < 3:
        raise ValueError(
            f"num_folds must be >= 3 for disjoint train/val/test, got {num_folds}"
        )
    if num_folds > len(sorted_actors):
        raise ValueError(
            f"num_folds ({num_folds}) exceeds number of unique actors "
            f"({len(sorted_actors)})"
        )

    # Round-robin assignment: actor i goes to fold (i % num_folds)
    fold_actors = [[] for _ in range(num_folds)]
    for i, actor in enumerate(sorted_actors):
        fold_actors[i % num_folds].append(actor)

    # Build split dicts
    splits = []
    for fold_idx in range(num_folds):
        test_actors = set(fold_actors[fold_idx])
        val_actors = set(fold_actors[(fold_idx + 1) % num_folds])
        train_entries, val_entries, test_entries = [], [], []

        for idx, fname in enumerate(filenames):
            actor_id = actor_fn(fname)
            entry = (fname, idx)
            if actor_id in test_actors:
                test_entries.append(entry)
            elif actor_id in val_actors:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

        splits.append({
            "train": train_entries,
            "val": val_entries,
            "test": test_entries,
        })

    return splits


def subsample_train_performers(
    split: dict,
    fraction: float,
    seed: int = 255,
    actor_fn=parse_diema_actor,
) -> dict:
    """Keep only a fraction of the training *performers*, all of their clips.

    For a performer learning curve, i.e. how accuracy scales with the number
    of people in the training set:
    train on 25 / 50 / 75 / 100% of the actors and read the slope at 100%.
    If test accuracy is still climbing there, the binding constraint is how
    many people the model has seen, and no amount of augmentation or schedule
    tuning substitutes for collecting more.

    **Subsampling is by actor, never by clip.** Dropping random clips would
    reduce the amount of data *and* leave every performer represented, which
    confounds the two things this experiment exists to separate: does the
    model need more examples, or more *people*? Keeping every clip of a
    smaller cohort isolates the second.

    Val and test are returned untouched — they are the measuring instrument
    and must not vary between arms.

    Selection is seeded and sorted, so a given ``(fraction, seed)`` picks the
    same actors on every machine and every rerun. It is also *nested*: the
    actors chosen at 0.25 are a subset of those at 0.5, because the draw is a
    prefix of one shuffled order. Without nesting, each arm would differ both
    in cohort size and in which particular performers it got, and a
    difficult-actor draw would masquerade as a size effect.

    Args:
        split: a split dict from :func:`build_lpo_split`, with ``"train"`` a
            list of ``(filename, index)`` pairs.
        fraction: share of training actors to keep, in (0, 1].
        seed: seeds the actor shuffle.
        actor_fn: filename stem -> actor ID.

    Returns:
        A new split dict; the input is not modified.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return dict(split)

    actors = sorted({actor_fn(fname) for fname, _ in split["train"]})
    n_keep = max(1, round(fraction * len(actors)))

    # A prefix of one shuffled order, so smaller fractions nest inside larger.
    order = list(np.random.default_rng(seed).permutation(len(actors)))
    keep = {actors[i] for i in order[:n_keep]}

    return {
        **split,
        "train": [(f, i) for f, i in split["train"] if actor_fn(f) in keep],
    }


def fold_of_clip(
    clip_name: str,
    filenames: list[str],
    num_folds: int,
    actor_fn=parse_diema_actor,
) -> int:
    """Return the fold index (1-indexed) whose *test* split contains ``clip_name``.

    Used by the ACII 2026 interpretability workflow to route each PLD
    clip to the model checkpoint that never saw its performer during
    training. Raises ``KeyError`` if the clip's actor doesn't appear in
    the dataset (e.g. misspelt or from an excluded intensity).

    Args:
        clip_name: filename stem (no extension)
        filenames: the full list of filenames in the preprocessed dataset
            (so the fold assignment matches what ``generate_lpo_splits``
            produces on the live data)
        num_folds: same K passed to ``generate_lpo_splits``
        actor_fn: actor extractor (default DIEMA convention)

    Returns:
        Fold index in ``[1, num_folds]``.
    """
    target_actor = actor_fn(clip_name)
    sorted_actors = sorted({actor_fn(f) for f in filenames})
    if target_actor not in sorted_actors:
        raise KeyError(
            f"Actor {target_actor!r} (from clip {clip_name!r}) is not in the dataset"
        )
    actor_idx = sorted_actors.index(target_actor)
    # Round-robin: actor at position i lands in fold (i % num_folds) → 1-indexed.
    return (actor_idx % num_folds) + 1


def build_lpo_split(data_path, fold: int, num_folds: int) -> dict:
    """Generate the LPO split dict for a single fold from a preprocessed npz.

    Loads filenames from the dataset, generates all K splits deterministically,
    and returns the split dict for the requested fold (1-indexed, matching
    the CLI convention).
    """
    preprocessed = pybvh_ml.load_preprocessed(data_path)
    filenames = list(preprocessed.get("filenames", []))
    if not filenames:
        raise ValueError(
            f"No filenames found in {data_path}. "
            "Cannot generate LPO splits without filename metadata."
        )
    all_splits = generate_lpo_splits(filenames, num_folds)
    return all_splits[fold - 1]
