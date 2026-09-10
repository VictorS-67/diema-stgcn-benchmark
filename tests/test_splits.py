"""Tests for LPO cross-validation split generation."""

import pytest

from emo_mocap.data.splits import (
    fold_of_clip,
    generate_lpo_splits,
    parse_diema_actor,
    subsample_train_performers,
)


# ---------------------------------------------------------------------------
# parse_diema_actor
# ---------------------------------------------------------------------------

class TestParseDiemaActor:
    def test_standard_filename(self):
        assert parse_diema_actor("JP_06_anger_1_H") == "JP_06"

    def test_taiwanese_actor(self):
        assert parse_diema_actor("TW_30_joy_2_M") == "TW_30"

    def test_augmented_filename(self):
        """Augmented filenames (from --augment-copies) keep the same actor ID."""
        assert parse_diema_actor("JP_06_anger_1_H_aug00") == "JP_06"
        assert parse_diema_actor("TW_30_joy_2_M_aug02") == "TW_30"

    def test_multi_word_emotion_still_works(self):
        """Actor ID is always the first two parts, regardless of what follows."""
        assert parse_diema_actor("JP_12_surprise_3_L") == "JP_12"


# ---------------------------------------------------------------------------
# generate_lpo_splits
# ---------------------------------------------------------------------------

def _make_filenames(actor_clips):
    """Helper: create DIEMA-style filenames from a dict of {actor: num_clips}.

    Returns a flat list of filenames in the order they'd appear in an npz.
    """
    filenames = []
    for actor, n in actor_clips.items():
        nat, pid = actor.split("_")
        for i in range(n):
            filenames.append(f"{nat}_{pid}_anger_{i}_H")
    return filenames


class TestGenerateLpoSplits:
    def test_basic_three_actors_three_folds(self):
        """Three actors, three folds: each split has one actor per train/val/test."""
        filenames = _make_filenames({"JP_01": 3, "JP_02": 4, "JP_03": 5})
        splits = generate_lpo_splits(filenames, num_folds=3)

        assert len(splits) == 3

        for split in splits:
            test_actors = {parse_diema_actor(f) for f, _ in split["test"]}
            val_actors = {parse_diema_actor(f) for f, _ in split["val"]}
            train_actors = {parse_diema_actor(f) for f, _ in split["train"]}
            # Three disjoint sets, covering all three actors.
            assert test_actors.isdisjoint(val_actors)
            assert test_actors.isdisjoint(train_actors)
            assert val_actors.isdisjoint(train_actors)
            assert test_actors | val_actors | train_actors == {"JP_01", "JP_02", "JP_03"}

    def test_k_fold_grouping(self):
        """Six actors into 3 folds: 2 actors per fold, val/test disjoint."""
        actors = {"JP_01": 2, "JP_02": 2, "JP_03": 2,
                  "TW_01": 2, "TW_02": 2, "TW_03": 2}
        filenames = _make_filenames(actors)
        splits = generate_lpo_splits(filenames, num_folds=3)

        assert len(splits) == 3

        for split in splits:
            test_actors = {parse_diema_actor(f) for f, _ in split["test"]}
            val_actors = {parse_diema_actor(f) for f, _ in split["val"]}
            train_actors = {parse_diema_actor(f) for f, _ in split["train"]}
            # No actor overlap between any pair.
            assert test_actors.isdisjoint(train_actors)
            assert val_actors.isdisjoint(train_actors)
            assert test_actors.isdisjoint(val_actors)
            assert len(test_actors) == 2 and len(val_actors) == 2

    def test_split_dict_format(self):
        """Each split dict has the correct format for Loader."""
        filenames = _make_filenames({"JP_01": 3, "JP_02": 4, "JP_03": 5})
        splits = generate_lpo_splits(filenames, num_folds=3)

        for split in splits:
            assert set(split.keys()) == {"train", "val", "test"}
            for key in ("train", "val", "test"):
                for entry in split[key]:
                    assert isinstance(entry, tuple)
                    assert len(entry) == 2
                    assert isinstance(entry[0], str)
                    assert isinstance(entry[1], int)
            # New contract: val and test are strictly disjoint.
            val_names = {f for f, _ in split["val"]}
            test_names = {f for f, _ in split["test"]}
            assert val_names.isdisjoint(test_names)

    def test_all_clips_covered_by_test_sets(self):
        """Every clip appears in exactly one fold's test set."""
        filenames = _make_filenames({"JP_01": 3, "JP_02": 4, "JP_03": 5})
        splits = generate_lpo_splits(filenames, num_folds=3)

        all_test_indices = []
        for split in splits:
            all_test_indices.extend(idx for _, idx in split["test"])
        assert sorted(all_test_indices) == list(range(len(filenames)))

    def test_every_actor_serves_as_val_exactly_once(self):
        """Rotating val: each actor is in exactly one fold's val."""
        actors = {f"JP_{i:02d}": 1 for i in range(6)}
        filenames = _make_filenames(actors)
        splits = generate_lpo_splits(filenames, num_folds=6)

        val_counts = {}
        for split in splits:
            for f, _ in split["val"]:
                actor = parse_diema_actor(f)
                val_counts[actor] = val_counts.get(actor, 0) + 1
        assert set(val_counts.values()) == {1}

    def test_fold_balance(self):
        """7 actors into 3 folds → groups of [3, 2, 2] actors (round-robin)."""
        actors = {f"JP_{i:02d}": 2 for i in range(7)}
        filenames = _make_filenames(actors)
        splits = generate_lpo_splits(filenames, num_folds=3)

        fold_actor_counts = [
            len({parse_diema_actor(f) for f, _ in s["test"]})
            for s in splits
        ]
        # Round-robin: 7 actors, 3 folds → [3, 2, 2]
        assert sorted(fold_actor_counts, reverse=True) == [3, 2, 2]

    def test_deterministic(self):
        """Same inputs produce identical splits."""
        filenames = _make_filenames({"JP_01": 3, "JP_02": 4, "TW_01": 5})
        splits_a = generate_lpo_splits(filenames, num_folds=3)
        splits_b = generate_lpo_splits(filenames, num_folds=3)

        for a, b in zip(splits_a, splits_b):
            assert a["train"] == b["train"]
            assert a["val"] == b["val"]
            assert a["test"] == b["test"]

    def test_num_folds_too_small(self):
        filenames = _make_filenames({"JP_01": 3, "JP_02": 3, "JP_03": 3})
        with pytest.raises(ValueError, match="num_folds must be >= 3"):
            generate_lpo_splits(filenames, num_folds=2)

    def test_num_folds_exceeds_actors(self):
        filenames = _make_filenames({"JP_01": 3, "JP_02": 4, "JP_03": 3})
        with pytest.raises(ValueError, match="exceeds number of unique actors"):
            generate_lpo_splits(filenames, num_folds=5)

    def test_train_val_test_indices_are_valid(self):
        """All indices in all three splits reference valid filename positions."""
        filenames = _make_filenames({"JP_01": 3, "JP_02": 4, "TW_01": 5})
        splits = generate_lpo_splits(filenames, num_folds=3)

        for split in splits:
            for key in ("train", "val", "test"):
                for fname, idx in split[key]:
                    assert 0 <= idx < len(filenames)
                    assert filenames[idx] == fname


class TestFoldOfClip:
    """Mapping individual clips to their fold's test set."""

    def test_clip_lands_in_its_actors_fold(self):
        filenames = _make_filenames({f"JP_{i:02d}": 2 for i in range(6)})
        splits = generate_lpo_splits(filenames, num_folds=3)
        for clip, _ in splits[0]["test"]:
            assert fold_of_clip(clip, filenames, num_folds=3) == 1
        for clip, _ in splits[1]["test"]:
            assert fold_of_clip(clip, filenames, num_folds=3) == 2

    def test_unknown_actor_raises(self):
        filenames = _make_filenames({"JP_01": 1, "JP_02": 1, "JP_03": 1})
        with pytest.raises(KeyError, match="not in the dataset"):
            fold_of_clip("ZZ_99_anger_1_M", filenames, num_folds=3)


# ---------------------------------------------------------------------------
# subsample_train_performers
# ---------------------------------------------------------------------------

def _split_with(n_actors, clips_per_actor=6):
    """A split whose train half holds ``n_actors`` performers."""
    train = [(f"JP_{a:02d}_anger_{c}_H", a * clips_per_actor + c)
             for a in range(n_actors) for c in range(clips_per_actor)]
    return {
        "train": train,
        "val": [("TW_90_joy_1_H", 900)],
        "test": [("TW_91_fear_1_H", 901)],
    }


class TestSubsampleTrainPerformers:
    def test_keeps_the_right_number_of_performers(self):
        split = _split_with(20)
        out = subsample_train_performers(split, 0.5, seed=1)
        assert len({parse_diema_actor(f) for f, _ in out["train"]}) == 10

    def test_keeps_every_clip_of_a_kept_performer(self):
        """By actor, never by clip — otherwise 'fewer people' and 'less data'
        are confounded, which is the one thing the experiment separates."""
        split = _split_with(20, clips_per_actor=6)
        out = subsample_train_performers(split, 0.25, seed=1)
        from collections import Counter
        counts = Counter(parse_diema_actor(f) for f, _ in out["train"])
        assert set(counts.values()) == {6}

    def test_val_and_test_are_untouched(self):
        split = _split_with(20)
        out = subsample_train_performers(split, 0.25, seed=1)
        assert out["val"] == split["val"]
        assert out["test"] == split["test"]

    def test_input_is_not_mutated(self):
        split = _split_with(20)
        before = len(split["train"])
        subsample_train_performers(split, 0.25, seed=1)
        assert len(split["train"]) == before

    def test_fractions_nest(self):
        """0.25's cohort must be a subset of 0.5's, else each arm differs in
        *which* performers it got as well as how many, and a hard-actor draw
        masquerades as a size effect."""
        split = _split_with(20)
        cohorts = {}
        for frac in (0.25, 0.5, 0.75, 1.0):
            out = subsample_train_performers(split, frac, seed=7)
            cohorts[frac] = {parse_diema_actor(f) for f, _ in out["train"]}
        assert cohorts[0.25] < cohorts[0.5] < cohorts[0.75] < cohorts[1.0]

    def test_deterministic_across_calls(self):
        split = _split_with(20)
        a = subsample_train_performers(split, 0.5, seed=3)["train"]
        b = subsample_train_performers(split, 0.5, seed=3)["train"]
        assert a == b

    def test_different_seeds_pick_different_cohorts(self):
        split = _split_with(20)
        a = {parse_diema_actor(f)
             for f, _ in subsample_train_performers(split, 0.5, seed=1)["train"]}
        b = {parse_diema_actor(f)
             for f, _ in subsample_train_performers(split, 0.5, seed=2)["train"]}
        assert a != b

    def test_fraction_one_is_a_noop(self):
        split = _split_with(20)
        assert subsample_train_performers(split, 1.0)["train"] == split["train"]

    def test_never_empties_the_training_set(self):
        split = _split_with(3)
        out = subsample_train_performers(split, 0.01, seed=1)
        assert len({parse_diema_actor(f) for f, _ in out["train"]}) == 1

    @pytest.mark.parametrize("bad", [0, -0.5, 1.5])
    def test_rejects_out_of_range_fractions(self, bad):
        with pytest.raises(ValueError, match="fraction"):
            subsample_train_performers(_split_with(10), bad)
