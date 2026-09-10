"""Ensemble softmax probabilities from N prediction CSVs.

Reads N CSVs produced by ``emo-predict --per-class-columns`` (one per
ensembled predictor — typically one per seed), averages the softmax vectors,
and emits a single CSV with the ensemble argmax plus uncertainty stats.

Usage::

    emo-ensemble \\
        --inputs preds_seed255.csv preds_seed42.csv preds_seed7.csv \\
        --output ensemble.csv \\
        --class-names configs/emo_to_idx.txt

Output columns::

    sample_name, true_label, ensemble_pred, agreement_count,
    entropy_of_mean, mean_entropy,
    mean_<c0>, mean_<c1>, ..., std_<c0>, std_<c1>, ...

* ``entropy_of_mean`` — H(mean softmax); how uncertain the ensemble is.
* ``mean_entropy``    — mean over predictors of H(softmax_i); individual uncertainty.
* ``agreement_count`` — how many input predictors' argmax matched the ensemble argmax.

The tool is task-agnostic: it infers the number of classes from the CSV
columns and falls back to integer class names (``mean_0`` etc.) if no
``--class-names`` file is supplied.
"""

import argparse
import csv
import math
from pathlib import Path


def _load_predictions(path):
    """Read a per-class-columns prediction CSV.

    Returns (rows, num_class) where each row is a dict with keys
    ``sample_name``, ``true_label`` (int), ``proba`` (list of floats).
    """
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        proba_cols = [c for c in fields if c.startswith("proba_")]
        if not proba_cols:
            raise ValueError(
                f"{path}: no proba_* columns found. "
                "Run emo-predict with --per-class-columns to produce this format."
            )
        # Sort numerically so 'proba_10' comes after 'proba_9'
        proba_cols.sort(key=lambda c: int(c.split("_", 1)[1]))

        rows = []
        for raw in reader:
            rows.append({
                "sample_name": raw["sample_name"],
                "true_label": int(raw["true_label"]),
                "proba": [float(raw[c]) for c in proba_cols],
            })
    return rows, len(proba_cols)


def _load_class_names(path, num_class):
    """Read an optional class-names file. Supports two formats:

    * one name per line (line number = class index)
    * ``name idx`` per line (e.g. ``configs/emo_to_idx.txt``)

    Returns a list of ``num_class`` strings. Falls back to ``["0","1",...]``
    when ``path`` is None.
    """
    if path is None:
        return [str(i) for i in range(num_class)]

    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]

    # Try 'name idx' format first (all lines must parse)
    pairs = []
    is_pair_format = True
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            is_pair_format = False
            break
        try:
            pairs.append((parts[0], int(parts[1])))
        except ValueError:
            is_pair_format = False
            break

    if is_pair_format:
        pairs.sort(key=lambda p: p[1])
        names = [n for n, _ in pairs]
    else:
        names = lines

    if len(names) < num_class:
        raise ValueError(
            f"Class-names file {path} has {len(names)} names but CSV has "
            f"{num_class} classes."
        )
    return names[:num_class]


def _entropy(p):
    """Shannon entropy (nats). Zeros are skipped to avoid 0*log(0)."""
    return -sum(v * math.log(v) for v in p if v > 0)


def ensemble_predictions(inputs_rows_list, num_class):
    """Compute per-sample ensemble stats from per-predictor row lists.

    All inputs must cover the same set of sample_names. Order is taken from
    the first input. Per-sample true_labels must also agree across inputs.

    Returns a list of dicts in the order of the first input.
    """
    if not inputs_rows_list:
        raise ValueError("ensemble_predictions: no inputs provided")

    anchor = inputs_rows_list[0]
    anchor_names = [r["sample_name"] for r in anchor]

    lookups = []
    for rows in inputs_rows_list:
        d = {r["sample_name"]: r for r in rows}
        if set(d) != set(anchor_names):
            missing = set(anchor_names) - set(d)
            extra = set(d) - set(anchor_names)
            raise ValueError(
                f"Input CSVs disagree on sample set: "
                f"{len(missing)} missing, {len(extra)} extra."
            )
        lookups.append(d)

    n = len(lookups)
    results = []
    for name in anchor_names:
        true_label = lookups[0][name]["true_label"]
        for L in lookups[1:]:
            if L[name]["true_label"] != true_label:
                raise ValueError(
                    f"Sample {name!r}: inputs disagree on true_label."
                )

        all_probas = [L[name]["proba"] for L in lookups]
        mean = [sum(p[k] for p in all_probas) / n for k in range(num_class)]
        if n > 1:
            var = [
                sum((p[k] - mean[k]) ** 2 for p in all_probas) / (n - 1)
                for k in range(num_class)
            ]
        else:
            var = [0.0] * num_class
        std = [math.sqrt(v) for v in var]

        ensemble_pred = max(range(num_class), key=lambda k: mean[k])
        agreement = sum(
            1 for p in all_probas
            if max(range(num_class), key=lambda k: p[k]) == ensemble_pred
        )

        results.append({
            "sample_name": name,
            "true_label": true_label,
            "ensemble_pred": ensemble_pred,
            "agreement_count": agreement,
            "entropy_of_mean": _entropy(mean),
            "mean_entropy": sum(_entropy(p) for p in all_probas) / n,
            "mean": mean,
            "std": std,
        })
    return results


def write_ensemble_csv(results, class_names, out_file):
    """Serialize ensemble results to an open file-like object."""
    writer = csv.writer(out_file)
    mean_cols = [f"mean_{c}" for c in class_names]
    std_cols = [f"std_{c}" for c in class_names]
    writer.writerow([
        "sample_name", "true_label", "ensemble_pred", "agreement_count",
        "entropy_of_mean", "mean_entropy",
        *mean_cols, *std_cols,
    ])
    for r in results:
        writer.writerow([
            r["sample_name"], r["true_label"], r["ensemble_pred"], r["agreement_count"],
            f"{r['entropy_of_mean']:.6f}", f"{r['mean_entropy']:.6f}",
            *[f"{v:.6f}" for v in r["mean"]],
            *[f"{v:.6f}" for v in r["std"]],
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble softmax probabilities from N per-class-columns prediction CSVs."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input CSVs (one per ensembled predictor; produced by "
             "emo-predict --per-class-columns)",
    )
    parser.add_argument("--output", required=True, help="Output ensemble CSV path")
    parser.add_argument(
        "--class-names", default=None,
        help="Optional file with one class name per line, or 'name idx' pairs "
             "(e.g. configs/emo_to_idx.txt). Renames mean_/std_ columns.",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("--inputs requires at least 2 CSVs to ensemble.")

    inputs_rows_list = []
    num_class_seen = None
    for p in args.inputs:
        rows, nc = _load_predictions(p)
        if num_class_seen is None:
            num_class_seen = nc
        elif nc != num_class_seen:
            parser.error(f"{p}: has {nc} classes; expected {num_class_seen}.")
        inputs_rows_list.append(rows)

    results = ensemble_predictions(inputs_rows_list, num_class_seen)
    class_names = _load_class_names(args.class_names, num_class_seen)

    with open(args.output, "w", newline="") as f:
        write_ensemble_csv(results, class_names, f)

    correct = sum(1 for r in results if r["ensemble_pred"] == r["true_label"])
    print(f"Ensemble of {len(args.inputs)} predictors on {len(results)} samples.")
    print(f"  Accuracy: {correct}/{len(results)} = {correct / len(results):.4f}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
