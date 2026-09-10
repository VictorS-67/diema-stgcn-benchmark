"""Per-emotion recall, precision and F1, plus the pooled confusion matrix.

An aggregate accuracy hides the shape of the problem, and two very different
situations produce the same headline: every emotion mediocre, or half of them
good and half at chance. On this corpus it is the former, with the errors
concentrated between emotions that are neighbours in valence and arousal, so
the confusion matrix is the more informative artefact.

Counts are pooled over folds and seeds, because a single fold holds too few
clips per class to read. With an unbalanced label set, read balanced accuracy,
the mean of the per-class recalls, next to plain accuracy.

Example:

    python scripts/per_class_analysis.py \
        --config configs/diema7_stgcn_recipe.yaml \
        --logs runs/diema7/seed255 --variants recipe \
        --labels configs/emo_to_idx_7.txt --out runs/per_class.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from emo_mocap.data.feeder import Feeder
from emo_mocap.data.splits import build_lpo_split
from emo_mocap.models.registry import get_model
from emo_mocap.tools.config import load_config_with_overrides
from emo_mocap.tools.runtime import configure_eval_runtime
from emo_mocap.training.lightning_model import LightningModel


def _load_labels(path):
    """Emotion names in index order, from the canonical mapping."""
    names = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name, idx = line.rsplit(maxsplit=1)
        names[int(idx)] = name.strip()
    return [names[i] for i in sorted(names)]


@torch.no_grad()
def _predict(model, cfg, indices, device):
    feeder = Feeder(
        data_path=cfg.data.data_path, indices=indices,
        clip_length=cfg.training.clip_length, target_repr=cfg.data.target_repr,
        seed=cfg.data.seed, test=True,
    )
    loader = torch.utils.data.DataLoader(
        feeder, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers or 0, pin_memory=True)
    preds, labels = [], []
    for x, y, _n in loader:
        preds.append(model(x.to(device))["logits"].argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--logs", required=True)
    p.add_argument("--variants", nargs="*", default=[""],
                   help="Subdirectories under --logs (e.g. seeds); '' for flat")
    p.add_argument("--folds", type=int, default=10,
                   help="Size of the LPO scheme — defines the splits, so it "
                        "cannot be shrunk for a trial run.")
    p.add_argument("--limit-folds", type=int, default=None,
                   help="Evaluate only the first N folds (trial runs).")
    p.add_argument("--labels", default="configs/emo_to_idx.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)
    configure_eval_runtime(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = cfg.model.num_class
    names = _load_labels(args.labels)[:K]

    confusion = np.zeros((K, K), dtype=np.int64)  # [true, predicted]
    per_variant = {}
    started = time.time()

    for variant in args.variants:
        vc = np.zeros((K, K), dtype=np.int64)
        for fold in range(1, min(args.limit_folds or args.folds, args.folds) + 1):
            fold_dir = (Path(args.logs) / variant / f"fold{fold:02d}" if variant
                        else Path(args.logs) / f"fold{fold:02d}")
            ckpts = sorted(fold_dir.glob("*/version_*/checkpoints/*.ckpt"))
            if not ckpts:
                print(f"{variant}/fold{fold:02d}: no checkpoint", flush=True)
                continue

            split = build_lpo_split(cfg.data.data_path, fold, args.folds)
            model = get_model(cfg.model.type).from_config(cfg)
            lit = LightningModel.load_from_checkpoint(
                str(ckpts[-1]), model=model, base_lr=cfg.training.base_lr,
                num_class=K, map_location=device)

            preds, labels = _predict(lit.model.eval().to(device),
                                     cfg, [i for _, i in split["test"]], device)
            for t, q in zip(labels.tolist(), preds.tolist()):
                vc[t, q] += 1
            print(f"{variant or 'default':>6s} fold{fold:02d} done", flush=True)

        confusion += vc
        per_variant[variant or "default"] = vc.tolist()

    support = confusion.sum(axis=1)
    tp = np.diag(confusion)
    recall = np.divide(tp, support, out=np.zeros(K), where=support > 0)
    predicted = confusion.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(K), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(K), where=denom > 0)
    chance = 1.0 / K

    print(f"\n{'emotion':>12s} {'recall':>8s} {'prec':>8s} {'F1':>8s} "
          f"{'support':>8s}  {'vs chance':>9s}")
    for i, name in enumerate(names):
        print(f"{name:>12s} {recall[i]*100:7.1f}% {precision[i]*100:7.1f}% "
              f"{f1[i]*100:7.1f}% {support[i]:8d}  {recall[i]/chance:8.2f}x")
    print(f"\noverall accuracy {tp.sum()/confusion.sum()*100:.2f}%   "
          f"macro-F1 {f1.mean()*100:.2f}%   chance {chance*100:.1f}%")
    print(f"per-class recall: min {recall.min()*100:.1f}% "
          f"max {recall.max()*100:.1f}%  spread {(recall.max()-recall.min())*100:.1f} pp")

    # The most-confused ordered pairs: what the errors actually are.
    off = confusion.astype(float).copy()
    np.fill_diagonal(off, 0)
    rate = np.divide(off, support[:, None], out=np.zeros_like(off),
                     where=support[:, None] > 0)
    pairs = sorted(((rate[i, j], i, j) for i in range(K) for j in range(K) if i != j),
                   reverse=True)[:6]
    print("\nmost-confused pairs (share of the true class sent elsewhere):")
    for r, i, j in pairs:
        print(f"  {names[i]:>12s} -> {names[j]:<12s} {r*100:5.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_meta": {"logs": args.logs, "variants": args.variants,
                  "config": args.config, "labels": names},
        "confusion_total": confusion.tolist(),
        "confusion_per_variant": per_variant,
        "recall": recall.tolist(), "precision": precision.tolist(),
        "f1": f1.tolist(), "support": support.tolist(),
    }, indent=1))
    print(f"\nwrote {out}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
