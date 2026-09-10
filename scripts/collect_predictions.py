"""Cache the per-clip test probabilities of an already-trained run.

Averaging several seeds' predictions is the cheapest accuracy you can buy: it
costs no training at all and is worth about four points on this task. It needs
the probability vector each model assigns to each clip, which this script
extracts from checkpoints already on disk and stores as a small ``.npz`` of
``names``, ``labels`` and ``probs``.

Because leave-performer-out gives each clip exactly one fold whose model never
saw its performer, the cache is assembled fold by fold and covers the corpus
once. Feed several of them to ``emo-ensemble`` or average them yourself.

Example:

    python scripts/collect_predictions.py \
        --config configs/diema7_stgcn_recipe.yaml \
        --logs runs/diema7/seed255 --variants recipe \
        --tag seed255 --out-dir runs/predictions
"""

import argparse
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


@torch.no_grad()
def _probs(model, cfg, indices, device):
    feeder = Feeder(
        data_path=cfg.data.data_path, indices=indices,
        clip_length=cfg.training.clip_length, target_repr=cfg.data.target_repr,
        seed=cfg.data.seed,
        # Load-bearing: a position-stream config scored without this silently
        # reads rotations. The same omission was live in
        # score_folds.py.
        streams=getattr(cfg.data, "streams", None),
        scale_normalize=getattr(cfg.data, "scale_normalize", False),
        test=True,               # deterministic sampling, no augmentation
    )
    loader = torch.utils.data.DataLoader(
        feeder, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers or 0, pin_memory=True)
    P, Y, N = [], [], []
    for x, y, n in loader:
        logits = model(x.to(device))["logits"].float()
        P.append(logits.softmax(dim=1).cpu().numpy())
        Y.append(y.numpy())
        N.extend(n)
    return np.concatenate(P), np.concatenate(Y), N


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--logs", required=True)
    p.add_argument("--variants", nargs="*", default=[""])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--limit-folds", type=int, default=None,
                   help="Score only the first N folds (trials); the split "
                        "itself still uses --folds, which LPO requires.")
    p.add_argument("--out-dir", default="results/predictions")
    p.add_argument("--tag", required=True,
                   help="Cache filename stem; one .npz per variant is written "
                        "as <tag>_<variant>.npz")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)
    configure_eval_runtime(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for variant in args.variants:
        names_all, labels_all, probs_all = [], [], []
        n_eval = min(args.limit_folds or args.folds, args.folds)
        for fold in range(1, n_eval + 1):
            fold_dir = (Path(args.logs) / variant / f"fold{fold:02d}" if variant
                        else Path(args.logs) / f"fold{fold:02d}")
            ckpts = sorted(fold_dir.glob("*/version_*/checkpoints/*.ckpt"))
            if not ckpts:
                print(f"  {variant}/fold{fold:02d}: no checkpoint", flush=True)
                continue
            # `last` is the rule the recipe reports on; sorting puts
            # last.ckpt after the epoch-numbered ones.
            ck = [c for c in ckpts if c.name == "last.ckpt"] or ckpts[-1:]

            split = build_lpo_split(cfg.data.data_path, fold, args.folds)
            model = get_model(cfg.model.type).from_config(cfg)
            lit = LightningModel.load_from_checkpoint(
                str(ck[0]), model=model, base_lr=cfg.training.base_lr,
                num_class=cfg.model.num_class, map_location=device)
            probs, labels, names = _probs(
                lit.model.eval().to(device), cfg,
                [i for _, i in split["test"]], device)
            probs_all.append(probs)
            labels_all.append(labels)
            names_all.extend(names)
            acc = (probs.argmax(1) == labels).mean() * 100
            print(f"  {variant or 'default':>10s} fold{fold:02d} "
                  f"n={len(names):4d} acc={acc:5.2f}", flush=True)

        if not names_all:
            print(f"  {variant}: nothing collected, skipped")
            continue
        probs_all = np.concatenate(probs_all)
        labels_all = np.concatenate(labels_all)
        acc = (probs_all.argmax(1) == labels_all).mean() * 100
        out = out_dir / f"{args.tag}_{variant or 'default'}.npz"
        np.savez_compressed(out, names=np.array(names_all), labels=labels_all,
                            probs=probs_all.astype(np.float32))
        print(f"  -> {out}  {len(names_all)} clips, pooled acc {acc:.2f}\n")

    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
