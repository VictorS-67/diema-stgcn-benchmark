"""Score every checkpoint of every fold of a run, and write one JSON summary.

This is the step that turns ten trained folds into a number you can quote. For
each checkpoint it finds under ``--logs`` it records, on the training,
validation and test split of that fold:

* accuracy and macro-F1
* cross-entropy and expected calibration error
* the same two after temperature scaling, with the temperature fitted on
  validation

The training column is scored the way the other two are, in eval mode with
augmentation off, so the generalisation gap is a measurement rather than an
inference from the running training loss. Read it: a run that has not reached
roughly 99.9% training accuracy has not finished fitting, and its test number
is not comparable with one that has.

Output layout is ``{variant: {fold: [one row per checkpoint]}}``. With the
recommended config only the final epoch is saved, so each fold has one row.

Example:

    python scripts/score_folds.py \
        --config configs/diema7_stgcn_recipe.yaml \
        --logs runs/diema7/seed255 --variants recipe --folds 10 \
        --out runs/diema7/seed255.json --override data.seed=255
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from emo_mocap.data.feeder import Feeder
from emo_mocap.data.loader import Loader
from emo_mocap.data.splits import build_lpo_split, subsample_train_performers
from emo_mocap.models.registry import get_model
from emo_mocap.tools.calibration import (
    expected_calibration_error as _ece,
    fit_temperature as _fit_temperature,
)
from emo_mocap.tools.config import load_config_with_overrides
from emo_mocap.tools.runtime import configure_eval_runtime
from emo_mocap.training.lightning_model import LightningModel

_EPOCH_RE = re.compile(r"epoch[=-](\d+)")


def _epoch_of(path: Path) -> int:
    """Training epoch of a checkpoint.

    Here — unlike in the preset comparison — the epoch *is* the x-axis, so
    ``last.ckpt``, which carries no epoch in its name, is read out of the
    file rather than guessed from ``cfg.training.max_epochs``: an override
    that shortened the run would otherwise place it at the wrong x.
    Lightning stores the completed-epoch count under ``"epoch"``.
    """
    m = _EPOCH_RE.search(path.name)
    if m:
        return int(m.group(1))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"]) - 1  # Lightning's count is 1-past the last


@torch.no_grad()
def _collect_logits(model, dataloader, device):
    """Run one split, returning stacked logits and labels on the CPU."""
    logits, labels = [], []
    model.eval().to(device)
    for batch in dataloader:
        x, y, _names = batch
        out = model(x.to(device))
        logits.append(out["logits"].float().cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(labels)


def _train_eval_loader(cfg, loader):
    """The training split, scored the way the other two splits are scored.

    ``loader.train_dataloader()`` is the *training* view: augmented, shuffled,
    last batch dropped. Fitting quality has to be measured on the same footing
    as validation and test — deterministic sampling, no augmentation, model in
    eval mode — otherwise the train/val gap conflates memorisation with how
    hard the augmentation happened to make each epoch.

    This is what turns the logged ``train_loss`` (train mode, augmented,
    dropout active) from a lower bound into a measurement, and it is the
    number that separates "the model has nothing left to fit" from "the model
    is still underfitting".
    """
    dataset = Feeder(
        data_path=cfg.data.data_path,
        # loader.train_indices already reflects data.train_performer_fraction,
        # because the split handed to the Loader was subsampled. Scoring the
        # full training split here would report fitting quality on performers
        # the model never saw, which is a test metric wearing a train label.
        indices=loader.train_indices,
        clip_length=cfg.training.clip_length,
        target_repr=cfg.data.target_repr,
        seed=cfg.data.seed,
        # Must mirror the Loader below: without it a position-stream config is
        # scored against rotation data, which fails loudly on the channel
        # count rather than quietly producing a wrong number.
        streams=getattr(cfg.data, "streams", None),
        scale_normalize=getattr(cfg.data, "scale_normalize", False),
        test=True,               # deterministic sampling, no augmentation
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers or 0, pin_memory=True,
    )


def _metrics(logits, labels, num_class, temperature=1.0):
    z = logits / temperature
    probs = z.softmax(dim=1)
    pred = probs.argmax(dim=1)
    acc = pred.eq(labels).float().mean().item()
    # macro-F1, computed directly so the script has no torchmetrics state to
    # reset between the many checkpoints it scores.
    f1s = []
    for c in range(num_class):
        tp = ((pred == c) & (labels == c)).sum().item()
        fp = ((pred == c) & (labels != c)).sum().item()
        fn = ((pred != c) & (labels == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return {
        "acc": acc,
        "f1": float(np.mean(f1s)),
        "nll": F.cross_entropy(z, labels).item(),
        "ece": _ece(probs, labels),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--logs", required=True, help="Root of the run's log tree")
    p.add_argument("--variants", nargs="*", default=[""],
                   help="Subdirectories under --logs to walk; '' for a flat tree")
    p.add_argument("--variant-override", default=None,
                   help="Config key set to each variant name, e.g. "
                        "model.edge_weighting. Omit when variants differ by "
                        "something the config already encodes.")
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=10,
                   help="Size of the LPO scheme (defines the splits).")
    p.add_argument("--limit-folds", type=int, default=None)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--score-train", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also score the training split (eval mode, no "
                        "augmentation) to measure the generalisation gap. "
                        "~7x the validation split, inference only.")
    args = p.parse_args()

    n_eval = min(args.limit_folds or args.folds, args.folds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logs_root = Path(args.logs)
    results: dict = {}
    splits: dict = {}
    missing: list[str] = []
    started = time.time()

    for variant in args.variants:
        extra = ([f"{args.variant_override}={variant}"]
                 if args.variant_override and variant else [])
        cfg = load_config_with_overrides(args.config, [*extra, *args.override])
        configure_eval_runtime(cfg)
        num_class = cfg.model.num_class
        results[variant or "default"] = {}

        for fold in range(1, n_eval + 1):
            # Keyed by (fold, fraction): a performer-fraction sweep asks the
            # same fold for differently-sized training cohorts, and a cache
            # keyed on fold alone would hand every arm the first one built.
            fraction = getattr(cfg.data, "train_performer_fraction", 1.0)
            if (fold, fraction) not in splits:
                split = build_lpo_split(cfg.data.data_path, fold, args.folds)
                if fraction != 1.0:
                    split = subsample_train_performers(
                        split, fraction, seed=cfg.data.seed)
                splits[fold, fraction] = split

            fold_dir = logs_root / variant / f"fold{fold:02d}" if variant \
                else logs_root / f"fold{fold:02d}"
            ckpt_dirs = sorted(fold_dir.glob("*/version_*/checkpoints"))
            if not ckpt_dirs:
                missing.append(f"{variant}/fold{fold:02d}")
                continue
            ckpt_dir = ckpt_dirs[-1]

            loader = Loader(
                data_path=cfg.data.data_path, split_dict=splits[fold, fraction],
                clip_length=cfg.training.clip_length,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.data.num_workers,
                target_repr=cfg.data.target_repr, seed=cfg.data.seed,
                # pybvh-ml >= 0.6: a config that packs position streams must
                # be scored on those streams. Omitting this silently falls
                # back to rotations.
                streams=getattr(cfg.data, "streams", None),
                scale_normalize=getattr(cfg.data, "scale_normalize", False),
            )
            loader.setup("fit")     # val split
            loader.setup("test")    # test split
            val_dl, test_dl = loader.val_dataloader(), loader.test_dataloader()
            train_dl = _train_eval_loader(cfg, loader) if args.score_train else None

            per_epoch = []
            # `last.ckpt` usually duplicates the final every-N checkpoint;
            # score each epoch once so the curve has no doubled points.
            by_epoch = {}
            for ck in sorted(ckpt_dir.glob("*.ckpt")):
                by_epoch.setdefault(_epoch_of(ck), ck)

            for epoch, ck in sorted(by_epoch.items()):
                model = get_model(cfg.model.type).from_config(cfg)
                lit = LightningModel.load_from_checkpoint(
                    str(ck), model=model, base_lr=cfg.training.base_lr,
                    num_class=num_class, map_location=device)

                vlog, vlab = _collect_logits(lit.model, val_dl, device)
                tlog, tlab = _collect_logits(lit.model, test_dl, device)
                # Temperature is fitted on validation only — using test would
                # be exactly the leakage this script is meant to measure.
                temp = _fit_temperature(vlog, vlab)

                record = {
                    "epoch": epoch,
                    "ckpt": ck.name,
                    "temperature": temp,
                    "val": _metrics(vlog, vlab, num_class),
                    "val_scaled": _metrics(vlog, vlab, num_class, temp),
                    "test": _metrics(tlog, tlab, num_class),
                    "test_scaled": _metrics(tlog, tlab, num_class, temp),
                }
                if train_dl is not None:
                    trlog, trlab = _collect_logits(lit.model, train_dl, device)
                    # Raw only: the temperature is fitted to correct confidence
                    # on unseen data, and the model has seen all of this. A
                    # "tempered train loss" would not mean anything.
                    record["train"] = _metrics(trlog, trlab, num_class)
                per_epoch.append(record)

                gap = (f" train_acc={record['train']['acc']*100:5.2f}"
                       if "train" in record else "")
                print(f"{variant or 'default':12s} fold{fold:02d} ep{epoch:3d} "
                      f"T={temp:5.3f}{gap} "
                      f"val_acc={record['val']['acc']*100:5.2f} "
                      f"test_acc={record['test']['acc']*100:5.2f} "
                      f"val_nll={record['val']['nll']:.3f}->"
                      f"{record['val_scaled']['nll']:.3f}", flush=True)

            per_epoch.sort(key=lambda r: r["epoch"])
            results[variant or "default"][str(fold)] = per_epoch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}  ({time.time() - started:.0f}s)")
    if missing:
        print(f"missing runs: {missing}")


if __name__ == "__main__":
    main()
