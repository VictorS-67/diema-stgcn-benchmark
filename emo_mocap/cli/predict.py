"""Generate predictions from a trained model.

Usage:
    # Single split, explicit checkpoint
    emo-predict --config configs/diema7_stgcn.yaml --checkpoint path/to/best.ckpt

    # LPO fold with auto-discovered best_val_acc checkpoint
    emo-predict --config configs/diema7_stgcn.yaml \
        --fold 3 --num-folds 10 --auto-checkpoint --output preds_fold03.csv

    # One column per class (proba_0 ... proba_{N-1}) for downstream ensembling
    emo-predict --config configs/diema7_stgcn.yaml --checkpoint best.ckpt \
        --per-class-columns --output preds.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl

from emo_mocap.tools.config import interpolate_log_dir, load_config_with_overrides
from emo_mocap.tools.runtime import configure_eval_runtime
from emo_mocap.tools.checkpoints import find_best_checkpoint
from emo_mocap.models.registry import get_model
from emo_mocap.data.loader import Loader
from emo_mocap.data.splits import build_lpo_split
from emo_mocap.training.lightning_model import LightningModel


def _resolve_checkpoint(args, cfg, use_lpo, parser):
    """Either return ``args.checkpoint`` or auto-discover from log_dir."""
    if not args.auto_checkpoint:
        return args.checkpoint

    resolved_log_dir = interpolate_log_dir(
        cfg.logging.log_dir, cfg.data.seed, args.fold
    )
    experiment_name = cfg.logging.experiment_name
    if experiment_name is None:
        if not use_lpo:
            parser.error(
                "--auto-checkpoint without logging.experiment_name in the config "
                "requires --fold/--num-folds (so the experiment name "
                "{model_type}_fold{NN} can be inferred)."
            )
        experiment_name = f"{cfg.model.type}_fold{args.fold:02d}"
    ckpt = find_best_checkpoint(resolved_log_dir, experiment_name, args.checkpoint_preset)
    print(f"Auto-discovered checkpoint ({args.checkpoint_preset}): {ckpt}")
    return str(ckpt)


def _save_interpretability(predictions, output_dir):
    """Save attention / prototype outputs to per-sample .npz files.

    Each .npz contains whatever tensors the model emitted under out["attention"]
    (ProtoGCN: topology, prototype_response, joint_saliency; STAGCN: node, edge),
    plus predicted label, probability vector, and true label. Samples that
    share a name (e.g., identical clip_name across folds) are overwritten;
    callers should namespace the output_dir per fold / per seed if that
    matters for their workflow.

    Args:
        predictions: list of predict_step output 5-tuples
        output_dir: Path-like; created if missing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for batch_result in predictions:
        predicted, proba, labels, sample_names, attention = batch_result
        if attention is None:
            # Model doesn't expose interpretability (e.g., STGCN); skip silently.
            continue
        for i, name in enumerate(sample_names):
            payload = {
                "predicted": int(predicted[i].item()),
                "proba": proba[i].detach().cpu().numpy(),
                "true_label": int(labels[i].item()),
            }
            for k, v in attention.items():
                # The per-layer "topology_all_layers" key is a list of tensors;
                # stack into a single (L, V, V) array for storage.
                if isinstance(v, list):
                    payload[k] = np.stack([t[i].detach().cpu().numpy() for t in v])
                else:
                    payload[k] = v[i].detach().cpu().numpy()
            np.savez(output_dir / f"{name}.npz", **payload)
            written += 1
    print(f"Saved interpretability artifacts for {written} samples to {output_dir}")


def _write_predictions(predictions, num_class, out_file, per_class_columns):
    """Write predictions to CSV in either the legacy or per-class-columns format."""
    writer = csv.writer(out_file)
    if per_class_columns:
        proba_cols = [f"proba_{i}" for i in range(num_class)]
        writer.writerow(["sample_name", "true_label", "predicted_label", *proba_cols])
    else:
        writer.writerow(["sample_name", "true_label", "predicted_label", "probabilities"])

    for batch_result in predictions:
        # predict_step returns (predicted, proba, labels, sample_names, attention).
        # CSV output ignores the attention payload; --save-interpretability handles it.
        predicted, proba, labels, sample_names = batch_result[:4]
        for i in range(len(predicted)):
            row = [sample_names[i], labels[i].item(), predicted[i].item()]
            if per_class_columns:
                row.extend(f"{p:.6f}" for p in proba[i].tolist())
            else:
                row.append(" ".join(f"{p:.4f}" for p in proba[i].tolist()))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", help="Path to model checkpoint")
    ckpt_group.add_argument(
        "--auto-checkpoint", action="store_true",
        help="Auto-discover the best checkpoint from logging.log_dir "
             "(requires --fold or logging.experiment_name in the config)",
    )
    parser.add_argument(
        "--checkpoint-preset", default="best_val_acc",
        help="Which checkpoint preset to load when --auto-checkpoint "
             "(best_val_acc, best_val_loss, best_train_loss, last)",
    )

    parser.add_argument("--fold", type=int, default=None,
                        help="Fold number for LPO cross-validation (1-indexed)")
    parser.add_argument("--num-folds", type=int, default=None,
                        help="Total number of LPO folds")

    parser.add_argument(
        "--per-class-columns", action="store_true",
        help="Write one proba_<idx> column per class instead of a single "
             "space-separated 'probabilities' column. Required for emo-ensemble.",
    )
    parser.add_argument("--output", default=None, help="Output CSV path (default: stdout)")
    parser.add_argument(
        "--save-interpretability", default=None, metavar="DIR",
        help="Save per-sample .npz files containing model attention / prototype "
             "outputs to DIR. Additive to --output; CSV format is unchanged. "
             "Models that don't expose attention (e.g., STGCN) produce no files.",
    )
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)

    # Determine split: on-the-fly LPO or from config
    use_lpo = args.fold is not None or args.num_folds is not None
    if use_lpo:
        if args.fold is None or args.num_folds is None:
            parser.error("--fold and --num-folds must be used together")
        if args.fold < 1 or args.fold > args.num_folds:
            parser.error(f"--fold must be between 1 and {args.num_folds}")
        split_dict = build_lpo_split(cfg.data.data_path, args.fold, args.num_folds)
        split_path = None
    else:
        split_dict = None
        split_path = cfg.data.split_path

    checkpoint_path = _resolve_checkpoint(args, cfg, use_lpo, parser)

    # Build model architecture from config and load trained weights
    model_cls = get_model(cfg.model.type)
    model = model_cls.from_config(cfg)
    lit_model = LightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        base_lr=cfg.training.base_lr,
        num_class=cfg.model.num_class,
    )

    target_repr = getattr(cfg.data, "target_repr", "euler")
    loader = Loader(
        data_path=cfg.data.data_path,
        split_path=split_path,
        split_dict=split_dict,
        clip_length=cfg.training.clip_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        target_repr=target_repr,
        seed=cfg.data.seed,
        streams=getattr(cfg.data, "streams", None),
        scale_normalize=getattr(cfg.data, "scale_normalize", False),
    )

    # Same numeric policy as emo-evaluate: predictions and the metrics they
    # are scored against must come out of the same arithmetic.
    trainer = pl.Trainer(
        deterministic=True,
        accelerator=cfg.training.accelerator,
        devices=getattr(cfg.training, "devices", "auto"),
        precision=configure_eval_runtime(cfg),
    )
    predictions = trainer.predict(lit_model, datamodule=loader)

    out_file = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        _write_predictions(
            predictions, cfg.model.num_class, out_file, args.per_class_columns
        )
    finally:
        if args.output:
            out_file.close()
            print(f"Predictions written to {args.output}")

    if args.save_interpretability is not None:
        _save_interpretability(predictions, args.save_interpretability)


if __name__ == "__main__":
    main()
