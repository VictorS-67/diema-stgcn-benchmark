"""On-disk checkpoint discovery for the conventional emo-train log layout.

Used by the predict / evaluate / ensemble CLIs so per-fold checkpoint paths
don't have to be passed by hand. The expected layout is:

    {log_dir}/{experiment_name}/version_{N}/checkpoints/{preset}-*.ckpt

where ``{experiment_name}`` is typically ``{model_type}_fold{NN}`` for LPO
runs (see ``emo_mocap/cli/train.py``).
"""

import re
from pathlib import Path


def find_best_checkpoint(
    log_dir: str | Path,
    experiment_name: str,
    preset: str = "best_val_acc",
) -> Path:
    """Find a saved checkpoint matching ``preset`` for one experiment.

    Searches the latest ``version_N`` directory first, then falls back to
    older versions if it has no matching ckpt. Returns the path to the
    first match.

    ``preset`` is a checkpointing preset name from the train CLI
    (``best_val_acc``, ``best_val_loss``, ``best_train_loss``, ``last``).

    Raises FileNotFoundError if no matching ckpt is found anywhere.
    """
    base = Path(log_dir) / experiment_name
    if not base.exists():
        raise FileNotFoundError(f"Experiment directory not found: {base}")

    versions = sorted(
        (d for d in base.iterdir()
         if d.is_dir() and re.match(r"^version_\d+$", d.name)),
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )
    if not versions:
        raise FileNotFoundError(f"No version_* dirs under {base}")

    if preset == "last":
        glob_pattern = "last.ckpt"
    else:
        # best_val_acc -> best-val-acc-*.ckpt (matches train.py's filename pattern)
        glob_pattern = f"{preset.replace('_', '-')}-*.ckpt"

    for v in versions:
        ckpt_dir = v / "checkpoints"
        if not ckpt_dir.exists():
            continue
        matches = sorted(ckpt_dir.glob(glob_pattern))
        if matches:
            # save_top_k=1 keeps a single matching file per version, so the
            # sort is just for determinism if a stray file ever lingers.
            return matches[-1]

    raise FileNotFoundError(
        f"No checkpoint matching {glob_pattern!r} found under any version_* dir of {base}"
    )
