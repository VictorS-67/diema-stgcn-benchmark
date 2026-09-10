"""Loader LightningDataModule for managing train/val/test data splits."""

import os
import pickle
from pathlib import Path

import torch.utils.data
import pytorch_lightning as pl

from emo_mocap.data.feeder import Feeder


def _default_num_workers() -> int:
    """Sensible default: half the available CPU cores, capped at 8.

    More than 8 workers rarely helps and uses extra memory. Capped rather
    than using all cores to leave headroom for the main process and OS.
    """
    return min((os.cpu_count() or 4) // 2, 8)


class EpochSeedCallback(pl.Callback):
    """Advance the train Feeder's augmentation epoch at each epoch start.

    The Feeder derives every sample's augmentation from ``(seed, epoch, idx)``,
    which makes a draw reproducible and worker-count-independent — but only
    varies across epochs if something tells it the epoch changed. Lightning's
    DataModule has no per-epoch hook, so this Callback carries the message.
    """

    @staticmethod
    def _set(trainer, epoch):
        dataset = getattr(trainer.datamodule, "dataset_train", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def setup(self, trainer, pl_module, stage):
        """Claim epoch 0 before the DataLoader is built, let alone forked.

        ``on_train_epoch_start`` alone is too late. Lightning prefetches the
        first training batches before *any* hook that can reach the dataset
        fires — including ``on_train_start`` — so those batches come from a
        Feeder that has never been told its epoch. The draws themselves are
        still right (the unset state reads as epoch 0, which is what the hook
        would have set a moment later), but every worker emits the Feeder's
        "you forgot to register me" warning on every single run, and a
        warning that cries wolf every time is how a genuine occurrence of
        that bug goes unnoticed.

        ``Callback.setup`` runs after the DataModule's ``setup`` — so
        ``dataset_train`` exists — and before dataloaders are constructed.
        Setting the epoch here is also what keeps the warning meaningful: it
        marks the dataset as managed by *this callback*, so the warning still
        fires for a hand-built Feeder that nothing is advancing.
        """
        if stage == "fit":
            self._set(trainer, 0)

    def on_train_epoch_start(self, trainer, pl_module):
        self._set(trainer, trainer.current_epoch)


class Loader(pl.LightningDataModule):
    """LightningDataModule that wraps Feeder datasets for each split.

    Loads a preprocessed npz file and a split pickle, creates Feeder
    instances for train/val/test, and provides DataLoaders for training.

    Args:
        data_path: path to the .npz data file (pybvh-ml format)
        split_path: path to the split dictionary pickle (mutually exclusive with split_dict)
        split_dict: in-memory split dictionary (mutually exclusive with split_path)
        clip_length: number of frames to sample (default: 64)
        batch_size: batch size for train/val (default: 64)
        num_workers: DataLoader workers (None = half the CPU cores, capped at 8)
        target_repr: target representation for model input (default: 'euler')
        debug: if True, limit to 100 samples (default: False)
        seed: base seed for the train Feeder's augmentation (default: 255)
        augmentation_pipeline: pybvh_ml.AugmentationPipeline or None
        euler_orders: per-joint Euler orders (for quat→Euler conversion;
            defaults to the dataset's stored orders)
        streams: stream tuple for the packed tensor (pybvh-ml >= 0.6);
            None keeps the historical root+rotations packing. See Feeder.

    Training runs must also register :class:`EpochSeedCallback` on the Trainer
    so the train Feeder's augmentation varies across epochs — ``emo-train``
    does this.
    """

    def __init__(
        self,
        data_path,
        split_path=None,
        split_dict=None,
        clip_length=64,
        batch_size=64,
        num_workers=None,
        target_repr="euler",
        debug=False,
        seed=255,
        augmentation_pipeline=None,
        euler_orders=None,
        streams=None,
        scale_normalize=False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else _default_num_workers()
        self.debug = debug
        self.clip_length = clip_length
        self.target_repr = target_repr
        self.seed = seed
        self.augmentation_pipeline = augmentation_pipeline
        self.euler_orders = euler_orders
        self.streams = streams
        self.scale_normalize = scale_normalize

        if split_dict is not None:
            self.split_dict = split_dict
        elif split_path is not None:
            with open(split_path, "rb") as f:
                self.split_dict = pickle.load(f)
        else:
            raise ValueError("Either split_path or split_dict must be provided")

        self.train_indices = [idx for _, idx in self.split_dict["train"]]
        self.val_indices = [idx for _, idx in self.split_dict["val"]]

        if "test" in self.split_dict and len(self.split_dict["test"]) > 0:
            self.test_indices = [idx for _, idx in self.split_dict["test"]]
        else:
            self.test_indices = self.val_indices

        if self.debug:
            debug_size = 100
            self.train_indices = self.train_indices[:debug_size]
            self.val_indices = self.val_indices[:debug_size]
            self.test_indices = self.test_indices[:debug_size]

    def setup(self, stage: str):
        common = dict(
            data_path=self.data_path,
            clip_length=self.clip_length,
            target_repr=self.target_repr,
            seed=self.seed,
            euler_orders=self.euler_orders,
            streams=self.streams,
            scale_normalize=self.scale_normalize,
        )

        if stage == "fit":
            self.dataset_train = Feeder(
                indices=self.train_indices,
                test=False,
                augmentation_pipeline=self.augmentation_pipeline,
                **common,
            )
            self.dataset_val = Feeder(
                indices=self.val_indices,
                test=True,
                **common,
            )

        if stage in ("test", "predict"):
            self.dataset_test = Feeder(
                indices=self.test_indices,
                test=True,
                **common,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
