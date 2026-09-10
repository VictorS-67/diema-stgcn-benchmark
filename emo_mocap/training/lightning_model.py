"""Model-agnostic Lightning training wrapper.

Works with any model that follows the BaseModel protocol (returns a dict
with at least {"logits": tensor}). Handles auxiliary losses generically
via the "aux_losses" key.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn

import torchmetrics
import pytorch_lightning as pl

from emo_mocap.tools.calibration import (
    expected_calibration_error,
    fit_temperature,
)


class LightningModel(pl.LightningModule):
    """Lightning wrapper for training any BaseModel subclass.

    Args:
        model: a BaseModel instance
        base_lr: base learning rate
        num_class: number of classes for metrics
        optimizer: 'SGD' or 'Adam' (default: 'SGD')
        scheduler_type: 'cosine' or 'step' (default: 'cosine')
        scheduler_params: for 'step', provide [step_size, gamma] (default: [])
        weight_decay: L2 regularization (default: 0.0001)
        aux_loss_weights: per-loss weight overrides, e.g. {"ab_logits": 0.5} (default: all 1.0)
    """

    def __init__(
        self,
        model,
        base_lr,
        num_class,
        optimizer="SGD",
        scheduler_type="cosine",
        scheduler_params=None,
        weight_decay=0.0001,
        aux_loss_weights=None,
        label_smoothing=0.0,
        provenance=None,
    ):
        """provenance: free-form dict describing *which data* this run consumed.

        Nothing here affects training. It exists because a checkpoint that does
        not record its own dataset cannot be compared to another one months
        later without archaeology — and this project has two corpus builds on
        disk whose accuracies differ by about a point, which is the same size
        as several of its findings. Anything placed here is written into
        `hparams.yaml` and into every checkpoint, so a run carries its own
        provenance wherever it is copied.
        """
        super().__init__()
        if scheduler_params is None:
            scheduler_params = []
        self.provenance = provenance or {}

        self.base_lr = base_lr
        self.model = model
        # Applied to the auxiliary losses too (they call the same loss_fn),
        # which is the right default: an aux head predicting the same 7 classes
        # should be regularised the same way as the main one.
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing
        self.optimizer_name = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params
        self.weight_decay = weight_decay
        self.aux_loss_weights = aux_loss_weights or {}

        if scheduler_type not in ("cosine", "step"):
            raise ValueError("Unsupported scheduler type")
        if optimizer not in ("SGD", "Adam"):
            raise ValueError("Unsupported optimizer type")
        if scheduler_type == "step" and not scheduler_params:
            raise ValueError("Scheduler params must be provided for step scheduler")

        self.save_hyperparameters(ignore=["model"])

        # Validation logits, kept for one epoch so the temperature can be
        # fitted at epoch end (see _on_validation_epoch_end). Stored as numpy:
        # Lightning validates under inference_mode, and inference tensors are
        # barred from autograd graphs, which the LBFGS fit needs. Round-
        # tripping through numpy drops that tag. It is also small — a few
        # hundred clips by num_class floats.
        self._val_logits: list[np.ndarray] = []
        self._val_labels: list[np.ndarray] = []

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_class, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)  # dict with at least {"logits": ...}
        loss = self.loss_fn(out["logits"], labels)

        # Some models (ProtoGCN) need the training labels to compute an
        # auxiliary loss but can't receive them through forward(). They return
        # the raw projection features under "aux_features" and own a submodule
        # that the wrapper invokes here, promoting the result into aux_losses
        # so the existing weighting loop handles it.
        if "aux_features" in out:
            for feat_name, feat_value in out["aux_features"].items():
                loss_name = feat_name.removesuffix("_projection")
                loss_module = getattr(self.model, loss_name, None)
                if loss_module is not None and callable(loss_module):
                    aux_value = loss_module(feat_value, labels)
                    out.setdefault("aux_losses", {})[loss_name] = aux_value

        # Handle auxiliary losses (e.g., STAGCN attention branch)
        if "aux_losses" in out:
            for name, value in out["aux_losses"].items():
                if name.endswith("_logits"):
                    aux_loss = self.loss_fn(value, labels)
                else:
                    aux_loss = value
                weight = self.aux_loss_weights.get(name, 1.0)
                loss = loss + weight * aux_loss
                self.log(f"train_{name}", aux_loss)

        self.log("train_loss", loss, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)
        loss = self.loss_fn(out["logits"], labels)
        predicted = torch.argmax(out["logits"], dim=1)

        self.log("val_loss", loss, prog_bar=True, batch_size=len(labels))
        self.val_acc(predicted, labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

        self._val_logits.append(out["logits"].detach().float().cpu().numpy())
        self._val_labels.append(labels.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        """Log a calibration-corrected validation loss alongside the raw one.

        Raw ``val_loss`` conflates two things: how well the model separates
        the classes, and how honest its confidence is. Confidence keeps
        drifting upward for as long as training runs, so raw ``val_loss``
        starts rising long before the model stops improving — on this dataset
        it bottoms out around epoch 30 while test accuracy is still climbing
        at epoch 100. Anyone reading it as an overfitting alarm stops far too
        early.

        Fitting a temperature per epoch removes the confidence term and
        leaves (approximately) the separation term. ``val_loss_tempered`` is
        therefore the honest answer to *"is this model still getting
        better?"*, which makes it a budget-sizing diagnostic: while it is
        still falling at the end of a run, the schedule is too short.

        It is a *diagnostic*, not a selection metric. Stopping or selecting on
        it measurably underperforms simply taking the last checkpoint, because
        the fitted minimum is broad and late and a few hundred validation
        clips cannot resolve it. ``val_temperature`` is logged next to it: it
        rises from ~1.3 to ~3.2 over a run, and that drift is the thing being
        corrected for.
        """
        logits, labels = self._val_logits, self._val_labels
        self._val_logits, self._val_labels = [], []

        if self.trainer.sanity_checking or not logits:
            return
        if self.trainer.world_size > 1:
            # Each rank holds only its shard, so a per-rank fit would report
            # as many different temperatures as there are devices. Gathering
            # is possible but this codebase trains on one GPU; refuse rather
            # than log a number that quietly means something else.
            warnings.warn(
                "val_loss_tempered is not computed under distributed "
                "training (world_size > 1); the temperature would be fitted "
                "per rank on a shard of the validation split.",
                RuntimeWarning, stacklevel=2,
            )
            return

        logits = torch.from_numpy(np.concatenate(logits))
        labels = torch.from_numpy(np.concatenate(labels)).long()

        temperature = fit_temperature(logits, labels)
        tempered = nn.functional.cross_entropy(logits / temperature, labels)

        self.log("val_temperature", temperature)
        self.log("val_loss_tempered", tempered)
        self.log("val_ece", expected_calibration_error(
            logits.softmax(dim=1), labels))
        self.log("val_ece_tempered", expected_calibration_error(
            (logits / temperature).softmax(dim=1), labels))

    def test_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)
        predicted = torch.argmax(out["logits"], dim=1)

        self.test_acc(predicted, labels)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.test_f1(predicted, labels)
        self.log("test_f1", self.test_f1, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        inputs, labels, sample_names = batch
        out = self(inputs)
        proba = torch.softmax(out["logits"], dim=1)
        predicted = torch.argmax(proba, dim=1)
        attention = out.get("attention", None)
        return predicted, proba, labels, sample_names, attention

    def _param_groups(self):
        """Split parameters into weight-decayed and not.

        The model owns the decision: ``BaseModel.no_weight_decay()`` returns
        the names of its own parameters that must be exempt (learned graph
        terms, typically). This wrapper stays model-agnostic — it never
        inspects parameter names for meaning, only for membership in the set
        the model handed it.
        """
        exempt = {f"model.{n}" for n in self.model.no_weight_decay()}
        known = {n for n, _ in self.named_parameters()}
        unknown = exempt - known
        if unknown:
            raise ValueError(
                f"{type(self.model).__name__}.no_weight_decay() named "
                f"parameters that do not exist: {sorted(unknown)}"
            )

        decayed, exempted = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (exempted if name in exempt else decayed).append(p)

        groups = [{"params": decayed, "weight_decay": self.weight_decay}]
        if exempted:
            groups.append({"params": exempted, "weight_decay": 0.0})
        return groups

    def configure_optimizers(self):
        groups = self._param_groups()
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                groups,
                lr=self.base_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                groups,
                lr=self.base_lr,
                weight_decay=self.weight_decay,
            )

        if self.scheduler_type == "step":
            step_size, gamma = self.scheduler_params
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
