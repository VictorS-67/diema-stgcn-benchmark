"""Base model protocol for all emotion recognition models.

Every model in this project must subclass BaseModel. This ensures a
consistent interface so that the training loop, CLI, and config system
work with any architecture (GCN, LSTM, Transformer, ...).
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base class for all emotion recognition models.

    Contract:
    - Input: always (N, C, T, V) tensor -- batch, channels, frames, joints.
      The model reshapes internally if it needs a different layout
      (e.g., an LSTM might reshape to (N, T, V*C)).
    - Output: always a dict with at least {"logits": Tensor of shape (N, num_class)}.
      Models may add extra keys for auxiliary outputs:
        - "aux_losses": dict of {name: value} -- added to main loss during training
        - "attention": any tensor -- for visualization/logging only
        - Any other model-specific key -- ignored by LightningModel unless handled
    """

    def _validate_input(self, x):
        """Guard against silent dimension-order bugs."""
        assert x.ndim == 4, (
            f"Expected input of shape (N, C, T, V), got {x.shape}. "
            f"All models receive (batch, channels, frames, joints)."
        )

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, C, T, V)
        Returns:
            dict with at least {"logits": Tensor (N, num_class)}
        """
        ...

    @property
    @abstractmethod
    def output_dim(self):
        """Return the number of output classes."""
        ...

    def no_weight_decay(self):
        """Names of this model's parameters that must not be weight-decayed.

        Weight decay pulls a parameter toward zero, which is the right prior
        for a linear layer's weights and the wrong one for a parameter whose
        zero point is meaningless or actively harmful. The clearest case is a
        learned graph: an edge-importance mask initialised to 1 decays toward
        "delete every edge", and a learned adjacency decays toward "no message
        passing at all" — a pressure the conv weights then have to fight.

        Return names relative to this model (as produced by
        ``self.named_parameters()``), e.g. ``"st_gcn_networks.0.gcn1.PA"``.
        ``LightningModel`` puts them in a ``weight_decay=0`` param group.
        The default is empty, so a model that doesn't care ignores this.
        """
        return set()

    @classmethod
    def from_config(cls, config):
        """Instantiate a model from a config namespace.

        Subclasses should override if they need custom config mapping.
        Default extracts: num_class, edge_index, num_nodes, in_channels, dropout.
        """
        return cls(
            num_class=config.model.num_class,
            edge_index=config.skeleton.inward_edges,
            num_nodes=config.skeleton.num_nodes,
            in_channels=config.model.in_channels,
            dropout=getattr(config.model, "dropout", 0.5),
        )
