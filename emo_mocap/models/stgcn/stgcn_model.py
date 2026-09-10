"""Full STGCN classification model.

Stacks 10 STGCN_Units (spatial-temporal graph convolution blocks) to
classify skeleton sequences into emotion categories. Supports both
the basic TCN variant and the multi-branch STGCN++ variant.
"""

import math

import torch
import torch.nn as nn

from emo_mocap.models.base import BaseModel
from emo_mocap.models.stgcn.ST_units import STGCN_Unit
from emo_mocap.models.stgcn.adj_matrix import GraphAAGCN
from emo_mocap.models.stgcn.spatial_units import resolve_edge_weighting


class STGCN_Model(BaseModel):
    """Full ST-GCN classification model.

    Pipeline: Input -> BatchNorm1d -> 10 STGCN_Units -> Global Avg Pool
              -> Dropout -> Linear FC -> {"logits": (N, num_class)}

    Args:
        num_class: number of output classes
        edge_index: list of (child, parent) edge tuples
        num_nodes: number of joints in the skeleton
        in_channels: number of input channels (3 for xyz)
        dropout: dropout rate for the classifier head (default: 0.5)
        edge_weighting: how much of the graph is learned, per block. One of
            'none' / 'init' / 'offset' / 'importance' (PySKL/mmaction2
            `unit_gcn` semantics; see spatial_units.ADAPTIVE_MODES).
            Legacy bools accepted: True -> 'init', False -> 'none'.
        plusplus: if True, use the STGCN++ multi-branch temporal unit
            (six parallel branches with dilations 1-4, PySKL) in place of
            the original ST-GCN's single 9-frame temporal convolution
            (default: True, here and in ``from_config``). The DIEMA
            campaign of 2026-08/09 ran with it False, i.e. the original
            block; every shipped DIEMA config sets it explicitly.
        unit_dropout: dropout within TCN_Unit_plus branches (default: 0.1);
            inert when ``plusplus`` is False.
        base_channels: width of the first four blocks; the ladder is
            base / 2*base / 4*base and the classifier reads 4*base
            (default: 64, the original ST-GCN's 64/128/256). Scales the
            whole network so capacity can be swept from one config key.
    """

    def __init__(
        self,
        num_class,
        edge_index,
        num_nodes,
        in_channels,
        dropout=0.5,
        edge_weighting="importance",
        plusplus=True,
        unit_dropout=0.1,
        base_channels=64,
    ):
        super().__init__()
        graph = GraphAAGCN(edge_index, num_nodes)
        A = torch.as_tensor(graph.A).clone().float()
        # How much of the graph is learned. Each block gets its own copy of the
        # state (PySKL/mmaction2 pass `A.clone()` per block; the original
        # ST-GCN keeps a per-layer `edge_importance` ParameterList), so blocks
        # at different depths can specialise their topology.
        self.edge_weighting = resolve_edge_weighting(edge_weighting)
        blk = dict(adaptive=self.edge_weighting, unit_dropout=unit_dropout)
        self.num_class = num_class
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        # Channel ladder: four blocks at the base width, three at twice it and
        # three at four times it, the temporal stride halving the frame count
        # at each widening -- the original ST-GCN's 64/128/256 when
        # base_channels is 64. One knob scales the whole ladder.
        c1, c2, c3 = base_channels, 2 * base_channels, 4 * base_channels
        self.st_gcn_networks = nn.ModuleList([
            STGCN_Unit(in_channels, c1, A, plusplus, residual=False, **blk),
            STGCN_Unit(c1, c1, A, plusplus, **blk),
            STGCN_Unit(c1, c1, A, plusplus, **blk),
            STGCN_Unit(c1, c1, A, plusplus, **blk),
            STGCN_Unit(c1, c2, A, plusplus, stride=2, **blk),
            STGCN_Unit(c2, c2, A, plusplus, **blk),
            STGCN_Unit(c2, c2, A, plusplus, **blk),
            STGCN_Unit(c2, c3, A, plusplus, stride=2, **blk),
            STGCN_Unit(c3, c3, A, plusplus, **blk),
            STGCN_Unit(c3, c3, A, plusplus, **blk),
        ])
        self.fc = nn.Linear(c3, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        self.drop_out = nn.Dropout(dropout) if dropout > 0 else lambda x: x

    def forward(self, x):
        self._validate_input(x)
        N, C, T, V = x.size()
        # (N, C, T, V) -> (N, V*C, T) for BatchNorm
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        # (N, V*C, T) -> (N, C, T, V) back to spatial layout
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for unit in self.st_gcn_networks:
            x = unit(x)
        # Global average pooling over time and joints
        c_new = x.size(1)
        x = x.view(N, c_new, -1)
        x = x.mean(2)
        x = self.drop_out(x)
        x = self.fc(x)
        return {"logits": x}

    @property
    def output_dim(self):
        return self.num_class

    def no_weight_decay(self):
        """The per-block graph terms — see ``BaseModel.no_weight_decay``.

        Under ``importance`` the mask starts at 1 and decay would drag it to 0,
        i.e. toward deleting every bone; under ``init`` decay shrinks the
        adjacency itself. Both are the wrong prior, and both would confound a
        mode comparison, since the modes are not equally affected.
        """
        return {
            name for name, _ in self.named_parameters()
            if name.endswith(("gcn1.A", "gcn1.PA"))
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            num_class=config.model.num_class,
            edge_index=config.skeleton.inward_edges,
            num_nodes=config.skeleton.num_nodes,
            in_channels=config.model.in_channels,
            edge_weighting=getattr(config.model, "edge_weighting", "importance"),
            # Default True matches the constructor. Until 2026-09-06 this
            # defaulted to False while the constructor said True, and the
            # DIEMA configs' explicit `plusplus: false` was the only thing
            # documenting which temporal block actually trained.
            plusplus=getattr(config.model, "plusplus", True),
            dropout=getattr(config.model, "dropout", 0.5),
            unit_dropout=getattr(config.model, "unit_dropout", 0.1),
            base_channels=getattr(config.model, "base_channels", 64),
        )


from emo_mocap.models.registry import register_model  # noqa: E402

register_model("stgcn", STGCN_Model)
