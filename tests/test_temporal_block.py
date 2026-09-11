"""Tests for which temporal block ``model.plusplus`` selects, and the default.

Each ST-GCN layer ends in a temporal convolution. ``plusplus=False`` gives the
original single 9-frame convolution; ``True`` gives the STGCN++ multi-branch
unit (a 1x1 convolution, a max-pool and four dilated 3-frame convolutions).

What these tests pin is the **default**, because it is not an arbitrary choice:
every number reported for DIEM-A was measured with the basic block, and the
multi-branch block has never been evaluated on this task. A config that omits
the key must therefore get the basic block. Upstream, the constructor and the
config loader once disagreed about this default and a whole campaign ran on a
block its author did not think was under test; the shipped configs set the key
explicitly, so nothing but a test like this would catch the default drifting.
"""

import torch

from emo_mocap.models.stgcn.ST_units import Basic_TCN_Unit, TCN_Unit_plus
from emo_mocap.models.stgcn.stgcn_model import STGCN_Model
from emo_mocap.tools.config import load_config
from tests.conftest import DIEMA_ROT_INWARD, DIEMA_ROT_NUM_NODES

RECIPES = ["configs/diema7_stgcn_recipe.yaml", "configs/diema13_stgcn_recipe.yaml"]


def _blocks(model):
    return [unit.tcn1 for unit in model.st_gcn_networks]


def _model(**kw):
    torch.manual_seed(42)
    return STGCN_Model(
        num_class=7, edge_index=DIEMA_ROT_INWARD, num_nodes=DIEMA_ROT_NUM_NODES,
        in_channels=3, dropout=0.0, **kw,
    )


class TestTheDefaultIsTheMeasuredBlock:
    def test_constructor_default_is_the_basic_block(self):
        assert all(isinstance(b, Basic_TCN_Unit) for b in _blocks(_model()))

    def test_from_config_default_is_the_basic_block(self, tmp_path):
        """A config that says nothing about the temporal block gets the one the
        reported numbers come from, not the unevaluated one."""
        path = tmp_path / "no_plusplus.yaml"
        path.write_text(
            "data: {data_path: x.npz}\n"
            "model: {type: stgcn, num_class: 7, in_channels: 3}\n"
            "skeleton: {num_nodes: 25, inward_edges: [[0, 1]]}\n"
        )
        model = STGCN_Model.from_config(load_config(path))
        assert all(isinstance(b, Basic_TCN_Unit) for b in _blocks(model))

    def test_shipped_recipes_still_ask_for_the_basic_block(self):
        for recipe in RECIPES:
            assert load_config(recipe).model.plusplus is False, recipe


class TestTheFlagStillSelects:
    def test_true_builds_the_multi_branch_block(self):
        assert all(isinstance(b, TCN_Unit_plus)
                   for b in _blocks(_model(plusplus=True)))

    def test_the_multi_branch_block_is_the_smaller_one(self):
        """Its branches split the channels, so it is cheaper despite seeing
        further in time. A silent flip of the default would move this by 2x."""
        size = lambda m: sum(p.numel() for p in m.parameters())
        assert size(_model(plusplus=True)) < size(_model(plusplus=False))
