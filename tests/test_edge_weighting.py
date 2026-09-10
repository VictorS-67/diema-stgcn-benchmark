"""Tests for STGCN's ``edge_weighting`` modes.

The four modes mirror PySKL / mmaction2's ``unit_gcn(adaptive=...)``
(``pyskl/models/gcns/utils/gcn.py``), so a result here is comparable to the
wider skeleton-action-recognition literature:

    none        fixed buffer                     skeleton is a hard constraint
    init        A is the parameter               skeleton is a warm start
    offset      A + PA, PA ~ U(-1e-6, 1e-6)      additive correction
    importance  A * PA, PA = 1                   original ST-GCN edge importance

What these tests pin: the per-mode parameterisation, that the learned state is
**per block** (every reference implementation learns the graph per layer), that
``importance`` structurally cannot invent edges while ``init`` / ``offset``
can, and that the four modes share identical conv initialisation for a given
seed — without which a mode-vs-mode comparison would be confounded by the draw.
"""

import pytest
import torch

from emo_mocap.models.stgcn.spatial_units import (
    ADAPTIVE_MODES,
    Basic_GCN_Unit,
    resolve_edge_weighting,
)
from emo_mocap.models.stgcn.stgcn_model import STGCN_Model
from tests.conftest import DIEMA_ROT_INWARD, DIEMA_ROT_NUM_NODES


def _model(mode, **kw):
    torch.manual_seed(42)
    return STGCN_Model(
        num_class=7, edge_index=DIEMA_ROT_INWARD, num_nodes=DIEMA_ROT_NUM_NODES,
        in_channels=3, plusplus=False, dropout=0.0, edge_weighting=mode, **kw,
    )


class TestModeResolution:
    @pytest.mark.parametrize("value, replacement", [(True, "init"), (False, "none")])
    def test_retired_bools_are_rejected_with_the_replacement(self, value, replacement):
        """`true` meant "A itself, shared across blocks" — not today's per-block
        `init`. Resolving it silently would answer a question nobody asked."""
        with pytest.raises(ValueError, match=replacement):
            resolve_edge_weighting(value)

    def test_strings_pass_through_case_insensitively(self):
        for mode in ADAPTIVE_MODES:
            assert resolve_edge_weighting(mode) == mode
            assert resolve_edge_weighting(mode.upper()) == mode

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="edge_weighting must be one of"):
            resolve_edge_weighting("adaptive")
        with pytest.raises(ValueError):
            Basic_GCN_Unit(3, 64, torch.zeros(3, 25, 25), adaptive="nope")

    def test_config_default_matches_pyskl(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            model=SimpleNamespace(num_class=7, in_channels=3),
            skeleton=SimpleNamespace(inward_edges=DIEMA_ROT_INWARD,
                                     num_nodes=DIEMA_ROT_NUM_NODES),
        )
        assert STGCN_Model.from_config(cfg).edge_weighting == "importance"


class TestParameterisation:
    @pytest.mark.parametrize("mode", ADAPTIVE_MODES)
    def test_forward_runs(self, mode):
        out = _model(mode)(torch.randn(2, 3, 32, 25))
        assert out["logits"].shape == (2, 7)

    def test_none_puts_no_graph_state_in_the_checkpoint(self):
        sd = _model("none").state_dict()
        assert not [k for k in sd if k.endswith((".A", ".PA"))]

    @pytest.mark.parametrize("mode, key", [("init", "A"), ("offset", "PA"),
                                           ("importance", "PA")])
    def test_learnable_modes_expose_one_key_per_block(self, mode, key):
        m = _model(mode)
        keys = [k for k in m.state_dict() if k.endswith("." + key)]
        assert len(keys) == len(m.st_gcn_networks) == 10

    def test_initial_values_match_pyskl(self):
        imp = _model("importance").st_gcn_networks[0].gcn1
        assert torch.equal(imp.PA, torch.ones_like(imp.PA))
        off = _model("offset").st_gcn_networks[0].gcn1
        assert off.PA.abs().max() <= 1e-6
        ini = _model("init").st_gcn_networks[0].gcn1
        assert isinstance(ini.A, torch.nn.Parameter)

    def test_effective_A_starts_equal_to_the_skeleton(self):
        """All four modes convolve with the same graph at step 0."""
        ref = _model("none").st_gcn_networks[0].gcn1.effective_A()
        for mode in ADAPTIVE_MODES:
            got = _model(mode).st_gcn_networks[0].gcn1.effective_A()
            assert torch.allclose(got, ref, atol=1e-5), mode


class TestPerLayerIndependence:
    """Every reference implementation learns the graph per layer, not once."""

    @pytest.mark.parametrize("mode, attr", [("init", "A"), ("offset", "PA"),
                                            ("importance", "PA")])
    def test_blocks_hold_distinct_parameters(self, mode, attr):
        # Compare the parameters themselves, not effective_A() — for
        # 'offset'/'importance' that returns a freshly allocated temporary,
        # and two temporaries can land on the same address once the first
        # is freed.
        m = _model(mode)
        params = [getattr(u.gcn1, attr) for u in m.st_gcn_networks]
        assert len({p.data_ptr() for p in params}) == len(params)

    def test_training_one_block_leaves_the_others_alone(self):
        m = _model("importance")
        first, second = m.st_gcn_networks[0].gcn1, m.st_gcn_networks[1].gcn1
        before = second.PA.detach().clone()
        with torch.no_grad():
            first.PA.add_(1.0)
        assert torch.equal(second.PA, before)


class TestSparsitySemantics:
    """The property that actually separates the modes."""

    def _grad_on_non_edges(self, mode):
        m = _model(mode)
        blk = m.st_gcn_networks[0].gcn1
        non_edge = blk.A.detach() == 0
        m(torch.randn(4, 3, 32, 25))["logits"].sum().backward()
        grad = (blk.A if mode == "init" else blk.PA).grad
        return int((grad[non_edge] != 0).sum()), int(non_edge.sum())

    def test_importance_cannot_invent_edges(self):
        """dL/dPA = dL/dA * A, which is exactly 0 where the skeleton has no bone."""
        grown, total = self._grad_on_non_edges("importance")
        assert total > 0 and grown == 0

    @pytest.mark.parametrize("mode", ["init", "offset"])
    def test_init_and_offset_can_invent_edges(self, mode):
        """Documented, deliberate behaviour — two of PySKL's four modes do this."""
        grown, total = self._grad_on_non_edges(mode)
        assert grown == total > 0


class TestComparability:
    def test_modes_share_conv_initialisation(self):
        """A mode sweep must isolate the graph, not the weight draw.

        PA is built after the conv init precisely so `nn.init.uniform_` in
        'offset' mode can't shift every subsequent conv's draw.
        """
        ref = _model("none").st_gcn_networks[0].gcn1.conv_d[0].weight
        for mode in ADAPTIVE_MODES:
            got = _model(mode).st_gcn_networks[0].gcn1.conv_d[0].weight
            assert torch.equal(got, ref), mode

    def test_param_count_delta_is_only_the_graph(self):
        n_none = sum(p.numel() for p in _model("none").parameters())
        for mode in ["init", "offset", "importance"]:
            n = sum(p.numel() for p in _model(mode).parameters())
            # 10 blocks x (3 subsets x 25 x 25)
            assert n - n_none == 10 * 3 * 25 * 25, mode


class TestWeightDecayExclusion:
    """The learned graph must not be dragged toward zero by weight decay.

    Under `importance` the mask starts at 1, so decay pushes it toward
    deleting every edge; under `init` it shrinks the adjacency itself. The
    modes are not equally affected, so leaving decay on would confound a
    mode comparison as well as being the wrong prior.
    """

    def _lit(self, mode, weight_decay=5e-4):
        # step scheduler, not cosine: cosine reads self.trainer.max_epochs and
        # these tests call configure_optimizers() without a Trainer attached.
        from emo_mocap.training.lightning_model import LightningModel
        return LightningModel(_model(mode), base_lr=0.1, num_class=7,
                              weight_decay=weight_decay,
                              scheduler_type="step", scheduler_params=[10, 0.1])

    def test_graph_terms_are_declared_exempt(self):
        for mode, suffix in [("init", "gcn1.A"), ("offset", "gcn1.PA"),
                             ("importance", "gcn1.PA")]:
            names = _model(mode).no_weight_decay()
            assert len(names) == 10, mode
            assert all(n.endswith(suffix) for n in names), mode

    def test_none_declares_nothing(self):
        assert _model("none").no_weight_decay() == set()

    def test_optimizer_puts_the_graph_in_a_zero_decay_group(self):
        lit = self._lit("importance")
        groups = lit.configure_optimizers()[0][0].param_groups
        decayed = {g["weight_decay"] for g in groups}
        assert decayed == {5e-4, 0.0}
        exempt = [g for g in groups if g["weight_decay"] == 0.0][0]
        assert len(exempt["params"]) == 10
        assert all(p.numel() == 3 * 25 * 25 for p in exempt["params"])

    def test_every_parameter_lands_in_exactly_one_group(self):
        lit = self._lit("importance")
        groups = lit.configure_optimizers()[0][0].param_groups
        grouped = [id(p) for g in groups for p in g["params"]]
        assert len(grouped) == len(set(grouped))
        assert set(grouped) == {id(p) for p in lit.parameters() if p.requires_grad}

    def test_single_group_when_nothing_is_exempt(self):
        groups = self._lit("none").configure_optimizers()[0][0].param_groups
        assert len(groups) == 1 and groups[0]["weight_decay"] == 5e-4

    def test_a_bad_declaration_fails_loudly(self):
        lit = self._lit("importance")
        lit.model.no_weight_decay = lambda: {"not.a.parameter"}
        with pytest.raises(ValueError, match="do not exist"):
            lit.configure_optimizers()
