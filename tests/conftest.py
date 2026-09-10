"""Shared fixtures for emo_mocap tests."""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Skeleton edge lists (hardcoded from emo_recognition.ipynb)
# ---------------------------------------------------------------------------

DIEMA_ROT_INWARD = [
    (0, 1),
    (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7),
    (9, 5), (10, 9), (11, 10), (12, 11),
    (13, 5), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19),
    (21, 1), (22, 21), (23, 22), (24, 23),
]
DIEMA_ROT_NUM_NODES = 25

NTU_INWARD_ORI = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
]
NTU_INWARD = [(i - 1, j - 1) for i, j in NTU_INWARD_ORI]
NTU_NUM_NODES = 25


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--gen-golden", action="store_true", default=False,
        help="Regenerate golden master output files",
    )


@pytest.fixture
def gen_golden(request):
    return request.config.getoption("--gen-golden")


@pytest.fixture
def diema_inward_edges():
    return DIEMA_ROT_INWARD


@pytest.fixture
def ntu_inward_edges():
    return NTU_INWARD


@pytest.fixture
def num_nodes_diema():
    return DIEMA_ROT_NUM_NODES


@pytest.fixture
def num_nodes_ntu():
    return NTU_NUM_NODES


@pytest.fixture
def sample_input():
    """Factory fixture: returns a function that creates random tensors."""
    def _make(N=2, C=3, T=64, V=25):
        return torch.randn(N, C, T, V)
    return _make


@pytest.fixture
def fixed_seed():
    """Set deterministic seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


@pytest.fixture
def diema_A():
    """Build the DIEMA rotation adjacency matrix (GraphAAGCN)."""
    from emo_mocap.models.stgcn.adj_matrix import GraphAAGCN
    graph = GraphAAGCN(DIEMA_ROT_INWARD, DIEMA_ROT_NUM_NODES)
    return graph.A


@pytest.fixture
def ntu_A():
    """Build the NTU adjacency matrix (GraphAAGCN)."""
    from emo_mocap.models.stgcn.adj_matrix import GraphAAGCN
    graph = GraphAAGCN(NTU_INWARD, NTU_NUM_NODES)
    return graph.A
