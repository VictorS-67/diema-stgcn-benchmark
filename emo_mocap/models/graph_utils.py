"""Generic graph utilities used across model families.

Lives at peer-level under `emo_mocap/models/` so any model package
(STGCN, ProtoGCN, future architectures) can use these without reaching
into another model's namespace.
"""

import torch


def to_dense_adj(edge_index, num_nodes):
    """Convert an `(E, 2)` edge index into a dense `(num_nodes, num_nodes)` adjacency matrix."""
    #assume edge_index is a tensor already
    effective_num_nodes = torch.tensor([int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0])
    assert effective_num_nodes.item() <= num_nodes

    adj = torch.zeros((num_nodes, num_nodes))

    for (i, j) in edge_index:
        adj[i, j] = 1

    return adj
