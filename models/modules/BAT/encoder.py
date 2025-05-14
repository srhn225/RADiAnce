#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .bat import BlockAdaptiveTransformer

from utils.gnn_utils import graph_to_batch


class BATEncoder(nn.Module):
    def __init__(self, hidden_size,
                 n_rbf=1, cutoff=7.0, edge_size=16, n_layers=3,
                 n_head=1, dropout=0.1) -> None:
        super().__init__()

        self.encoder = BlockAdaptiveTransformer(
            d_scaler=hidden_size,
            d_vector=hidden_size,
            d_hidden=hidden_size,
            n_head=n_head,
            n_layers=n_layers,
            d_rbf=n_rbf,
            cutoff=cutoff,
            d_edge=edge_size,
            update_coord=True
        )


    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, mask = graph_to_batch(H, block_id, mask_is_pad=False)
        Z, _ = graph_to_batch(Z, block_id, mask_is_pad=False)
        N, L, hidden_size = H.shape
        V = torch.zeros((N, L, hidden_size, 3), dtype=torch.float, device=H.device)
        H, V, Z = self.encoder(H, V, Z, mask, edges, edge_attr)
        H, Z = H[mask], Z[mask]
        block_repr = scatter_sum(H, block_id, dim=0)           # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        # graph_repr = scatter_mean(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, Z