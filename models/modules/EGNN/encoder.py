#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import utils.register as R

from models.modules.GET.tools import _unit_edges_from_block_edges
from utils.nn_utils import std_conserve_scatter_sum
from .egnn import EGNN


class EGNNEncoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            edge_size,
            n_layers,
            sparse_k=3,
        ) -> None:
        super().__init__()

        self.encoder = EGNN(
            in_node_nf = hidden_size,
            hidden_nf = hidden_size,
            out_node_nf = hidden_size,
            in_edge_nf = edge_size,
            n_layers = n_layers,
        )
        self.sparse_k = sparse_k


    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None):
        (atom_row, atom_col), (edge_id, _, _) = _unit_edges_from_block_edges(block_id, edges.T, Z.unsqueeze(1), self.sparse_k)
        atom_edges = torch.stack([atom_row, atom_col], dim=0) # [2, Eu]
        atom_edge_attr = edge_attr[edge_id] if edge_attr is not None else None

        if topo_edges is not None: atom_edges = torch.cat([atom_edges, topo_edges], dim=-1)
        if topo_edge_attr is not None:
            if atom_edge_attr is None: atom_edge_attr = topo_edge_attr
            else: atom_edge_attr = torch.cat([atom_edge_attr, topo_edge_attr], dim=0) # [Eu, d_edge]

        H, Z = self.encoder(H, Z, atom_edges, atom_edge_attr)

        return H, Z
