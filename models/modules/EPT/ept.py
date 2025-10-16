#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum, scatter_std

import utils.register as R

from utils.nn_utils import stable_norm, std_conserve_scatter_sum, graph_to_batch_nx

from ..GET.tools import _unit_edges_from_block_edges
from .radial_basis import RadialBasis
import numpy as np
try:
    from xformers.ops import memory_efficient_attention as attn_func
    xformers_enable = True
except:
    xformers_enable = False


@R.register('XTransEncoderAct')
class XTransEncoderAct(nn.Module):
    def __init__(self, hidden_size, ffn_size, n_rbf, cutoff=7.0, z_requires_grad=False, 
                 edge_size=16, n_layers=3, n_head=4, pre_norm=False, use_edge_feat=False, sparse_k=3, local_mask=False, attn_bias=True,
                 efficient=False, vector_act='none', 
                 # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
        ) -> None:
        super().__init__()

        self.encoder = Transformer(
            d_hidden = hidden_size, d_ffn = ffn_size, n_heads = n_head, n_layers = n_layers,
            n_rbf = n_rbf, d_edge = edge_size, cutoff = cutoff, use_edge_feat = use_edge_feat, local_mask = local_mask, attn_bias = attn_bias,
            layer_norm = 'pre' if pre_norm else 'post', sparse_k = sparse_k, efficient = efficient,
            vector_act = vector_act, 
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z
    
@R.register('XTransEncoderActrag')
class XTransEncoderActrag(nn.Module):
    def __init__(self, hidden_size, ffn_size, n_rbf, cutoff=7.0, z_requires_grad=False, 
                 edge_size=16, n_layers=3, n_head=4, pre_norm=False, use_edge_feat=False, sparse_k=3, local_mask=False, attn_bias=True,
                 efficient=False, vector_act='none', 
                 # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
        ) -> None:
        super().__init__()

        self.encoder = Transformerrag(
            d_hidden = hidden_size, d_ffn = ffn_size, n_heads = n_head, n_layers = n_layers,
            n_rbf = n_rbf, d_edge = edge_size, cutoff = cutoff, use_edge_feat = use_edge_feat, local_mask = local_mask, attn_bias = attn_bias,
            layer_norm = 'pre' if pre_norm else 'post', sparse_k = sparse_k, efficient = efficient,
            vector_act = vector_act, 
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask,prompt_feature=prompt_feature)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z


@R.register('XTransEncoderActincontext')
class XTransEncoderActincontext(nn.Module):
    def __init__(self, hidden_size, ffn_size, n_rbf, cutoff=7.0, z_requires_grad=False, 
                 edge_size=16, n_layers=3, n_head=4, pre_norm=False, use_edge_feat=False, sparse_k=3, local_mask=False, attn_bias=True,
                 efficient=False, vector_act='none', 
                 # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
        ) -> None:
        super().__init__()

        self.encoder = Transformerragincontext(
            d_hidden = hidden_size, d_ffn = ffn_size, n_heads = n_head, n_layers = n_layers,
            n_rbf = n_rbf, d_edge = edge_size, cutoff = cutoff, use_edge_feat = use_edge_feat, local_mask = local_mask, attn_bias = attn_bias,
            layer_norm = 'pre' if pre_norm else 'post', sparse_k = sparse_k, efficient = efficient,
            vector_act = vector_act, 
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask,prompt_feature=prompt_feature)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z
@R.register('XTransEncoderActadaLN')
class XTransEncoderActadaLN(nn.Module):
    def __init__(self, hidden_size, ffn_size, n_rbf, cutoff=7.0, z_requires_grad=False, 
                 edge_size=16, n_layers=3, n_head=4, pre_norm=False, use_edge_feat=False, sparse_k=3, local_mask=False, attn_bias=True,
                 efficient=False, vector_act='none', 
                 # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
        ) -> None:
        super().__init__()

        self.encoder = TransformeradaLN(
            d_hidden = hidden_size, d_ffn = ffn_size, n_heads = n_head, n_layers = n_layers,
            n_rbf = n_rbf, d_edge = edge_size, cutoff = cutoff, use_edge_feat = use_edge_feat, local_mask = local_mask, attn_bias = attn_bias,
            layer_norm = 'pre' if pre_norm else 'post', sparse_k = sparse_k, efficient = efficient,
            vector_act = vector_act, 
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask,prompt_feature=prompt_feature)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z
@R.register('XTransEncoderActadaLNAttn')
class XTransEncoderActadaLNAttn(nn.Module):
    def __init__(self, hidden_size, ffn_size, n_rbf, cutoff=7.0, z_requires_grad=False, 
                 edge_size=16, n_layers=3, n_head=4, pre_norm=False, use_edge_feat=False, sparse_k=3, local_mask=False, attn_bias=True,
                 efficient=False, vector_act='none', 
                 # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
        ) -> None:
        super().__init__()

        self.encoder = TransformeradaLNAttn(
            d_hidden = hidden_size, d_ffn = ffn_size, n_heads = n_head, n_layers = n_layers,
            n_rbf = n_rbf, d_edge = edge_size, cutoff = cutoff, use_edge_feat = use_edge_feat, local_mask = local_mask, attn_bias = attn_bias,
            layer_norm = 'pre' if pre_norm else 'post', sparse_k = sparse_k, efficient = efficient,
            vector_act = vector_act, 
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask,prompt_feature=prompt_feature)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z
class Transformerrag(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            n_layers,
            n_rbf,
            d_edge,
            cutoff=7.0,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            use_edge_feat = False,
            local_mask = False,
            attn_bias = True,
            sparse_k=None,
            efficient=False,
            vector_act='none',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print("xformers are not downloaded, change into custom attention mechanism. "
                  "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                  "or seek 'https://github.com/facebookresearch/xformers' for more details.")
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden)
        )

        self.final_v = GVPFFNLayer(
            d_hidden, d_ffn, act_fn, d_output=1
        )

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)    
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == 'pre':
            self.ln = nn.LayerNorm(d_hidden)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EPTLayerrag(
                d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
            ))#n_layers: 6

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None: edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        if edge_attr is not None: edge_attr = edge_attr[block_edge_id]
        
        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0) # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col] # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat) # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2) # [Ne, 3, d_hidden]
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device) * float('-inf')
            rbf_feat_batch[~H_mask] = 0.0 # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device)
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float('-inf')
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(3,0,4,1,2) # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1) # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)    

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f'layer_{i}'](
                H_batch, V_batch, cached_info,prompt_feature
            )
        
        if self.layer_norm == 'pre':
            H_batch = self.ln(H_batch)


        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph

class TransformeradaLNAttn(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            n_layers,
            n_rbf,
            d_edge,
            cutoff=7.0,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            use_edge_feat = False,
            local_mask = False,
            attn_bias = True,
            sparse_k=None,
            efficient=False,
            vector_act='none',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print("xformers are not downloaded, change into custom attention mechanism. "
                  "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                  "or seek 'https://github.com/facebookresearch/xformers' for more details.")
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden)
        )

        self.final_v = GVPFFNLayer(
            d_hidden, d_ffn, act_fn, d_output=1
        )

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)    
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == 'pre':
            self.ln = nn.LayerNorm(d_hidden)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EPTLayerAdaLNAttn(
                d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
            ))#n_layers: 6

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None: edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        if edge_attr is not None: edge_attr = edge_attr[block_edge_id]
        
        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0) # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col] # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat) # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2) # [Ne, 3, d_hidden]
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device) * float('-inf')
            rbf_feat_batch[~H_mask] = 0.0 # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device)
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float('-inf')
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(3,0,4,1,2) # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1) # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)    

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f'layer_{i}'](
                H_batch, V_batch, cached_info,prompt_feature
            )
        
        if self.layer_norm == 'pre':
            H_batch = self.ln(H_batch)


        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph
class TransformeradaLN(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            n_layers,
            n_rbf,
            d_edge,
            cutoff=7.0,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            use_edge_feat = False,
            local_mask = False,
            attn_bias = True,
            sparse_k=None,
            efficient=False,
            vector_act='none',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print("xformers are not downloaded, change into custom attention mechanism. "
                  "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                  "or seek 'https://github.com/facebookresearch/xformers' for more details.")
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden)
        )

        self.final_v = GVPFFNLayer(
            d_hidden, d_ffn, act_fn, d_output=1
        )

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)    
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == 'pre':
            self.ln = nn.LayerNorm(d_hidden)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EPTLayerAdaLNZero(
                d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
            ))#n_layers: 6

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None: edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        if edge_attr is not None: edge_attr = edge_attr[block_edge_id]
        
        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0) # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col] # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat) # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2) # [Ne, 3, d_hidden]
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device) * float('-inf')
            rbf_feat_batch[~H_mask] = 0.0 # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device)
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float('-inf')
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(3,0,4,1,2) # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1) # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)    

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f'layer_{i}'](
                H_batch, V_batch, cached_info,prompt_feature
            )
        
        if self.layer_norm == 'pre':
            H_batch = self.ln(H_batch)


        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph
class Transformerragincontext(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            n_layers,
            n_rbf,
            d_edge,
            cutoff=7.0,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            use_edge_feat = False,
            local_mask = False,
            attn_bias = True,
            sparse_k=None,
            efficient=False,
            vector_act='none',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print("xformers are not downloaded, change into custom attention mechanism. "
                  "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                  "or seek 'https://github.com/facebookresearch/xformers' for more details.")
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden)
        )

        self.final_v = GVPFFNLayer(
            d_hidden, d_ffn, act_fn, d_output=1
        )

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)    
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == 'pre':
            self.ln = nn.LayerNorm(d_hidden)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EPTLayerInContextConditioning(
                d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
            ))#n_layers: 6

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None,prompt_feature=None):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None: edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        if edge_attr is not None: edge_attr = edge_attr[block_edge_id]
        
        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0) # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col] # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat) # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2) # [Ne, 3, d_hidden]
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device) * float('-inf')
            rbf_feat_batch[~H_mask] = 0.0 # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device)
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float('-inf')
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(3,0,4,1,2) # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1) # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)    

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f'layer_{i}'](
                H_batch, V_batch, cached_info,prompt_feature
            )
        
        if self.layer_norm == 'pre':
            H_batch = self.ln(H_batch)


        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph

class Transformer(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            n_layers,
            n_rbf,
            d_edge,
            cutoff=7.0,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            use_edge_feat = False,
            local_mask = False,
            attn_bias = True,
            sparse_k=None,
            efficient=False,
            vector_act='none',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print("xformers are not downloaded, change into custom attention mechanism. "
                  "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                  "or seek 'https://github.com/facebookresearch/xformers' for more details.")
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden)
        )

        self.final_v = GVPFFNLayer(
            d_hidden, d_ffn, act_fn, d_output=1
        )

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)    
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == 'pre':
            self.ln = nn.LayerNorm(d_hidden)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EPTLayer(
                d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
            ))

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None: edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        if edge_attr is not None: edge_attr = edge_attr[block_edge_id]
        
        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0) # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col] # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat) # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2) # [Ne, 3, d_hidden]
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device) * float('-inf')
            rbf_feat_batch[~H_mask] = 0.0 # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device)
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float('-inf')
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(3,0,4,1,2) # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1) # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)    

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f'layer_{i}'](
                H_batch, V_batch, cached_info
            )
        
        if self.layer_norm == 'pre':
            H_batch = self.ln(H_batch)


        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph


class EPTLayer(nn.Module):
    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            layer_idx=-1,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            efficient = False,
            vector_act = 'none',
            attn_bias = True
        ):
        super(EPTLayer, self).__init__()
        self.attn_layer = SubLayerWrapper(
            SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias = attn_bias),
            d_hidden,
            layer_norm,
            residual
        )
        self.ffn_layer = SubLayerWrapper(
            GVPFFNLayer(d_hidden, d_ffn, act_fn, vector_act = vector_act),
            d_hidden,
            layer_norm,
            residual
        )
        self.layer_idx = layer_idx

    def forward(self, H, V, cached_info=None):

        H, V = self.attn_layer(H, V, cached_info=cached_info)
        H, V = self.ffn_layer(H, V)

        return H, V


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_hidden, n_heads=4, layer_idx=-1):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.layer_idx = layer_idx



        self.attn = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=n_heads, batch_first=True)
        self.scaler_o = nn.Linear(d_hidden, d_hidden)

    def forward(self, H, cached_info, prompt_feature):
        D_batch, rbf_feat_batch, H_mask = cached_info
        B, N, D = H.shape
        P = prompt_feature.shape[0] // B
        prompt_feature = prompt_feature.reshape(B, P, D)



        res, attn_weight = self.attn(H, prompt_feature, prompt_feature)
        # torch.set_printoptions(profile="full")
        # # # print(attn_weight.mean(dim=1))
        # print(attn_weight)
        return self.scaler_o(res)


class EPTLayerrag(nn.Module):
    def __init__(
        self,
        d_hidden,
        d_ffn,
        n_heads,
        layer_idx=-1,
        act_fn=nn.SiLU(),
        layer_norm='pre',
        residual=True,
        efficient=False,
        vector_act='none',
        attn_bias=True,
        use_cross_attn=True
    ):
        super(EPTLayerrag, self).__init__()
        


        self.attn_layer = SubLayerWrapper(
            SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias=attn_bias),
            d_hidden, layer_norm, residual
        )
        


        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn_layer = SubLayerWrapperH(
                CrossAttentionLayer(d_hidden, n_heads),  
                d_hidden, layer_norm, residual
            )
        
        self.ffn_layer = SubLayerWrapper(
            GVPFFNLayer(d_hidden, d_ffn, act_fn, vector_act=vector_act),
            d_hidden, layer_norm, residual
        )
        self.layer_idx = layer_idx

    def forward(self, H, V, cached_info=None, prompt_feature=None,generate_mask=None):


        H, V = self.attn_layer(H, V, cached_info=cached_info)
        


        # prompt_feature=None
        if self.use_cross_attn and prompt_feature is not None:


            H = self.cross_attn_layer(H, cached_info=cached_info,prompt_feature=prompt_feature)
            # print(H.shape)
        else:


            pass
            


        H, V = self.ffn_layer(H, V)
        
        return H, V


class AdaLNZero(nn.Module):
    def __init__(self, d_hidden, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden, elementwise_affine=False)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_hidden)
        )


        nn.init.zeros_(self.cond_mlp[1].weight)
        nn.init.zeros_(self.cond_mlp[1].bias)

    def forward(self, x, cond):
        # x: (batch, seq_len, d_hidden)
        # cond: (batch, cond_len, cond_dim)
        x_norm = self.norm(x)


        cond_mean = cond.mean(dim=1)  # (batch, cond_dim)
        cond_out = self.cond_mlp(cond_mean).unsqueeze(1)  # (batch, 1, 2*d_hidden)
        scale, shift = cond_out.chunk(2, dim=-1)
        return x_norm * (1 + scale) + shift

class AdaLNZeroAttn(nn.Module):

    def __init__(self, d_hidden: int, cond_dim: int, n_heads: int = 4):
        super().__init__()
        assert d_hidden % n_heads == 0, "n_heads must divide d_hidden"

        self.n_heads = n_heads
        self.d_head  = d_hidden // n_heads



        self.norm = nn.LayerNorm(d_hidden, elementwise_affine=False)



        self.q_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.k_proj = nn.Linear(cond_dim, d_hidden, bias=False)
        self.v_proj = nn.Linear(cond_dim, d_hidden, bias=False)



        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_hidden, 2 * d_hidden)
        )
        nn.init.zeros_(self.cond_mlp[1].weight)
        nn.init.zeros_(self.cond_mlp[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x    : (B, L, d_hidden)
        cond : (B, M, cond_dim)
        """
        B, L, _ = x.shape
        M = cond.size(1)



        x_norm = self.norm(x)                                   # (B, L, d_h)



        def _reshape(t, proj):
            t = proj(t)
            B, S, D = t.shape
            return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = _reshape(x_norm, self.q_proj)   # (B, H, L, d_head)
        k = _reshape(cond,   self.k_proj)   # (B, H, M, d_head)
        v = _reshape(cond,   self.v_proj)   # (B, H, M, d_head)

        # 3) scaled‑dot‑product attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1))       # (B, H, L, M)
        attn = (attn_logits / math.sqrt(self.d_head)).softmax(dim=-1)



        ctx = torch.matmul(attn, v)                              # (B, H, L, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, -1)    # (B, L, d_hidden)



        scale_shift = self.cond_mlp(ctx)                         # (B, L, 2*d_h)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x_norm * (1.0 + scale) + shift
class EPTLayerAdaLNZero(nn.Module):
    """
    DiT-like block with adaLN-Zero conditioning for both attention and feed\-forward,
    including residual scaling by learned alpha parameters.
    """
    def __init__(
        self,
        d_hidden,
        d_ffn,
        n_heads,
        layer_idx=-1,
        act_fn=nn.SiLU(),
        layer_norm='pre',
        residual=True,
        efficient=False,
        vector_act='none',
        attn_bias=True,
    ):
        super().__init__()
        # AdaLN-Zero before self-attn
        self.adaln_attn = AdaLNZero(d_hidden, d_hidden)
        # Self-attention
        self.self_attn = SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias=attn_bias)
        # alpha1: residual scaling after attention
        self.alpha_attn = nn.Linear(d_hidden, d_hidden)
        nn.init.zeros_(self.alpha_attn.weight)
        nn.init.zeros_(self.alpha_attn.bias)
        # AdaLN-Zero before ffn
        self.adaln_ffn = AdaLNZero(d_hidden, d_hidden)
        # Feed-forward
        self.ffn = GVPFFNLayer(d_hidden, d_ffn, act_fn,vector_act=vector_act)
        # alpha2: residual scaling after ffn
        self.alpha_ffn = nn.Linear(d_hidden, d_hidden)
        nn.init.zeros_(self.alpha_ffn.weight)
        nn.init.zeros_(self.alpha_ffn.bias)
        self.residual = residual

    def forward(self, H, V, cached_info=None, prompt_feature=None):
        # H: (batch, seq, hidden_dim)
        # V: (batch, seq, vec_dim)
        # cond: (batch, cond_len, cond_dim)
        # Self-Attention block
        # 1. AdaLN-Zero
        B, N, D = H.shape
        P = prompt_feature.shape[0] // B
        prompt_feature = prompt_feature.reshape(B, P, D)
        H_norm = self.adaln_attn(H, prompt_feature)
        # 2. Attention
        H_attn, V = self.self_attn(H_norm, V, cached_info=cached_info)
        # 3. Residual scaling
        alpha1 = self.alpha_attn(prompt_feature.mean(dim=1)).unsqueeze(1)  # (batch,1,hidden_dim)
        H = H + alpha1 * H_attn if self.residual else alpha1 * H_attn

        # Feed-Forward block
        # 1. AdaLN-Zero
        H_norm2 = self.adaln_ffn(H, prompt_feature)
        # 2. FFN
        H_ffn, V = self.ffn(H_norm2, V)
        # 3. Residual scaling
        alpha2 = self.alpha_ffn(prompt_feature.mean(dim=1)).unsqueeze(1)
        H = H + alpha2 * H_ffn if self.residual else alpha2 * H_ffn

        return H, V
class EPTLayerAdaLNAttn(nn.Module):
    """
    DiT-like block with adaLN-Zero conditioning for both attention and feed\-forward,
    including residual scaling by learned alpha parameters.
    """
    def __init__(
        self,
        d_hidden,
        d_ffn,
        n_heads,
        layer_idx=-1,
        act_fn=nn.SiLU(),
        layer_norm='pre',
        residual=True,
        efficient=False,
        vector_act='none',
        attn_bias=True,
    ):
        super().__init__()
        # AdaLN-Zero before self-attn
        self.adaln_attn = AdaLNZeroAttn(d_hidden, d_hidden)
        # Self-attention
        self.self_attn = SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias=attn_bias)
        # alpha1: residual scaling after attention
        self.alpha_attn = nn.Linear(d_hidden, d_hidden)
        nn.init.zeros_(self.alpha_attn.weight)
        nn.init.zeros_(self.alpha_attn.bias)
        # AdaLN-Zero before ffn
        self.adaln_ffn = AdaLNZeroAttn(d_hidden, d_hidden)
        # Feed-forward
        self.ffn = GVPFFNLayer(d_hidden, d_ffn, act_fn,vector_act=vector_act)
        # alpha2: residual scaling after ffn
        self.alpha_ffn = nn.Linear(d_hidden, d_hidden)
        nn.init.zeros_(self.alpha_ffn.weight)
        nn.init.zeros_(self.alpha_ffn.bias)
        self.residual = residual

    def forward(self, H, V, cached_info=None, prompt_feature=None):
        # H: (batch, seq, hidden_dim)
        # V: (batch, seq, vec_dim)
        # cond: (batch, cond_len, cond_dim)
        # Self-Attention block
        # 1. AdaLN-Zero
        B, N, D = H.shape
        P = prompt_feature.shape[0] // B
        prompt_feature = prompt_feature.reshape(B, P, D)
        H_norm = self.adaln_attn(H, prompt_feature)
        # 2. Attention
        H_attn, V = self.self_attn(H_norm, V, cached_info=cached_info)
        # 3. Residual scaling
        alpha1 = self.alpha_attn(prompt_feature.mean(dim=1)).unsqueeze(1)  # (batch,1,hidden_dim)
        H = H + alpha1 * H_attn if self.residual else alpha1 * H_attn

        # Feed-Forward block
        # 1. AdaLN-Zero
        H_norm2 = self.adaln_ffn(H, prompt_feature)
        # 2. FFN
        H_ffn, V = self.ffn(H_norm2, V)
        # 3. Residual scaling
        alpha2 = self.alpha_ffn(prompt_feature.mean(dim=1)).unsqueeze(1)
        H = H + alpha2 * H_ffn if self.residual else alpha2 * H_ffn

        return H, V 
class SelfAttnLayer(nn.Module):

    def __init__(self, d_hidden, n_heads, layer_idx=-1, efficient=False, attn_bias=True):

        super(SelfAttnLayer, self).__init__()

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = self.d_hidden // self.n_heads
        self.layer_idx = layer_idx
        self.factor = 0.5 / math.sqrt(self.d_head)
        self.efficient = efficient
        self.scaler_q = nn.Linear(d_hidden, d_hidden * 4, bias=attn_bias)
        self.scaler_k = nn.Linear(d_hidden, d_hidden * 4, bias=attn_bias)
        self.scaler_v = nn.Linear(d_hidden, d_hidden, bias=attn_bias)
        self.vector_v = nn.Linear(d_hidden, d_hidden, bias = False)
        self.scaler_o = nn.Linear(d_hidden, d_hidden)
        self.vector_o = nn.Linear(d_hidden, d_hidden, bias = False)

    def forward(self, H, V, cached_info=None):

        # H : [B, N, d_hidden]
        # V : [B, N, 3, d_hidden]

        batch_size, num_nodes = H.shape[0], H.shape[1]

        D_batch, rbf_feat_batch, H_mask = cached_info  

        H_q = self.scaler_q(H).view(batch_size, num_nodes, self.n_heads, -1)
        H_k = self.scaler_k(H).view(batch_size, num_nodes, self.n_heads, -1)
        H_v = self.scaler_v(H).view(batch_size, num_nodes, self.n_heads, -1)
        V_v = self.vector_v(V).view(batch_size, num_nodes, 3, self.n_heads, -1).transpose(-2, -3).flatten(start_dim=-2) 
        V_attn = torch.cat([H_v, V_v], dim=-1)

        bias = rbf_feat_batch[self.layer_idx] + D_batch.unsqueeze(1)
        mask = H_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        bias = bias.masked_fill(mask == 0, float("-inf"))

        if not self.efficient:
            attn = torch.einsum('bnhd, bmhd -> bhnm', H_q, H_k)
            attn = F.softmax(attn * self.factor + bias, dim=-1)
            res = torch.einsum('bhnm, bmhd -> bnhd', attn, V_attn)

        else:
            res = attn_func(
                query = H_q,
                key = H_k,
                value = V_attn,
                attn_bias = bias.expand(-1, self.n_heads, -1, -1)
            )


        H_res = res[:, :, :, :self.d_head].reshape(batch_size, num_nodes, self.d_hidden)
        V_res = res[:, :, :, self.d_head:].reshape(batch_size, num_nodes, self.n_heads, 3, self.d_head).transpose(-2, -3).reshape(batch_size, num_nodes, 3, self.d_hidden)
        
        H_o = self.scaler_o(H_res) 
        V_o = self.vector_o(V_res)

        return H_o, V_o


class GVPFFNLayer(nn.Module):

    def __init__(self, d_hidden, d_ffn, act_fn=nn.SiLU(), d_output=None, vector_act='none'):

        super(GVPFFNLayer, self).__init__()

        self.d_hidden = d_hidden
        self.d_ffn = d_ffn
        self.act_fn = act_fn
        self.d_output = d_hidden if d_output is None else d_output

        self.linear_v = nn.Linear(d_hidden, d_hidden + self.d_output, bias=False)
        self.ffn_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_ffn),
            act_fn,
            nn.Linear(d_ffn, d_hidden + self.d_output)
        )

        self.vector_act = vector_act
        if self.vector_act == 'layernorm':
            self.vector_layernorm = nn.LayerNorm(self.d_output)

    def vector_act_func(self, Vs):
        if self.vector_act == 'none':
            return Vs
        elif self.vector_act == 'sigmoid':
            return F.sigmoid(Vs)
        elif self.vector_act == 'tanh':
            return F.tanh(Vs)
        elif self.vector_act == 'layernorm':
            return self.vector_layernorm(Vs)
        elif self.vector_act == 'one':
            return torch.ones_like(Vs)

    def forward(self, H, V):

        V_proj = self.linear_v(V)
        V1, V2 = V_proj[...,:self.d_hidden], V_proj[...,self.d_hidden:]
        scaler = torch.cat([H, V1.norm(dim=-2)], dim=-1)
        scaler_out = self.ffn_mlp(scaler)
        H_out, V_update = scaler_out[...,:self.d_hidden], scaler_out[...,self.d_hidden:]
        V_out = self.vector_act_func(V_update).unsqueeze(-2) * V2
        return H_out, V_out    
    
class EPTLayerInContextConditioning(nn.Module):
    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            layer_idx=-1,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            efficient = False,
            vector_act = 'none',
            attn_bias = True
    ):
        super().__init__()


        self.q_proj = nn.Linear(d_hidden, d_hidden)
        self.k_proj = nn.Linear(d_hidden, d_hidden)


        # self.v_proj = nn.Linear(d_hidden, d_hidden)


        mlp_hidden = (2 * d_hidden)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_hidden, mlp_hidden),
            act_fn,
            nn.Linear(mlp_hidden, d_hidden),
        )
        self.attn_layer = SubLayerWrapper(
            SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias = attn_bias),
            d_hidden,
            layer_norm,
            residual
        )


        self.ffn = SubLayerWrapper(
            GVPFFNLayer(d_hidden, d_ffn, act_fn, vector_act=vector_act),
            d_hidden, layer_norm='pre', residual=True
        )

    def forward(self, H, V, cached_info=None, prompt_feature=None):
        # H: [B, N, d_hidden]
        # prompt_feature: [B, P, d_hidden]
        B, N, D = H.shape
        P = prompt_feature.shape[0] // B
        prompt_feature = prompt_feature.reshape(B, P, D)
        if prompt_feature is not None:
            B, N, D = H.shape
            P = prompt_feature.size(1)



            Q = self.q_proj(H)                        # [B, N, D]
            K = self.k_proj(prompt_feature)           # [B, P, D]
            # bnp = (Q · K^T) / sqrt(D)
            attn_scores = torch.einsum('bnd,bpd->bnp', Q, K) / (D ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, N, P]



            # prompt_sel: [B, N, D] = attn_weights @ prompt_feature
            prompt_sel = torch.einsum('bnp,bpd->bnd', attn_weights, prompt_feature)



            H_cat = torch.cat([H, prompt_sel], dim=2)        # [B, N, 2D]
            H = self.fusion_mlp(H_cat)                       # [B, N, D]
        H, V = self.attn_layer(H, V, cached_info=cached_info)


        H, V = self.ffn(H, V)
        return H, V

class SubLayerWrapper(nn.Module):

    def __init__(self, sub_layer, d_hidden, layer_norm = 'pre', residual = True):

        super(SubLayerWrapper, self).__init__()
        self.sub_layer = sub_layer
        self.d_hidden = d_hidden
        self.layer_norm = layer_norm
        self.ln = nn.LayerNorm(d_hidden)
        self.residual = residual

    def forward(self, H, V, **kwargs):
        H0, V0 = H, V
        if self.layer_norm == 'pre':
            H = self.ln(H0)
        H, V = self.sub_layer(H, V, **kwargs)
        if self.residual:
            H = H + H0
            V = V + V0
        if self.layer_norm == 'post':
            H = self.ln(H)
        return H, V

class SubLayerWrapperH(nn.Module):

    def __init__(self, sub_layer, d_hidden, layer_norm = 'pre', residual = True):

        super(SubLayerWrapperH, self).__init__()
        self.sub_layer = sub_layer
        self.d_hidden = d_hidden
        self.layer_norm = layer_norm
        self.ln = nn.LayerNorm(d_hidden)
        self.residual = residual

    def forward(self, H, **kwargs):
        H0= H
        if self.layer_norm == 'pre':
            H = self.ln(H0)
        H= self.sub_layer(H, **kwargs)
        if self.residual:
            H = H + H0

        if self.layer_norm == 'post':
            H = self.ln(H)
        return H

if __name__ == '__main__':
    d_hidden = 64
    d_ffn = 16
    d_edge = 16
    n_rbf = 16
    n_heads = 4
    n_layers = 3
    device = torch.device('cuda:0')

    # d_hidden, d_ffn, n_heads, n_layers, n_rbf, d_edge, cutoff=7.0, act_fn=nn.SiLU(), layer_norm = 'pre', residual = True, sparse_k=3, svd_k=128

    model = Transformer(d_hidden, d_ffn, n_heads, n_layers, n_rbf, d_edge=d_edge,  use_edge_feat=True)
    model.to(device)
    model.eval()
    
    block_id = torch.tensor([0,0,1,1,1,1,2,2,2,3,4,4,5,6,6,6,6,7,7], dtype=torch.long).to(device)
    batch_id = torch.tensor([0,0,0,0,0,1,1,1], dtype=torch.long).to(device)
    src_dst = torch.tensor([[0,1], [2,3], [1,3], [2,4], [3, 0], [3, 3], [5,7], [7,6], [5,6], [6,7]], dtype=torch.long).to(device)
    src_dst = src_dst.T
    edge_attr = torch.randn(len(src_dst[0]), d_edge).to(device)
    n_unit = block_id.shape[0]

    H = torch.randn(n_unit, d_hidden, device=device)
    Z = torch.randn(n_unit, 3, device=device)

    H1, V1 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q2, t2 = U.mm(V), torch.randn(3, device=device)

    unit_batch_id = batch_id[block_id]
    Z[unit_batch_id == 0] = torch.matmul(Z[unit_batch_id == 0], Q1) + t1
    Z[unit_batch_id == 1] = torch.matmul(Z[unit_batch_id == 1], Q2) + t2
    # Z = torch.matmul(Z, Q) + t

    H2, V2 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    print(f'invariant feature: {torch.abs(H1 - H2).sum()}')
    V1[unit_batch_id == 0] = torch.einsum('nih, ij -> njh', V1[unit_batch_id == 0], Q1)
    V1[unit_batch_id == 1] = torch.einsum('nih, ij -> njh', V1[unit_batch_id == 1], Q2)
    print(f'equivariant feature: {torch.abs(V1 - V2).sum()}')