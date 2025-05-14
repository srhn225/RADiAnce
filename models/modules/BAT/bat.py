#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_sum


def _stable_norm(input, *args, **kwargs):
    '''
        For L2: norm = sqrt(\sum x^2) = (\sum x^2)^{1/2}
        the gradient will have zero in divider if \sum x^2 = 0
        it is not ok to direct add eps to all x, since x might be a small but negative value
        this function deals with this problem
    '''
    input = input.clone()
    with torch.no_grad():
        sign = torch.sign(input)
        input = torch.abs(input)
        input.clamp_(min=1e-10)
        input = sign * input
    return torch.norm(input, *args, **kwargs)


def _concat_scaler_vector(H, V):
    '''
        Args:
            H: [..., Dh]
            V: [..., Dv, 3]
        Returns:
            out: [..., Dh + Dv*3]
    '''
    prefix_shape = V.shape[:-2]
    return torch.cat([
        H, V.reshape(*prefix_shape, -1)
    ], dim=-1)


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 10.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            dist: [N]
        Returns:
            out: [N, num_gaussians]
        '''
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class GVPSO3Layer(nn.Module):
    '''
        SO(3)-equivariant variant of GVP layer
    '''
    def __init__(self, d_scaler, d_vector, d_hidden, d_scaler_out=None, d_vector_out=None,
                 act_fn=nn.SiLU(), vector_act='none', bias=True):

        super(GVPSO3Layer, self).__init__()

        self.d_hidden = d_hidden
        self.act_fn = act_fn
        self.d_scaler_out = d_scaler if d_scaler_out is None else d_scaler_out
        self.d_vector_out = d_vector if d_vector_out is None else d_vector_out

        self.linear_v = nn.Linear(2 * d_vector, d_hidden + self.d_vector_out, bias=False) # bias must be false
        self.ffn_mlp = nn.Sequential(
            nn.Linear(d_scaler + d_hidden, d_hidden, bias=bias),
            act_fn,
            nn.Linear(d_hidden, self.d_scaler_out + self.d_vector_out, bias=bias)
        )

        self.vector_act = vector_act
        if self.vector_act == 'layernorm':
            self.vector_layernorm = nn.LayerNorm(self.d_vector_out)

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
        '''
        Args:
            H: [..., d_scaler],
            V: [..., d_vector, 3]
        Returns:
            H_out: [..., d_scaler_out]
            V_out: [..., d_vector_out, 3]
        '''
        # transpose for better handling
        V = V.transpose(-1, -2) # [N, 3, d_vector]

        # cross product between (v0, v1, ..., vd) and (v1, v2, ..., vd, v0)
        V_cross = torch.cross(V, torch.cat([V[..., 1:], V[..., 0:1]], dim=-1), dim=-2) # [N, 3, d_vector]
        V = torch.cat([V, V_cross], dim=-1) # [N, 3, 2 * d_vector]

        V_proj = self.linear_v(V) # [N, d_hidden + d_vector_out]
        V1, V2 = V_proj[...,:self.d_hidden], V_proj[...,self.d_hidden:] # [N, 3, d_hidden], [N, 3, d_vector_out]
        scaler = torch.cat([H, _stable_norm(V1, dim=-2)], dim=-1) # [N, d_scaler + d_hidden]
        scaler_out = self.ffn_mlp(scaler) # [N, d_scaler_out + d_vector_out]
        H_out, V_update = scaler_out[...,:self.d_scaler_out], scaler_out[...,self.d_scaler_out:] # [N, d_scaler_out], [N, d_vector_out]
        V_out = self.vector_act_func(V_update).unsqueeze(-2) * V2   # [N, 3, d_vector_out]

        # transpose back
        V_out = V_out.transpose(-1, -2)

        return H_out, V_out
    

class BlockAdaptiveTransformerLayer(nn.Module):
    '''
        Block-Adaptive Transformer Layer, with each node represented by a set of mini-nodes
        with embeddings, vectors, and coordinates
    '''
    
    def __init__(self, d_scaler, d_vector, d_hidden, n_head, d_rbf=32, cutoff=10.0, d_edge=0, vector_layernorm=True, update_coord=False) -> None:
        super().__init__()
        self.d_scaler = d_scaler
        self.d_vector = d_vector
        self.n_head = n_head
        self.d_rbf = d_rbf
        self.cutoff = cutoff
        self.d_edge = d_edge
        self.update_coord = update_coord

        assert d_scaler % n_head == 0
        assert d_vector % n_head == 0
        assert d_rbf % n_head == 0

        self.gW_qk = GVPSO3Layer(
            d_scaler // n_head, d_vector // n_head, d_hidden, 2 * d_scaler // n_head, 2 * d_vector // n_head,
            vector_act='layernorm' if vector_layernorm else 'none')
        self.rbf = GaussianSmearing(stop=cutoff, num_gaussians=d_rbf)
        self.W_R = nn.Linear(self.d_rbf // n_head, 1, bias=False)
        self.att_act_fn = nn.SiLU()

        self.gW_v = GVPSO3Layer(
            (d_scaler + d_rbf) // n_head, d_vector // n_head + 1, d_hidden, d_scaler // n_head, d_vector // n_head,
            vector_act='layernorm' if vector_layernorm else 'none')

        if d_edge > 0:
            self.W_E = nn.Sequential(
                nn.Linear(d_edge, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, n_head)
            )

        if update_coord:
            self.W_coord = nn.Linear(self.d_vector, 1, bias=False)


    def forward(self, H, V, X, mask, edge_index, edge_attr=None):
        '''
        Args:
            H: [N, L, Dh], L is the length of the biggest block, H is n_head
            V: [N, L, Dv, 3]
            X: [N, L, 3]
            mask: [N, L],
            edge_index: [2, E]
            edge_attr: [E, d_edge]
        Returns:
            H_out: [N, L, Dh]
            V_out: [N, L, Dv, 3]
            X_out: [N, L, 3]
        '''
        # Step0: multihead
        N, L = H.shape[:2]
        H = H.view(N, L, self.n_head, self.d_scaler // self.n_head)
        H = H.transpose(1, 2) # [N, H, L, Dh/H]
        V = V.view(N, L, self.n_head, self.d_vector // self.n_head, 3)
        V = V.transpose(1, 2) # [N, H, L, Dv/H, 3]

        # Step1: prepare Q, K, R (geometric distance) for attention
        H_qk, V_qk = self.gW_qk(H, V)
        H_q, H_k = H_qk.split(self.d_scaler // self.n_head, dim=-1) # [N, H, L, Dh] * 2
        V_q, V_k = V_qk.split(self.d_vector // self.n_head, dim=-2) # [N, H, L, Dv, 3] * 2
        
        row, col = edge_index
        X_ij = X[row].unsqueeze(-2) - X[col].unsqueeze(-3) # [E, L, L, 3] (N*ni*nj)
        D_ij = _stable_norm(X_ij, dim=-1) # [E, L, L] (N*ni*nj)
        E, L, _ = D_ij.shape
        R_ij = self.rbf(D_ij.flatten()).view(E, L, L, self.d_rbf) # [E, L, L, d_rbf] (N*ni*nj)
        R_ij = R_ij.view(E, L, L, self.n_head, self.d_rbf // self.n_head)
        R_ij = R_ij.transpose(2, 3).transpose(1, 2) # [E, H, L, L, d_rbf/H]
        alpha_mask = mask[row].unsqueeze(-1) & mask[col].unsqueeze(-2) # [E, L, L]
        alpha_mask = alpha_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1) # [E, H, L, L]
        X_ij = X_ij.unsqueeze(1).repeat(1, self.n_head, 1, 1, 1) # [E, H, L, L, 3]

        # Step2: attention
        if edge_attr is None:
            edge_attr = 1.0
            assert self.d_edge == 0
        else:
            edge_attr = self.W_E(edge_attr) # [E, H]
            edge_attr = edge_attr.unsqueeze(-1).unsqueeze(-1) # [E, H, 1, 1]
        alpha_ij = self.att_act_fn(
            torch.matmul(
                _concat_scaler_vector(H_q, V_q)[row], # [E, H, L, (Dh + Dv*3)/H], L=>ni
                _concat_scaler_vector(H_k, V_k).transpose(-1, -2)[col] # [E, H, (Dh + Dv*3)/H, L], L=>nj
            ) * self.W_R(R_ij).squeeze(-1) * edge_attr # [E, H, L, L]
        ) # [E, H, L, L], E*ni*nj
        alpha_ij = torch.where(alpha_mask, alpha_ij, torch.zeros_like(alpha_ij))
        alpha_ij = alpha_ij.unsqueeze(-1) # [E, H, L, L, 1], E*ni*nj*1

        # Step3: message
        H_v, V_v = self.gW_v(
            torch.cat([H[col], (alpha_ij * R_ij).sum(dim=-3)], dim=-1),
            torch.cat([V[col], (alpha_ij * X_ij).sum(dim=-3).unsqueeze(-2)], dim=-2)
        ) # [E, H, L, Dh/H], [E, H, L, Dv/H, 3] (L=>nj)
        alpha_ij = alpha_ij.squeeze(-1) # [E, H, L, L]
        H_v = torch.matmul(alpha_ij, H_v) # [E, H, L, Dh/H] (L=>ni)
        V_v = torch.matmul(alpha_ij, V_v.flatten(start_dim=-2)).view(*V_v.shape[:-2], -1, 3) # [E, H, L, Dv/H, 3] (L=>ni)

        # Step4: aggregate
        dH = scatter_sum(H_v, row, dim=0, dim_size=H.shape[0])  # [N, H, L, Dh/H]
        dV = scatter_sum(V_v, row, dim=0, dim_size=V.shape[0])  # [N, H, L, Dv/H, 3]
        
        # Step5: update
        H_out, V_out = H + dH, V + dV # [N, H, L, Dh/H], [N, H, L, Dv/H, 3]
        H_out = H_out.transpose(1, 2).view(N, L, self.d_scaler)     # [N, H, L, Dh]
        V_out = V_out.transpose(1, 2).view(N, L, self.d_vector, 3)  # [N, H, L, Dv, 3]

        if self.update_coord:
            X_out = X + self.W_coord(V_out.transpose(-1, -2)).squeeze(-1)
        else:
            X_out = X
        
        return H_out, V_out, X_out


class BlockAdaptiveTransformer(nn.Module):
    def __init__(self, d_scaler, d_vector, d_hidden, n_head, n_layers, d_rbf=32, cutoff=10.0, d_edge=0, vector_layernorm=True, update_coord=False) -> None:
        super().__init__()
        self.n_layers = n_layers
        layers, lns = [], [] # layers and layernorms
        for i in range(n_layers):
            layers.append(BlockAdaptiveTransformerLayer(
                d_scaler, d_vector, d_hidden, n_head, d_rbf, cutoff, d_edge, vector_layernorm, update_coord
            ))
            lns.append(nn.LayerNorm(d_scaler))
        self.layers = nn.ModuleList(layers)
        self.lns = nn.ModuleList(lns)
        
    
    def forward(self, H, V, X, mask, edge_index, edge_attr=None):
        for i in range(self.n_layers):
            H, V, X = self.layers[i](H, V, X, mask, edge_index, edge_attr)
            H = self.lns[i](H)
        return H, V, X


if __name__ == '__main__':


    d_scaler = 10
    d_vector = 10
    d_hidden = 4 * d_scaler
    d_scaler_out = 5
    d_vector_out = 5
    n_nodes = 4

    device = torch.device('cuda:0')
    
    def random_mat(reflection=False):
        U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
        if (torch.linalg.det(U) * torch.linalg.det(V) < 0 and not reflection) or \
           (torch.linalg.det(U) * torch.linalg.det(V) > 0 and reflection):
            U[:, -1] = -U[:, -1]
        return U.mm(V)

    print('Testing GVP SO(3) layer')

    gvp = GVPSO3Layer(d_scaler, d_vector, d_hidden, d_scaler_out, d_vector_out)
    gvp.to(device)
    gvp.eval()

    H, V = torch.randn((n_nodes, d_scaler)), torch.randn(n_nodes, d_vector, 3)
    H, V = H.to(device), V.to(device)

    H1, V1 = gvp(H, V)

    # random rotaion matrix
    Q = random_mat()
    # random reflection matrix
    R = random_mat(reflection=True)

    # test under SO(3) transformation
    H2, V2 = gvp(H, torch.matmul(V, Q[None, ...]))
    print(f'Invariant feature under SO(3): {torch.abs(H1 - H2).sum()}')
    print(f'equivariant feature under SO(3): {torch.abs(torch.matmul(V1, Q[None, ...]) - V2).sum()}')

    # test under reflection
    H3, V3 = gvp(H, torch.matmul(V, R))
    print(f'Scalers under reflection: {torch.abs(H1 - H3).sum()}')
    print(f'Vectors under reflection: {torch.abs(torch.matmul(V1, R[None, ...]) - V3).sum()}')

    print()
    print('Testing Block-Adaptive Transformer Layer')
    l_max = 20
    d_rbf = 32
    n_head = 2
    n_layers = 6

    # bat = BlockAdaptiveTransformerLayer(d_scaler, d_vector, d_hidden, d_rbf, update_coord=True)
    bat = BlockAdaptiveTransformer(d_scaler, d_vector, d_hidden, n_head, n_layers, d_rbf, vector_layernorm=False, update_coord=True)
    bat.to(device)
    bat.eval()

    H, V = torch.randn((n_nodes, l_max, d_scaler)), torch.randn(n_nodes, l_max, d_vector, 3)
    X = torch.randn((n_nodes, l_max, 3))
    H, V, X = H.to(device), V.to(device), X.to(device)
    mask = H.sum(-1) > 0 # [N, L]
    print('block size:', mask.sum(-1)) # [N]
    edge_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edge_index.append((i, j))
    edge_index = torch.tensor(edge_index).transpose(0, 1) # [2, N * N]
    edge_index = edge_index.to(device)

    H1, V1, X1 = bat(H, V, X, mask, edge_index)

    # translation
    t = torch.randn(3, device=device)

    # test under SE(3) transformation
    H2, V2, X2 = bat(H, torch.matmul(V, Q[None, ...]), torch.matmul(X, Q[None, ...]) + t, mask, edge_index)
    print(f'Invariant feature under SE(3): {torch.abs(H1 - H2).sum()}')
    print(f'equivariant feature under SO(3): {torch.abs(torch.matmul(V1, Q[None, ...]) - V2).sum()}')
    print(f'equivariant feature under SE(3): {torch.abs(torch.matmul(X1, Q[None, ...]) + t - X2).sum()}')

    # test under reflection
    H3, V3, X3 = bat(H, torch.matmul(V, R[None, ...]), torch.matmul(X, R[None, ...]) + t, mask, edge_index)
    print(f'Scalers under reflection: {torch.abs(H1 - H3).sum()}')
    print(f'Vectors under reflection: {torch.abs(torch.matmul(V1, R[None, ...]) - V3).sum()}')
    print(f'Coordinates under reflection: {torch.abs(torch.matmul(X1, R[None, ...]) - X3).sum()}')

    # test change values of padding part
    H[~mask], V[~mask], X[~mask] = 0.5, 0.6, 0.7
    H4, V4, X4 = bat(H, V, X, mask, edge_index)
    print(f'Scalers when paddings are changed: {torch.abs(H1[mask] - H4[mask]).sum()}')
    print(f'Vectors when paddings are changed: {torch.abs(V1[mask] - V4[mask]).sum()}')
    print(f'Coordinates when paddings are changed: {torch.abs(X1[mask] - X4[mask]).sum()}')

    # test permutation invariance
    H_clone, V_clone, X_clone = H.clone(), V.clone(), X.clone()
    H_clone[0, 0], H_clone[0, 1] = H[0, 1], H[0, 0]
    V_clone[0, 0], V_clone[0, 1] = V[0, 1], V[0, 0]
    X_clone[0, 0], X_clone[0, 1] = X[0, 1], X[0, 0]
    mask_clone = mask.clone()
    mask_clone[0, 0], mask_clone[0, 1] = mask[0, 1], mask[0, 0]
    H1_clone, V1_clone, X1_clone = H1.clone(), V1.clone(), X1.clone()
    H1_clone[0, 0], H1_clone[0, 1] = H1[0, 1], H1[0, 0]
    V1_clone[0, 0], V1_clone[0, 1] = V1[0, 1], V1[0, 0]
    X1_clone[0, 0], X1_clone[0, 1] = X1[0, 1], X1[0, 0]
    H5, V5, X5 = bat(H_clone, V_clone, X_clone, mask_clone, edge_index)
    print(f'Scalers on permutation: {torch.abs(H1_clone[mask] - H5[mask]).sum()}')
    print(f'Vectors on permutation: {torch.abs(V1_clone[mask] - V5[mask]).sum()}')
    print(f'Coordinates on permutation: {torch.abs(X1_clone[mask] - X5[mask]).sum()}')