#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from data.bioparse import VOCAB

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import SinusoidalPositionEmbedding
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean

from .diffusion.dpm_full import FullDPM,FullDPMRAG,FullDPMCFG,FullDPMRAGfull,FullDPMRAGmask
from ..IterVAE.model_edge import CondIterAutoEncoderEdge
from ..modules.nn import GINEConv, MLP


@R.register('LDMMolDesignClean')
class LDMMolDesignClean(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            max_unmask_ratio=0.0,
            is_aa_corrupt_ratio=0.1,
            diffusion_opt={}
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        self.autoencoder: CondIterAutoEncoderEdge = torch.load(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        latent_size = self.autoencoder.latent_size

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size, # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.max_unmask_ratio = max_unmask_ratio
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
        ):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
            ) # [Nblock, d_latent], [Nblock, 3]

        position_embedding = self.position_encoding(position_ids)

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction (sample topo ratio [0, 1] in training)
        unmask_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] * self.max_unmask_ratio
        unmask = torch.rand_like(generate_mask, dtype=torch.float) < unmask_ratio
        topo_generate_mask = generate_mask & (~unmask) # originally 1 in generate mask and 0 in unmask
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)

        # is aa embedding (sample 50% for generation part)
        corrupt_mask = topo_generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths
        )

        # loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    # def latent_geometry_guidance(self, X, generate_mask, batch_ids, tolerance=3, **kwargs):
    #     assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
    #            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
    #     return dist_energy(
    #         X, generate_mask, batch_ids,
    #         self.consec_dist_mean, self.consec_dist_std,
    #         tolerance=tolerance, **kwargs
    #     )

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        # aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            sample_opt={
                'pbar': False,
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            topo_generate_mask=None,
            return_tensor=False,
        ):

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        if topo_generate_mask is None: topo_generate_mask = generate_mask

        # ensure there is no data leakage
        S[topo_generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[topo_generate_mask[block_ids]] = 0
        ctx_atom_mask = ~topo_generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # encoding context
        self.autoencoder.eval()
        Zh, Zx, _, signed_Zx_log_var, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
        ) # [Nblock, d_latent], [Nblock, 3]

        # if 'energy_func' in sample_opt:
        #     if sample_opt['energy_func'] is None:
        #         pass
        #     elif sample_opt['energy_func'] == 'default':
        #         sample_opt['energy_func'] = self.latent_geometry_guidance
        #     # otherwise this should be a function
        

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        traj = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            **sample_opt
        )
        X_0, H_0 = traj[0]
        X_0 = torch.where(generate_mask[:, None].expand_as(X_0), X_0, Zx)
        H_0 = torch.where(generate_mask[:, None].expand_as(H_0), H_0, Zh)

        # unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # autodecoder decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=topo_generate_mask
        )

@R.register('LDMMolDesignRAG')
class LDMMolDesignRAG(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            max_unmask_ratio=0.0,
            is_aa_corrupt_ratio=0.1,
            diffusion_opt={}
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        self.autoencoder: CondIterAutoEncoderEdge = torch.load(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        latent_size = self.autoencoder.latent_size

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size, # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPMRAG(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.max_unmask_ratio = max_unmask_ratio
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            prompt_feature, #[bs*topn,hidden_dim] 
        ):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''
        
        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
            ) # [Nblock, d_latent], [Nblock, 3]

        position_embedding = self.position_encoding(position_ids)

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction (sample topo ratio [0, 1] in training)
        unmask_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] * self.max_unmask_ratio
        unmask = torch.rand_like(generate_mask, dtype=torch.float) < unmask_ratio
        topo_generate_mask = generate_mask & (~unmask) # originally 1 in generate mask and 0 in unmask
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)

        # is aa embedding (sample 50% for generation part)
        corrupt_mask = topo_generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature
        )

        # loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    # def latent_geometry_guidance(self, X, generate_mask, batch_ids, tolerance=3, **kwargs):
    #     assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
    #            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
    #     return dist_energy(
    #         X, generate_mask, batch_ids,
    #         self.consec_dist_mean, self.consec_dist_std,
    #         tolerance=tolerance, **kwargs
    #     )

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        # aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            sample_opt={
                'pbar': False,
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            topo_generate_mask=None,
            return_tensor=False,
            prompt_feature=None,
        ):

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        if topo_generate_mask is None: topo_generate_mask = generate_mask

        # ensure there is no data leakage
        S[topo_generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[topo_generate_mask[block_ids]] = 0
        ctx_atom_mask = ~topo_generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # encoding context
        self.autoencoder.eval()
        Zh, Zx, _, signed_Zx_log_var, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
        ) # [Nblock, d_latent], [Nblock, 3]

        # if 'energy_func' in sample_opt:
        #     if sample_opt['energy_func'] is None:
        #         pass
        #     elif sample_opt['energy_func'] == 'default':
        #         sample_opt['energy_func'] = self.latent_geometry_guidance
        #     # otherwise this should be a function
        

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        traj = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature,
            **sample_opt
        )
        X_0, H_0 = traj[0]
        X_0 = torch.where(generate_mask[:, None].expand_as(X_0), X_0, Zx)
        H_0 = torch.where(generate_mask[:, None].expand_as(H_0), H_0, Zh)

        # unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # autodecoder decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=topo_generate_mask
        )
        
@R.register('LDMMolDesignRAGfull')
class LDMMolDesignRAGfull(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            max_unmask_ratio=0.0,
            is_aa_corrupt_ratio=0.1,
            diffusion_opt={}
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        self.autoencoder: CondIterAutoEncoderEdge = torch.load(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        latent_size = self.autoencoder.latent_size

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size, # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPMRAGfull(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.max_unmask_ratio = max_unmask_ratio
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            prompt_feature, #[bs*topn,hidden_dim]
            prompt_mask, 
        ):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''
        
        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
            ) # [Nblock, d_latent], [Nblock, 3]

        position_embedding = self.position_encoding(position_ids)

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction (sample topo ratio [0, 1] in training)
        unmask_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] * self.max_unmask_ratio
        unmask = torch.rand_like(generate_mask, dtype=torch.float) < unmask_ratio
        topo_generate_mask = generate_mask & (~unmask) # originally 1 in generate mask and 0 in unmask
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)

        # is aa embedding (sample 50% for generation part)
        corrupt_mask = topo_generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature,
            prompt_mask=prompt_mask
        )

        # loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    # def latent_geometry_guidance(self, X, generate_mask, batch_ids, tolerance=3, **kwargs):
    #     assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
    #            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
    #     return dist_energy(
    #         X, generate_mask, batch_ids,
    #         self.consec_dist_mean, self.consec_dist_std,
    #         tolerance=tolerance, **kwargs
    #     )

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        # aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            sample_opt={
                'pbar': False,
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            topo_generate_mask=None,
            return_tensor=False,
            prompt_feature=None,
            prompt_mask=None,
        ):

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        if topo_generate_mask is None: topo_generate_mask = generate_mask

        # ensure there is no data leakage
        S[topo_generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[topo_generate_mask[block_ids]] = 0
        ctx_atom_mask = ~topo_generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # encoding context
        self.autoencoder.eval()
        Zh, Zx, _, signed_Zx_log_var, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
        ) # [Nblock, d_latent], [Nblock, 3]

        # if 'energy_func' in sample_opt:
        #     if sample_opt['energy_func'] is None:
        #         pass
        #     elif sample_opt['energy_func'] == 'default':
        #         sample_opt['energy_func'] = self.latent_geometry_guidance
        #     # otherwise this should be a function
        

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        traj = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature,
            prompt_mask=prompt_mask,
            **sample_opt
        )
        X_0, H_0 = traj[0]
        X_0 = torch.where(generate_mask[:, None].expand_as(X_0), X_0, Zx)
        H_0 = torch.where(generate_mask[:, None].expand_as(H_0), H_0, Zh)

        # unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # autodecoder decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=topo_generate_mask
        )

@R.register('LDMMolDesignCFG')
class LDMMolDesignCFG(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            max_unmask_ratio=0.0,
            is_aa_corrupt_ratio=0.1,
            drop=0,
            diffusion_opt={}
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic
        self.drop=drop
        self.autoencoder: CondIterAutoEncoderEdge = torch.load(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        latent_size = self.autoencoder.latent_size

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size, # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPMRAG(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.max_unmask_ratio = max_unmask_ratio
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            prompt_feature, #[bs*topn,hidden_dim] 
        ):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''
        bs=lengths.shape[0]
        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
            ) # [Nblock, d_latent], [Nblock, 3]

        position_embedding = self.position_encoding(position_ids)

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction (sample topo ratio [0, 1] in training)
        unmask_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] * self.max_unmask_ratio
        unmask = torch.rand_like(generate_mask, dtype=torch.float) < unmask_ratio
        topo_generate_mask = generate_mask & (~unmask) # originally 1 in generate mask and 0 in unmask
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)

        # is aa embedding (sample 50% for generation part)
        corrupt_mask = topo_generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        if self.drop > 0 :


            batch_size =bs
            


            total_entries = prompt_feature.shape[0]
            topn = total_entries // batch_size
            


            sample_drop_mask = torch.rand(batch_size, device=prompt_feature.device) < self.drop
            


            entry_drop_mask = sample_drop_mask.repeat_interleave(topn).view(-1, 1)
            


            prompt_feature = prompt_feature * (~entry_drop_mask)
        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature
        )

        # loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    # def latent_geometry_guidance(self, X, generate_mask, batch_ids, tolerance=3, **kwargs):
    #     assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
    #            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
    #     return dist_energy(
    #         X, generate_mask, batch_ids,
    #         self.consec_dist_mean, self.consec_dist_std,
    #         tolerance=tolerance, **kwargs
    #     )

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        # aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            sample_opt={
                'pbar': False,
                'guidance_scale': 7.5,
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            topo_generate_mask=None,
            return_tensor=False,
            prompt_feature=None,
        ):
        bs=lengths.shape[0]
        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        if topo_generate_mask is None: topo_generate_mask = generate_mask

        # ensure there is no data leakage
        S[topo_generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[topo_generate_mask[block_ids]] = 0
        ctx_atom_mask = ~topo_generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # encoding context
        self.autoencoder.eval()
        Zh, Zx, _, signed_Zx_log_var, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
        ) # [Nblock, d_latent], [Nblock, 3]

        # if 'energy_func' in sample_opt:
        #     if sample_opt['energy_func'] is None:
        #         pass
        #     elif sample_opt['energy_func'] == 'default':
        #         sample_opt['energy_func'] = self.latent_geometry_guidance
        #     # otherwise this should be a function
        

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        


        traj_cond = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature,
            **sample_opt
        )        


        batch_size =bs


        total_entries = prompt_feature.shape[0]
        topn = total_entries // batch_size


        sample_drop_mask = torch.ones(batch_size, dtype=torch.bool, device=prompt_feature.device)
        


        entry_drop_mask = sample_drop_mask.repeat_interleave(topn).view(-1, 1)
        


        prompt_feature = prompt_feature * (~entry_drop_mask)
        
        traj_uncond = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature,
            **sample_opt
        )
        X_cond, H_cond = traj_cond[0]
        X_uncond, H_uncond = traj_uncond[0]



        guidance_scale = sample_opt.pop('guidance_scale', 7.5)
        H_cfg = H_uncond + guidance_scale * (H_cond - H_uncond)
        X_cfg = X_uncond + guidance_scale * (X_cond - X_uncond)



        H_0 = torch.where(generate_mask[:, None].expand_as(H_cfg), H_cfg, Zh)
        X_0 = torch.where(generate_mask[:, None].expand_as(X_cfg), X_cfg, Zx)



        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # autodecoder decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=topo_generate_mask
        )
        
@R.register('LDMMolDesignRAGmask')
class LDMMolDesignRAGmask(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            max_unmask_ratio=0.0,
            is_aa_corrupt_ratio=0.1,
            diffusion_opt={}
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        self.autoencoder: CondIterAutoEncoderEdge = torch.load(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        latent_size = self.autoencoder.latent_size

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size, # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPMRAGmask(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.max_unmask_ratio = max_unmask_ratio
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            prompt_feature, #[bs*topn,hidden_dim]
            
        ):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''
        
        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
            ) # [Nblock, d_latent], [Nblock, 3]

        position_embedding = self.position_encoding(position_ids)

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction (sample topo ratio [0, 1] in training)
        unmask_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] * self.max_unmask_ratio
        unmask = torch.rand_like(generate_mask, dtype=torch.float) < unmask_ratio
        topo_generate_mask = generate_mask & (~unmask) # originally 1 in generate mask and 0 in unmask
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)

        # is aa embedding (sample 50% for generation part)
        corrupt_mask = topo_generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature,
        )

        # loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    # def latent_geometry_guidance(self, X, generate_mask, batch_ids, tolerance=3, **kwargs):
    #     assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
    #            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
    #     return dist_energy(
    #         X, generate_mask, batch_ids,
    #         self.consec_dist_mean, self.consec_dist_std,
    #         tolerance=tolerance, **kwargs
    #     )

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        # aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            sample_opt={
                'pbar': False,
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            topo_generate_mask=None,
            return_tensor=False,
            prompt_feature=None,

        ):

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        if topo_generate_mask is None: topo_generate_mask = generate_mask

        # ensure there is no data leakage
        S[topo_generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[topo_generate_mask[block_ids]] = 0
        ctx_atom_mask = ~topo_generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # encoding context
        self.autoencoder.eval()
        Zh, Zx, _, signed_Zx_log_var, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
        ) # [Nblock, d_latent], [Nblock, 3]

        # if 'energy_func' in sample_opt:
        #     if sample_opt['energy_func'] is None:
        #         pass
        #     elif sample_opt['energy_func'] == 'default':
        #         sample_opt['energy_func'] = self.latent_geometry_guidance
        #     # otherwise this should be a function
        

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), topo_generate_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        traj = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            prompt_feature=prompt_feature,
            **sample_opt
        )
        X_0, H_0 = traj[0]
        X_0 = torch.where(generate_mask[:, None].expand_as(X_0), X_0, Zx)
        H_0 = torch.where(generate_mask[:, None].expand_as(H_0), H_0, Zh)

        # unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # autodecoder decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=topo_generate_mask)