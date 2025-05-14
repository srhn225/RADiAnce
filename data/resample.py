#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json

import numpy as np
import torch


class ClusterResampler:
    def __init__(self, cluster_path: str) -> None:
        idx2prob = []
        with open(cluster_path, 'r') as fin:
            for line in fin:
                cluster_n_member = int(line.strip().split('\t')[-1])
                idx2prob.append(1 / cluster_n_member)
        total = sum(idx2prob)
        idx2prob = [p / total for p in idx2prob]
        self.idx2prob = np.array(idx2prob)

    def __call__(self, n_sample:int, replace: bool=False):
        idxs = np.random.choice(len(self.idx2prob), size=n_sample, replace=replace, p=self.idx2prob)
        return idxs


class SizeResampler:
    def __init__(
            self,
            mode = 'block_by_atom', # atom, block, block_by_atom
            size_min = None,        # atom size, block size, atom size
            size_max = None,
        ) -> None:

        # load frequency record
        freq_json = os.path.join(os.path.dirname(__file__), 'size_freq.json')
        with open(freq_json, 'r') as fin:
            self.size_freq = json.load(fin)
        
        self.mode = mode
        self.size_min = size_min
        self.size_max = size_max

        first_mode = 'atom' if mode == 'block_by_atom' else mode
        self.v, self.p = self._get_dist_by_range(self.size_freq[first_mode], self.size_min, self.size_max)

    def _get_dist_by_range(self, size_freq_dict, size_min = None, size_max = None):
        sizes, probs = [], []
        for key in size_freq_dict:
            key = int(key)
            if size_min is not None and key < size_min: continue
            if size_max is not None and key > size_max: continue
            sizes.append(key)
            probs.append(size_freq_dict[str(key)])
        # normalize
        sizes, probs = np.array(sizes), np.array(probs)
        probs = probs / probs.sum()
        return sizes, probs

    def __call__(self, n_sample: int):
        if self.mode == 'atom' or self.mode == 'block':
            return np.random.choice(self.v, size=n_sample, p=self.p).tolist()
        elif self.mode == 'block_by_atom':
            atom_size = np.random.choice(self.v, size=n_sample, p=self.p)
            final = []
            for a in atom_size:
                v, p = self._get_dist_by_range(self.size_freq[self.mode][str(a)])
                final.append(np.random.choice(v, size=1, p=p).tolist()[0])
            return final
        else:
            raise ValueError(f'Mode {self.mode} not recognized')


def _get_bin_idx(space_size, config):
    bounds = config['bounds']
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)


def sample_atom_num(space_size, config):
    bin_idx = _get_bin_idx(space_size, config)
    num_atom_list, prob_list = config['bins'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)


class SizeSamplerByPocketSpace:
    def __init__(self, size_min=None, size_max=None):
        super().__init__()
        self.size_min = size_min
        self.size_max = size_max
        
        # load frequency record
        current_dir = os.path.dirname(os.path.abspath(__file__))
        freq_json = os.path.join(current_dir, 'size_freq.json')
        with open(freq_json, 'r') as fin:
            self.size_freq = json.load(fin)['block_by_atom']
        self.config_atom_num = np.load(os.path.join(current_dir, '_atom_num_dist.npy'), allow_pickle=True).item()
    
    def __call__(self, n_sample: int, pocket_pos: list):
        pocket_size = self.get_space_size(torch.tensor(pocket_pos, dtype=torch.float))

        final_sizes = []

        for _ in range(n_sample):
            num_atoms_lig = sample_atom_num(pocket_size.item(), self.config_atom_num).astype(int)
            while str(num_atoms_lig) not in self.size_freq:
                num_atoms_lig = sample_atom_num(pocket_size.item(), self.config_atom_num).astype(int)

            # Turn num atoms to num blocks
            num_block_dist = self.size_freq[str(num_atoms_lig)]
            sizes, probs = [], []
            for key in num_block_dist:
                sizes.append(int(key))
                probs.append(num_block_dist[key])
            probs = np.array(probs)
            probs = probs / probs.sum()
            num_blocks = np.random.choice(sizes, size=1, p=probs)[0]
            if self.size_min is not None: num_blocks = max(self.size_min, num_blocks)
            final_sizes.append(num_blocks)

        return final_sizes

    def get_space_size(self, pos):
        aa_dist = torch.pdist(pos)
        aa_dist = torch.sort(aa_dist, descending=True)[0]
        return torch.median(aa_dist[:10])

    def get_expected_atom_num(self, pocket_pos: list):
        pocket_size = self.get_space_size(torch.tensor(pocket_pos, dtype=torch.float))
        bin_idx = _get_bin_idx(pocket_size, self.config_atom_num)
        num_atom_list, prob_list = self.config_atom_num['bins'][bin_idx]
        return (np.array(num_atom_list) * np.array(prob_list)).sum()


if __name__ == '__main__':
    sampler = SizeResampler(size_min=20)
    print(sampler(10))