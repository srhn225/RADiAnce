#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from data.bioparse.hierarchy import remove_mols, add_dummy_mol
from data.bioparse.utils import recur_index
from utils import register as R

from .peptide import PeptideDataset
from .resample import SizeResampler, SizeSamplerByPocketSpace
from .base import transform_data
from .feature_dataset import FeatureDataset
import torch
import json

@R.register('MoleculeDataset')
class MoleculeDataset(PeptideDataset):

    def __init__(self, mmap_dir, specify_data = None, specify_index = None, cluster = None, length_type = 'atom', sample_size = False):
        super().__init__(mmap_dir, specify_data, specify_index, cluster, length_type)
        if sample_size:
            # self.size_sampler = SizeResampler(**sample_size_opt)
            self.size_sampler = SizeSamplerByPocketSpace()
        else:
            self.size_sampler = None

    def __getitem__(self, idx):
        if self.size_sampler is None:
            data = super().__getitem__(idx)
            # set position ids of small molecules to zero
            data['position_ids'][data['generate_mask']] = 0
            return data

        # change the size of the small molecule
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)

        lig_chain = summary.ligand_chain_ids[0]
        
        props = self._properties[idx]
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        
        pocket_pos = []
        for block_id in pocket_block_ids:
            for atom in recur_index(cplx, block_id):
                pocket_pos.append(atom.coordinate)
        size = self.size_sampler(1, pocket_pos)[0]
        cplx = remove_mols(cplx, summary.ligand_chain_ids) # remove ground truth molecule
        cplx = add_dummy_mol(cplx, size, summary.ligand_chain_ids[0])
        lig_block_ids = [(lig_chain, block.id) for block in cplx[lig_chain]]
        
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in lig_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in lig_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in lig_block_ids]
        
        data = transform_data(cplx, pocket_block_ids + lig_block_ids)
        data['generate_mask'] = torch.tensor(generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(center_mask, dtype=torch.bool)
        # set position ids of small molecules to zero
        data['position_ids'][data['generate_mask']] = 0
        
        return data
    
    def get_expected_atom_num(self, idx):
        # change the size of the small molecule
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)

        props = self._properties[idx]
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        
        pocket_pos = []
        for block_id in pocket_block_ids:
            for atom in recur_index(cplx, block_id):
                pocket_pos.append(atom.coordinate)
        expect_atom_num = self.size_sampler.get_expected_atom_num(pocket_pos)
        return expect_atom_num

@R.register('MoleculeDatasetPrompt')
class MoleculeDatasetPrompt(PeptideDataset):

    def __init__(self, mmap_dir, specify_data = None, specify_index = None, cluster = None, length_type = 'atom', sample_size = False):
        super().__init__(mmap_dir, specify_data, specify_index, cluster, length_type)
        if sample_size:
            # self.size_sampler = SizeResampler(**sample_size_opt)
            self.size_sampler = SizeSamplerByPocketSpace()
        else:
            self.size_sampler = None

    def __getitem__(self, idx):
        if self.size_sampler is None:
            data = super().__getitem__(idx)
            # set position ids of small molecules to zero
            data['position_ids'][data['generate_mask']] = 0
            data['name']=self.get_summary(idx).id
            return data

        # change the size of the small molecule
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)

        lig_chain = summary.ligand_chain_ids[0]
        
        props = self._properties[idx]
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        
        pocket_pos = []
        for block_id in pocket_block_ids:
            for atom in recur_index(cplx, block_id):
                pocket_pos.append(atom.coordinate)
        size = self.size_sampler(1, pocket_pos)[0]
        cplx = remove_mols(cplx, summary.ligand_chain_ids) # remove ground truth molecule
        cplx = add_dummy_mol(cplx, size, summary.ligand_chain_ids[0])
        lig_block_ids = [(lig_chain, block.id) for block in cplx[lig_chain]]
        
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in lig_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in lig_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in lig_block_ids]
        
        data = transform_data(cplx, pocket_block_ids + lig_block_ids)
        data['generate_mask'] = torch.tensor(generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(center_mask, dtype=torch.bool)
        # set position ids of small molecules to zero
        data['position_ids'][data['generate_mask']] = 0
        data['name']=summary.id
        return data
    
    def get_expected_atom_num(self, idx):
        # change the size of the small molecule
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)

        props = self._properties[idx]
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        
        pocket_pos = []
        for block_id in pocket_block_ids:
            for atom in recur_index(cplx, block_id):
                pocket_pos.append(atom.coordinate)
        expect_atom_num = self.size_sampler.get_expected_atom_num(pocket_pos)
        return expect_atom_num
@R.register('MoleculeDatasetRAG')
class MoleculeDatasetRAG(PeptideDataset):

    def __init__(self, mmap_dir, specify_data = None, specify_index = None, cluster = None, length_type = 'atom', sample_size = False,
            feature_path:str ='./configs/contrastive/calculate/match_cache/ligand_mmap',# path to file containing the feature
            topn_index_path:str="",# path to topn index of every item
            topn_num:int=10):
        super().__init__(mmap_dir, specify_data, specify_index, cluster, length_type)
        if sample_size:
            # self.size_sampler = SizeResampler(**sample_size_opt)
            self.size_sampler = SizeSamplerByPocketSpace()
        else:
            self.size_sampler = None
        # rag
        self.length_type = length_type
        self.feature_path = feature_path
        self.topn_index_path = topn_index_path
        self.prompt_dataset=FeatureDataset(self.feature_path)
        with open(topn_index_path, 'r') as f:
            results_dict = json.load(f)
        self.topn=results_dict
        self.topn_num=topn_num
        # rag end

    def __getitem__(self, idx):
        if self.size_sampler is None:
            data = super().__getitem__(idx)
            # set position ids of small molecules to zero
            data['position_ids'][data['generate_mask']] = 0

            # rag
            data['name']=self.get_summary(idx).id
            topn :list[str] =self.topn[data['name']]
            features=[]
            i=0
            for ligand in topn:
                try:
                    if data['name']==ligand:
                        continue
                    features.append(torch.tensor(self.prompt_dataset.get_by_id(ligand)))
                    i+=1
                except Exception as e:
                    print(f'ligand {ligand} not found in prompt dataset')
                    print(e)
                    continue
                if i>=self.topn_num:
                    break

            data["prompt_feature"]=torch.stack(features, dim=0)# [topn,hidden_dim],after collate:[bs*topn,hidden_dim]
            del data['name']
            # rag_end
            return data

        # change the size of the small molecule
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)

        lig_chain = summary.ligand_chain_ids[0]
        
        props = self._properties[idx]
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        
        pocket_pos = []
        for block_id in pocket_block_ids:
            for atom in recur_index(cplx, block_id):
                pocket_pos.append(atom.coordinate)
        size = self.size_sampler(1, pocket_pos)[0]
        cplx = remove_mols(cplx, summary.ligand_chain_ids) # remove ground truth molecule
        cplx = add_dummy_mol(cplx, size, summary.ligand_chain_ids[0])
        lig_block_ids = [(lig_chain, block.id) for block in cplx[lig_chain]]
        
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in lig_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in lig_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in lig_block_ids]
        
        data = transform_data(cplx, pocket_block_ids + lig_block_ids)
        data['generate_mask'] = torch.tensor(generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(center_mask, dtype=torch.bool)
        # set position ids of small molecules to zero
        data['position_ids'][data['generate_mask']] = 0
        # rag
        data['name']=self.get_summary(idx).id
        topn :list[str] =self.topn[data['name']]
        features=[]
        i=0
        for ligand in topn:
            try:
                if data['name']==ligand:
                    continue
                features.append(torch.tensor(self.prompt_dataset.get_by_id(ligand)))
                i+=1
            except Exception as e:
                print(f'ligand {ligand} not found in prompt dataset')
                print(e)
                continue
            if i>=self.topn_num:
                break

        data["prompt_feature"]=torch.stack(features, dim=0)# [topn,hidden_dim],after collate:[bs*topn,hidden_dim]
        del data['name']
        # rag_end
        return data
    
    def get_expected_atom_num(self, idx):
        # change the size of the small molecule
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)

        props = self._properties[idx]
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        
        pocket_pos = []
        for block_id in pocket_block_ids:
            for atom in recur_index(cplx, block_id):
                pocket_pos.append(atom.coordinate)
        expect_atom_num = self.size_sampler.get_expected_atom_num(pocket_pos)
        return expect_atom_num

if __name__ == '__main__':
    import sys
    dataset = MoleculeDataset(sys.argv[1])
    print(dataset[0])
    print(len(dataset[0]['A']))