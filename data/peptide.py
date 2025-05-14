#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import Optional

from utils import register as R

from .resample import ClusterResampler
from .base import BaseDataset, Summary
from .feature_dataset import FeatureDataset
import json
import torch
from torch.nn.utils.rnn import pad_sequence    
@R.register('PeptideDataset')
class PeptideDataset(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            cluster: Optional[str] = None,
            length_type: str = 'atom'
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    ########## Start of Overloading ##########
    def get_id(self, idx):
        idx = self.dynamic_idxs[idx]
        return self._indexes[idx][0]

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        props = self._properties[idx]
        if self.length_type == 'atom':
            return props['pocket_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['pocket_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        _id = self._indexes[idx][0]
        props = self._properties[idx]

        # get indexes (pocket + peptide)
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        cplx = self.get_raw_data(idx)
        pep_chain = props['ligand_chain_ids'][0]
        pep_block_ids = [(pep_chain, block.id) for block in cplx[pep_chain]]
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in pep_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in pep_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in pep_block_ids]

        return Summary(
            id=_id,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=props['ligand_sequences'][0], # peptide has only one chain
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=pocket_block_ids + pep_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        data = super().__getitem__(idx)
        # peptide position ids start from 1\
        gen_mask = data['generate_mask']
        pep_position_ids = data['position_ids'][gen_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data['position_ids'][gen_mask] = pep_position_ids
        return data
    
@R.register('PeptideDatasetPrompt')
class PeptideDatasetPrompt(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            cluster: Optional[str] = None,
            length_type: str = 'atom'
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    ########## Start of Overloading ##########
    def get_id(self, idx):
        idx = self.dynamic_idxs[idx]
        return self._indexes[idx][0]

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        props = self._properties[idx]
        if self.length_type == 'atom':
            return props['pocket_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['pocket_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        _id = self._indexes[idx][0]
        props = self._properties[idx]

        # get indexes (pocket + peptide)
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        cplx = self.get_raw_data(idx)
        pep_chain = props['ligand_chain_ids'][0]
        pep_block_ids = [(pep_chain, block.id) for block in cplx[pep_chain]]
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in pep_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in pep_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in pep_block_ids]

        return Summary(
            id=_id,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=props['ligand_sequences'][0], # peptide has only one chain
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=pocket_block_ids + pep_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        data = super().__getitem__(idx)
        # peptide position ids start from 1\
        gen_mask = data['generate_mask']
        pep_position_ids = data['position_ids'][gen_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data['position_ids'][gen_mask] = pep_position_ids
        summary=self.get_summary(idx)
        data["name"]=summary.id
        data['id']=summary.id
        data['cdr']= None
        data["ref_seq"]=summary.ref_seq
        data["target_chain_ids"]=summary.target_chain_ids
        data["ligand_chain_ids"]=summary.ligand_chain_ids
        return data
@R.register('PeptideDatasetRAG')
class PeptideDatasetRAG(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            cluster: Optional[str] = None,
            length_type: str = 'atom',
            feature_path:str ='./configs/contrastive/calculate/match_cache/ligand_mmap',# path to file containing the feature
            topn_index_path:str="",# path to topn index of every item
            topn_num:int=10
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
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

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    ########## Start of Overloading ##########
    def get_id(self, idx):
        idx = self.dynamic_idxs[idx]
        return self._indexes[idx][0]

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        props = self._properties[idx]
        if self.length_type == 'atom':
            return props['pocket_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['pocket_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        _id = self._indexes[idx][0]
        props = self._properties[idx]

        # get indexes (pocket + peptide)
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        cplx = self.get_raw_data(idx)
        pep_chain = props['ligand_chain_ids'][0]
        pep_block_ids = [(pep_chain, block.id) for block in cplx[pep_chain]]
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in pep_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in pep_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in pep_block_ids]

        return Summary(
            id=_id,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=props['ligand_sequences'][0], # peptide has only one chain
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=pocket_block_ids + pep_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        data = super().__getitem__(idx)
        # peptide position ids start from 1\
        gen_mask = data['generate_mask']
        pep_position_ids = data['position_ids'][gen_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data['position_ids'][gen_mask] = pep_position_ids
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


@R.register('PeptideDatasetRAGfull')
class PeptideDatasetRAGfull(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            cluster: Optional[str] = None,
            length_type: str = 'atom',
            feature_path:str ='./configs/contrastive/calculate/match_cache/ligand_mmap',# path to file containing the feature
            topn_index_path:str="",# path to topn index of every item
            topn_num:int=10
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
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

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    ########## Start of Overloading ##########
    def get_id(self, idx):
        idx = self.dynamic_idxs[idx]
        return self._indexes[idx][0]

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        props = self._properties[idx]
        if self.length_type == 'atom':
            return props['pocket_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['pocket_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        _id = self._indexes[idx][0]
        props = self._properties[idx]

        # get indexes (pocket + peptide)
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        cplx = self.get_raw_data(idx)
        pep_chain = props['ligand_chain_ids'][0]
        pep_block_ids = [(pep_chain, block.id) for block in cplx[pep_chain]]
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in pep_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in pep_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in pep_block_ids]

        return Summary(
            id=_id,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=props['ligand_sequences'][0], # peptide has only one chain
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=pocket_block_ids + pep_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        data = super().__getitem__(idx)
        # peptide position ids start from 1\
        gen_mask = data['generate_mask']
        pep_position_ids = data['position_ids'][gen_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data['position_ids'][gen_mask] = pep_position_ids
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

        data["prompt_feature"]=torch.cat(features, dim=0)# [totallen,hd]
        del data['name']
        # rag_end
        return data   
    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'name':
                results[key] = values
            elif key == 'bonds': # need to add offsets
                offset = 0
                for i, bonds in enumerate(values):
                    bonds[:, :2] = bonds[:, :2] + offset # src/dst
                    offset += len(batch[i]['A'])
                results[key] = torch.cat(values, dim=0)
            elif key == 'prompt_feature':
                
                


                padded_tensor = pad_sequence(values, batch_first=True, padding_value=0.0)# [bs,total_len_max,hiddendim]
                max_len = padded_tensor.shape[1]


                mask = torch.ones((len(batch), max_len), dtype=bool)
                for i, seq in enumerate(values):
                    mask[i, :seq.shape[0]] =False

                results[key] = padded_tensor
                results['prompt_mask'] = mask
            else:
                results[key] = torch.cat(values, dim=0)
        return results    
if __name__ == '__main__':
    import sys
    dataset = PeptideDataset(sys.argv[1])
    print(dataset[0])
    print(len(dataset[0]['A']))