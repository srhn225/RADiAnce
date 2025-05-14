#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import random
from typing import Optional, List
import torch
from utils import register as R

from .resample import ClusterResampler
from .base import BaseDataset, Summary
from .feature_dataset import FeatureDataset
import json
@R.register('AntibodyDataset')
class AntibodyDataset(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            # cluster: Optional[str] = None,
            length_type: str = 'atom',
            cdr_type: List[str] = ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3'],
            test_mode: bool = False # extend all CDRs
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        # self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type
        self.test_mode = test_mode

        self.idx_tup = []
        if test_mode:       
            for idx, prop in enumerate(self._properties):
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: self.idx_tup.append((idx, 'HCDR1'))
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: self.idx_tup.append((idx, 'HCDR2'))
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: self.idx_tup.append((idx, 'HCDR3'))
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: self.idx_tup.append((idx, 'LCDR1'))
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: self.idx_tup.append((idx, 'LCDR2'))
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: self.idx_tup.append((idx, 'LCDR3'))
        else:
            for idx, prop in enumerate(self._properties):
                flag = False
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: flag = True 
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: flag = True 
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: flag = True 
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: flag = True 
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: flag = True 
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: flag = True 
                if flag: self.idx_tup.append((idx, None))

        self.cdr_type = cdr_type

    ########## Start of Overloading ##########
    def __len__(self):
        return len(self.idx_tup)

    def get_len(self, idx):
        props = self._properties[self.idx_tup[idx][0]]
        if self.length_type == 'atom':
            return props['epitope_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['epitope_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_raw_data(self, idx):
        idx, _ = self.idx_tup[idx]
        return super().get_raw_data(idx)

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        idx, cdr = self.idx_tup[idx]
        props = self._properties[idx]
        _id = self._indexes[idx][0]

        if cdr is None:
            assert not self.test_mode
            # randomly sample one available CDR
            choices = []
            for i in range(1, 4):
                if i in props['heavy_model_mark']: choices.append(f'HCDR{i}')
            for i in range(1, 4):
                if i in props['light_model_mark']: choices.append(f'LCDR{i}')
            cdr = random.choice(list(set(choices).intersection(set(self.cdr_type))))

        # get indexes (pocket + peptide)
        epitope_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['epitope_block_id']]
        hchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['heavy_model_block_id']]
        lchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['light_model_block_id']]

        generate_mask = [0 for _ in epitope_block_ids]
        for m in props['heavy_model_mark']:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in props['light_model_mark']:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        ref_seq = props['heavy_chain_sequence'] if cdr.startswith('H') else props['light_chain_sequence']
        mark = props['heavy_chain_mark'] if cdr.startswith('H') else props['light_chain_mark']

        start, end = mark.index(cdr[-1]), mark.rindex(cdr[-1])
        ref_seq = ref_seq[start:end + 1]

        return Summary(
            id=_id + '/' + cdr,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=ref_seq, # the selected CDR
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=epitope_block_ids + hchain_block_ids + lchain_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if len(item['bonds']) == 0: print(self.get_summary(idx))
        return item
    
@R.register('AntibodyDatasetPrompt')
class AntibodyDatasetPrompt(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            # cluster: Optional[str] = None,
            length_type: str = 'atom',
            cdr_type: List[str] = ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3'],
            test_mode: bool = False # extend all CDRs
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        # self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type
        self.test_mode = test_mode

        self.idx_tup = []
        if test_mode:       
            for idx, prop in enumerate(self._properties):
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: self.idx_tup.append((idx, 'HCDR1'))
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: self.idx_tup.append((idx, 'HCDR2'))
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: self.idx_tup.append((idx, 'HCDR3'))
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: self.idx_tup.append((idx, 'LCDR1'))
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: self.idx_tup.append((idx, 'LCDR2'))
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: self.idx_tup.append((idx, 'LCDR3'))
        else:
            for idx, prop in enumerate(self._properties):
                flag = False
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: flag = True 
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: flag = True 
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: flag = True 
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: flag = True 
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: flag = True 
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: flag = True 
                if flag: self.idx_tup.append((idx, None))

        self.cdr_type = cdr_type

    ########## Start of Overloading ##########
    def __len__(self):
        return len(self.idx_tup)

    def get_len(self, idx):
        props = self._properties[self.idx_tup[idx][0]]
        if self.length_type == 'atom':
            return props['epitope_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['epitope_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_raw_data(self, idx):
        idx, _ = self.idx_tup[idx]
        return super().get_raw_data(idx)

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        idx, cdr = self.idx_tup[idx]
        props = self._properties[idx]
        _id = self._indexes[idx][0]

        if cdr is None:
            assert not self.test_mode
            # randomly sample one available CDR
            choices = []
            for i in range(1, 4):
                if i in props['heavy_model_mark']: choices.append(f'HCDR{i}')
            for i in range(1, 4):
                if i in props['light_model_mark']: choices.append(f'LCDR{i}')
            cdr = random.choice(list(set(choices).intersection(set(self.cdr_type))))

        # get indexes (pocket + peptide)
        epitope_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['epitope_block_id']]
        hchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['heavy_model_block_id']]
        lchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['light_model_block_id']]

        generate_mask = [0 for _ in epitope_block_ids]
        for m in props['heavy_model_mark']:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in props['light_model_mark']:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        ref_seq = props['heavy_chain_sequence'] if cdr.startswith('H') else props['light_chain_sequence']
        mark = props['heavy_chain_mark'] if cdr.startswith('H') else props['light_chain_mark']

        start, end = mark.index(cdr[-1]), mark.rindex(cdr[-1])
        ref_seq = ref_seq[start:end + 1]

        return Summary(
            id=_id + '/' + cdr,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=ref_seq, # the selected CDR
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=epitope_block_ids + hchain_block_ids + lchain_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if len(item['bonds']) == 0: print(self.get_summary(idx))
        summary=self.get_summary(idx)
        item['name']=summary.id
        item['id']=summary.id.split("/",1)[0]
        item['cdr']=summary.id.split("/",1)[1]
        item["ref_seq"]=summary.ref_seq
        item["target_chain_ids"]=summary.target_chain_ids
        item["ligand_chain_ids"]=summary.ligand_chain_ids
        return item
    
@R.register('AntibodyDatasetRAG')
class AntibodyDatasetRAG(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            # cluster: Optional[str] = None,
            length_type: str = 'atom',
            cdr_type: List[str] = ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3'],
            test_mode: bool = False, # extend all CDRs
            feature_path:str ='./configs/contrastive/calculate/match_cache/ligand_mmap',# path to file containing the feature
            topn_index_path:str="",# path to topn index of every item
            topn_num:int=10
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.topn_num=topn_num
        # self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type
        self.test_mode = test_mode
        self.feature_path = feature_path
        self.topn_index_path = topn_index_path

        self.idx_tup = []
        if test_mode:       
            for idx, prop in enumerate(self._properties):
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: self.idx_tup.append((idx, 'HCDR1'))
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: self.idx_tup.append((idx, 'HCDR2'))
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: self.idx_tup.append((idx, 'HCDR3'))
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: self.idx_tup.append((idx, 'LCDR1'))
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: self.idx_tup.append((idx, 'LCDR2'))
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: self.idx_tup.append((idx, 'LCDR3'))
        else:
            for idx, prop in enumerate(self._properties):
                flag = False
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: flag = True 
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: flag = True 
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: flag = True 
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: flag = True 
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: flag = True 
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: flag = True 
                if flag: self.idx_tup.append((idx, None))

        self.cdr_type = cdr_type
        self.prompt_dataset=FeatureDataset(self.feature_path)
        with open(topn_index_path, 'r') as f:
            results_dict = json.load(f)
        self.topn=results_dict

    ########## Start of Overloading ##########
    def __len__(self):
        return len(self.idx_tup)

    def get_len(self, idx):
        props = self._properties[self.idx_tup[idx][0]]
        if self.length_type == 'atom':
            return props['epitope_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['epitope_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_raw_data(self, idx):
        idx, _ = self.idx_tup[idx]
        return super().get_raw_data(idx)

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        idx, cdr = self.idx_tup[idx]
        props = self._properties[idx]
        _id = self._indexes[idx][0]

        if cdr is None:
            assert not self.test_mode
            # randomly sample one available CDR
            choices = []
            for i in range(1, 4):
                if i in props['heavy_model_mark']: choices.append(f'HCDR{i}')
            for i in range(1, 4):
                if i in props['light_model_mark']: choices.append(f'LCDR{i}')
            cdr = random.choice(list(set(choices).intersection(set(self.cdr_type))))

        # get indexes (pocket + peptide)
        epitope_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['epitope_block_id']]
        hchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['heavy_model_block_id']]
        lchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['light_model_block_id']]

        generate_mask = [0 for _ in epitope_block_ids]
        for m in props['heavy_model_mark']:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in props['light_model_mark']:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        ref_seq = props['heavy_chain_sequence'] if cdr.startswith('H') else props['light_chain_sequence']
        mark = props['heavy_chain_mark'] if cdr.startswith('H') else props['light_chain_mark']

        start, end = mark.index(cdr[-1]), mark.rindex(cdr[-1])
        ref_seq = ref_seq[start:end + 1]

        return Summary(
            id=_id + '/' + cdr,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=ref_seq, # the selected CDR
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=epitope_block_ids + hchain_block_ids + lchain_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if len(item['bonds']) == 0: print(self.get_summary(idx))
        item['name']=self.get_summary(idx).id
        topn :list[str] =self.topn[item['name']]
        features=[]
        i=0
        for ligand in topn:
            try:
                if item['name']==ligand:
                    continue
                features.append(torch.tensor(self.prompt_dataset.get_by_id(ligand)))
                i+=1
            except Exception as e:
                print(f'ligand {ligand} not found in prompt dataset')
                print(e)
                continue
            if i>=self.topn_num:
                break

        item["prompt_feature"]=torch.stack(features, dim=0)# [topn,hidden_dim],after collate:[bs*topn,hidden_dim]
        del item['name']
        return item
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
            else:
                results[key] = torch.cat(values, dim=0)
        return results
from torch.nn.utils.rnn import pad_sequence    
@R.register('AntibodyDatasetRAGfull')
class AntibodyDatasetRAGfull(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            # cluster: Optional[str] = None,
            length_type: str = 'atom',
            cdr_type: List[str] = ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3'],
            test_mode: bool = False, # extend all CDRs
            feature_path:str ='./configs/contrastive/calculate/match_cache/ligand_mmap',# path to file containing the feature
            topn_index_path:str="",# path to topn index of every item
            topn_num:int=10
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.topn_num=topn_num
        # self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type
        self.test_mode = test_mode
        self.feature_path = feature_path
        self.topn_index_path = topn_index_path

        self.idx_tup = []
        if test_mode:       
            for idx, prop in enumerate(self._properties):
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: self.idx_tup.append((idx, 'HCDR1'))
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: self.idx_tup.append((idx, 'HCDR2'))
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: self.idx_tup.append((idx, 'HCDR3'))
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: self.idx_tup.append((idx, 'LCDR1'))
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: self.idx_tup.append((idx, 'LCDR2'))
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: self.idx_tup.append((idx, 'LCDR3'))
        else:
            for idx, prop in enumerate(self._properties):
                flag = False
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: flag = True 
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: flag = True 
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: flag = True 
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: flag = True 
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: flag = True 
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: flag = True 
                if flag: self.idx_tup.append((idx, None))

        self.cdr_type = cdr_type
        self.prompt_dataset=FeatureDataset(self.feature_path)
        with open(topn_index_path, 'r') as f:
            results_dict = json.load(f)
        self.topn=results_dict

    ########## Start of Overloading ##########
    def __len__(self):
        return len(self.idx_tup)

    def get_len(self, idx):
        props = self._properties[self.idx_tup[idx][0]]
        if self.length_type == 'atom':
            return props['epitope_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['epitope_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_raw_data(self, idx):
        idx, _ = self.idx_tup[idx]
        return super().get_raw_data(idx)

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        idx, cdr = self.idx_tup[idx]
        props = self._properties[idx]
        _id = self._indexes[idx][0]

        if cdr is None:
            assert not self.test_mode
            # randomly sample one available CDR
            choices = []
            for i in range(1, 4):
                if i in props['heavy_model_mark']: choices.append(f'HCDR{i}')
            for i in range(1, 4):
                if i in props['light_model_mark']: choices.append(f'LCDR{i}')
            cdr = random.choice(list(set(choices).intersection(set(self.cdr_type))))

        # get indexes (pocket + peptide)
        epitope_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['epitope_block_id']]
        hchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['heavy_model_block_id']]
        lchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['light_model_block_id']]

        generate_mask = [0 for _ in epitope_block_ids]
        for m in props['heavy_model_mark']:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in props['light_model_mark']:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        ref_seq = props['heavy_chain_sequence'] if cdr.startswith('H') else props['light_chain_sequence']
        mark = props['heavy_chain_mark'] if cdr.startswith('H') else props['light_chain_mark']

        start, end = mark.index(cdr[-1]), mark.rindex(cdr[-1])
        ref_seq = ref_seq[start:end + 1]

        return Summary(
            id=_id + '/' + cdr,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=ref_seq, # the selected CDR
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=epitope_block_ids + hchain_block_ids + lchain_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if len(item['bonds']) == 0: print(self.get_summary(idx))
        item['name']=self.get_summary(idx).id
        topn :list[str] =self.topn[item['name']]
        features=[]
        i=0
        for ligand in topn:
            try:
                if item['name']==ligand:
                    continue
                features.append(torch.tensor(self.prompt_dataset.get_by_id(ligand)))
                i+=1
            except Exception as e:
                print(f'ligand {ligand} not found in prompt dataset')
                print(e)
                continue
            if i>=self.topn_num:
                break

        item["prompt_feature"]=torch.cat(features,dim=0)# [total_len,hiddendim]
        del item['name']
        return item
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
    
    
@R.register('AntibodyDatasetCopycat')
class AntibodyDatasetRAGCopycat(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            # cluster: Optional[str] = None,
            length_type: str = 'atom',
            cdr_type: List[str] = ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3'],
            test_mode: bool = False, # extend all CDRs
            feature_path:str ='./configs/contrastive/calculate/match_cache/ligand_mmap',# path to file containing the feature
            topn_index_path:str="",# path to topn index of every item
            topn_num:int=10
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.topn_num=topn_num
        # self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type
        self.test_mode = test_mode
        self.feature_path = feature_path
        self.topn_index_path = topn_index_path

        self.idx_tup = []
        if test_mode:       
            for idx, prop in enumerate(self._properties):
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: self.idx_tup.append((idx, 'HCDR1'))
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: self.idx_tup.append((idx, 'HCDR2'))
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: self.idx_tup.append((idx, 'HCDR3'))
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: self.idx_tup.append((idx, 'LCDR1'))
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: self.idx_tup.append((idx, 'LCDR2'))
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: self.idx_tup.append((idx, 'LCDR3'))
        else:
            for idx, prop in enumerate(self._properties):
                flag = False
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: flag = True 
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: flag = True 
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: flag = True 
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: flag = True 
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: flag = True 
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: flag = True 
                if flag: self.idx_tup.append((idx, None))

        self.cdr_type = cdr_type
        self.prompt_dataset=FeatureDataset(self.feature_path)
        with open(topn_index_path, 'r') as f:
            results_dict = json.load(f)
        self.topn=results_dict

    ########## Start of Overloading ##########
    def __len__(self):
        return len(self.idx_tup)

    def get_len(self, idx):
        props = self._properties[self.idx_tup[idx][0]]
        if self.length_type == 'atom':
            return props['epitope_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['epitope_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_raw_data(self, idx):
        idx, _ = self.idx_tup[idx]
        return super().get_raw_data(idx)

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        idx, cdr = self.idx_tup[idx]
        props = self._properties[idx]
        _id = self._indexes[idx][0]

        if cdr is None:
            assert not self.test_mode
            # randomly sample one available CDR
            choices = []
            for i in range(1, 4):
                if i in props['heavy_model_mark']: choices.append(f'HCDR{i}')
            for i in range(1, 4):
                if i in props['light_model_mark']: choices.append(f'LCDR{i}')
            cdr = random.choice(list(set(choices).intersection(set(self.cdr_type))))

        # get indexes (pocket + peptide)
        epitope_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['epitope_block_id']]
        hchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['heavy_model_block_id']]
        lchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['light_model_block_id']]

        generate_mask = [0 for _ in epitope_block_ids]
        for m in props['heavy_model_mark']:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in props['light_model_mark']:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        ref_seq = props['heavy_chain_sequence'] if cdr.startswith('H') else props['light_chain_sequence']
        mark = props['heavy_chain_mark'] if cdr.startswith('H') else props['light_chain_mark']

        start, end = mark.index(cdr[-1]), mark.rindex(cdr[-1])
        ref_seq = ref_seq[start:end + 1]

        return Summary(
            id=_id + '/' + cdr,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=ref_seq, # the selected CDR
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=epitope_block_ids + hchain_block_ids + lchain_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if len(item['bonds']) == 0: 
            print(self.get_summary(idx))
        item_name = self.get_summary(idx).id
        


        try:
            self_feature = torch.tensor(self.prompt_dataset.get_by_id(item_name))
            features = [self_feature] * self.topn_num
        except Exception as e:
            raise RuntimeError(f"Cannot get self feature for {item_name}") from e
        


        item["prompt_feature"] = torch.stack(features, dim=0)
        return item
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
            else:
                results[key] = torch.cat(values, dim=0)
        return results
    
    
    
    
if __name__ == '__main__':
    import sys
    dataset = AntibodyDataset(sys.argv[1], specify_index=sys.argv[2])
    print(dataset[0])
    print(len(dataset[0]['position_ids']))