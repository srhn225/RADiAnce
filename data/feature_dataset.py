import torch

from data.bioparse.hierarchy import remove_mols, add_dummy_mol
from data.bioparse.utils import recur_index
from utils import register as R


from .resample import SizeResampler, SizeSamplerByPocketSpace
from .base import transform_data
import numpy as np
from typing import Union, Dict
from .mmap_dataset import MMAPDataset

class FeatureDataset(MMAPDataset):

    
    def __init__(self, mmap_dir: str, 
                 specify_data: str = None, 
                 specify_index: str = None,
                 auto_convert: bool = True):

        super().__init__(mmap_dir, specify_data, specify_index)
        


        self.id2idx = {}  # type: Dict[str, int]
        self._build_id_mapping()
        


        self.auto_convert = auto_convert
    
    def _build_id_mapping(self):
        for idx, (data_id, _, _) in enumerate(self._indexes):
            if data_id in self.id2idx:
                raise ValueError(f"Duplicate ID found: {data_id}")
            self.id2idx[data_id] = idx
    
    def __contains__(self, data_id: str) -> bool:
        return data_id in self.id2idx
    
    def get_by_id(self, data_id: str) -> Union[list, np.ndarray]:
        if data_id not in self.id2idx:
            raise KeyError(f"ID {data_id} not found in dataset")
        return self[self.id2idx[data_id]]
    
    def __getitem__(self, index: int) -> Union[list, np.ndarray]:
        data = super().__getitem__(index)
        
        if self.auto_convert:


            dim = self._properties[index].get("dim", None)
            if dim is not None:
                return np.array(data, dtype=np.float32).reshape(dim)
            return np.array(data, dtype=np.float32)
        return data
    
    @property
    def ids(self) -> list:
        return list(self.id2idx.keys())
    
    def get_properties(self, data_id: str) -> dict:
        if data_id not in self.id2idx:
            raise KeyError(f"ID {data_id} not found in dataset")
        return self._properties[self.id2idx[data_id]]





if __name__ == "__main__":
    dataset = FeatureDataset("features/protein/match_cache_protein/ligand_mmap")