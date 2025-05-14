#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import numpy as np

from .hierarchy import Block, Complex
from .utils import is_aa, extract_atom_coords


def add_cb(input_array):
    #from protein mpnn
    #The virtual Cβ coordinates were calculated using ideal angle and bond length definitions: b = Cα - N, c = C - Cα, a = cross(b, c), Cβ = -0.58273431*a + 0.56802827*b - 0.54067466*c + Cα.
    N,CA,C,O = input_array
    b = CA - N
    c = C - CA
    a = np.cross(b,c)
    CB = np.around(-0.58273431*a + 0.56802827*b - 0.54067466*c + CA,3)
    return CB #np.array([N,CA,C,CB,O])


def _all_not_none(vals: List):
    if len(vals) == 0: return True
    return vals[0] is not None and _all_not_none(vals[1:])


def blocks_to_cb_coords(blocks):
    cb_coords = []
    for block in blocks:
        coords = extract_atom_coords(block, ['CB', 'N', 'CA', 'C', 'O'])
        if coords[0] is not None: cb_coords.append(coords[0])
        elif _all_not_none(coords[1:]): cb_coords.append(add_cb(np.array(coords[1:])))
        else:
            coords = [atom.get_coord() for atom in block]
            cb_coords.append(np.mean(coords, axis=0))
    return np.array(cb_coords)


def compute_pocket(cplx: Complex, id_set1: List[str], id_set2: List[str], dist_th: float=10.0):
    '''
        Compute the pocket block indexes between two parts defined by id_set1 and id_set2.
        For amino acids, the coordinate of Cb is used to calculate distances.
        For small molecules, the center of mass is used to calculate distances.
        dist_th defines the distance cutoff for extracting the pocket
    '''
    def _extract_block_index(id_set):
        blocks, indexes = [], []
        for _id in id_set:
            molecule = cplx[_id]
            blocks.extend(molecule)
            for block in molecule: indexes.append((_id, block.id)) # (chain id, block id)
        return blocks, indexes
    
    blocks1, indexes1 = _extract_block_index(id_set1)
    blocks2, indexes2 = _extract_block_index(id_set2)

    cb_coords1 = blocks_to_cb_coords(blocks1)
    cb_coords2 = blocks_to_cb_coords(blocks2)
    dist = np.linalg.norm(cb_coords1[:, None] - cb_coords2[None, :], axis=-1)  # [N1, N2]
    
    on_interface = dist < dist_th
    if_indexes1 = np.nonzero(on_interface.sum(axis=1) > 0)[0]
    if_indexes2 = np.nonzero(on_interface.sum(axis=0) > 0)[0]

    return (
        [indexes1[i] for i in if_indexes1], # pocket on chains in id_set1
        [indexes2[i] for i in if_indexes2]  # pocket on chains in id_set2
    )
