#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import numpy as np
from rdkit.Chem import rdchem
from biotite.structure import BondType as BT

from . import const
from .hierarchy import Atom, Block, Complex, BondType


def is_aa(block: Block) -> bool:
    '''return whether this block is an amino acid'''
    return is_standard_aa(block.name)
    if block.name in const.AA_GEOMETRY: # known amino acids
        return True
    # non-canonical amino acids: with N, CA, C, O (WARN: not so reliable)
    profile = { atom_name: False for atom_name in const.backbone_atoms }
    for atom in block:
        if atom.name in profile: profile[atom.name] = True
    for atom_name in profile:
        if not profile[atom_name]: return False
    return True


def is_standard_aa(abrv: str) -> bool:
    '''3-code abbreviation'''
    return abrv in const.chi_angles_atoms


def bond_type_from_rdkit(bond: rdchem.Bond) -> BondType:
    '''Convert RDKit bond type to custom BondType.'''
    if bond == rdchem.BondType.SINGLE or bond.GetBondType() == rdchem.BondType.SINGLE:
        return BondType.SINGLE
    elif bond == rdchem.BondType.DOUBLE or bond.GetBondType() == rdchem.BondType.DOUBLE:
        return BondType.DOUBLE
    elif bond == rdchem.BondType.TRIPLE or bond.GetBondType() == rdchem.BondType.TRIPLE:
        return BondType.TRIPLE
    elif bond == rdchem.BondType.AROMATIC or bond.GetBondType() == rdchem.BondType.AROMATIC:
        return BondType.AROMATIC
    else:
        return BondType.NONE
    

def bond_type_from_biotite(bond):
    if isinstance(bond, BT):
        biotite_bond = bond
    else: 
        biotite_bond = BT(bond)
    return BondType(biotite_bond.without_aromaticity().value)


def bond_type_to_rdkit(bond: BondType):
    '''Convert custom BondType to RDKit bond type'''
    if bond == BondType.SINGLE:
        return rdchem.BondType.SINGLE
    elif bond == BondType.DOUBLE:
        return rdchem.BondType.DOUBLE
    elif bond == BondType.TRIPLE:
        return rdchem.BondType.TRIPLE
    elif bond == BondType.AROMATIC:
        return rdchem.BondType.AROMATIC
    else:
        return None
    

def extract_atom_coords(block: Block, names: List[str]) -> List:
    '''extract atom coords given the names'''
    name2coords = {}
    for atom in block:
        name2coords[atom.name] = atom.get_coord()
    coords = []
    for name in names: coords.append(name2coords.get(name, None))
    return coords


def recur_index(obj, index: tuple):
    for _id in index:
        obj = obj[_id]
    return obj


def index_to_numerical_index(obj, index: tuple):
    numerical_index = []
    for i in index:
        numerical_index.append(obj.id2idx[i])
        obj = obj[i]
    return tuple(numerical_index)


def overwrite_block(cplx: Complex, index: tuple, block: Block):
    if not isinstance(index[0], int): index = index_to_numerical_index(cplx, index)
    mol = cplx[index[0]]
    mol.blocks[index[1]] = block


def format_standard_aa_block(block: Block) -> Block:
    assert is_standard_aa(block)
    # TODO: rename atoms according to the amino acid type
    # i.e. for atom in block: atom.name = formatted name (N, CA, C, O, CB, ...)
    return None


def renumber_res_id(res_ids: List[tuple]):
    # assume res_ids are ordered within each chain
    offset_map = {}
    new_res_ids = []
    for chain, (res_nb, insert_code) in res_ids:
        if chain not in offset_map: offset_map[chain] = 0
        if insert_code == '':
            new_res_ids.append((chain, (res_nb + offset_map[chain], '')))
        else:
            offset_map[chain] += 1
            new_res_ids.append((chain, (res_nb + offset_map[chain], '')))
    return new_res_ids