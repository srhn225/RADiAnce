#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import BondType
from rdkit.Geometry import Point3D

BOND_CONFIG = None
DISTANCE_BINS = None
ANGLE_CONFIG = None


def _init():
    global BOND_CONFIG, DISTANCE_BINS, ANGLE_CONFIG
    BOND_CONFIG = np.load(os.path.join(
        os.path.dirname(__file__), '_ref_length_distribution.npy'), allow_pickle=True
    ).tolist()
    DISTANCE_BINS = np.arange(1.1, 1.7, 0.005)[:-1]

    for v in BOND_CONFIG.values():
        assert len(DISTANCE_BINS) + 1 == len(v)

    ANGLE_CONFIG = np.load(os.path.join(
        os.path.dirname(__file__), '_ref_angle_distribution.npy'), allow_pickle=True
    ).tolist()


_init()

BOND_TYPES = {
        BondType.UNSPECIFIED: 0,
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.AROMATIC: 4,
    }

def get_bond_type(bond):
    return BOND_TYPES[bond.GetBondType()]

def bond_length_from_mol(mol):

    pos = mol.GetConformer().GetPositions()
    pdist = pos[None, :] - pos[:, None]
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))
    defs, all_distances = [], []
    for bond in mol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = get_bond_type(bond)
        distance = pdist[s_idx, e_idx]
        if s_sym > e_sym: s_sym, e_sym = e_sym, s_sym
        defs.append((s_sym, e_sym, bond_type))
        all_distances.append(distance)
    return defs, all_distances


def get_bond_length_probs(bond_defs, distances):
    assert len(bond_defs) == len(distances)
    # to distance bins
    bins = np.searchsorted(DISTANCE_BINS, distances)
    # to probabilities
    probs = []
    for bond_def, bin_idx in zip(bond_defs, bins):
        if bond_def not in BOND_CONFIG: continue
        probs.append(BOND_CONFIG[bond_def][bin_idx])
    return probs


def bond_length_probs_from_mol(mol):
    defs, dists = bond_length_from_mol(mol)
    return get_bond_length_probs(defs, dists)



def bond_angle_from_mol(mol):
    angles = []
    types = [] # double counting
    conf =  mol.GetConformer(id=0)
    for atom in mol.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 2:  # At least two neighbors are required to form an angle
            continue
        for a1, a2 in combinations(neighbors, 2):
            angle = rdMolTransforms.GetAngleDeg(conf, a1, atom.GetIdx(), a2)
            tup = (mol.GetAtomWithIdx(a1).GetAtomicNum(), 
                   get_bond_type(mol.GetBondBetweenAtoms(a1, atom.GetIdx())), 
                   atom.GetAtomicNum(), 
                   get_bond_type(mol.GetBondBetweenAtoms(atom.GetIdx(), a2)), 
                   mol.GetAtomWithIdx(a2).GetAtomicNum())
            idx = (a1, atom.GetIdx(), a2)
            angles.append(angle)
            types.append(tup)

            tup_rev = (mol.GetAtomWithIdx(a2).GetAtomicNum(), 
                       get_bond_type(mol.GetBondBetweenAtoms(a2, atom.GetIdx())), 
                       atom.GetAtomicNum(), 
                       get_bond_type(mol.GetBondBetweenAtoms(atom.GetIdx(), a1)), 
                       mol.GetAtomWithIdx(a1).GetAtomicNum())
            types.append(tup_rev)
            angles.append(angle)

    return types, angles


def get_bond_angle_probs(angle_defs, angles):
    assert len(angle_defs) == len(angles)
    # to distance bins
    bins = np.searchsorted(np.arange(0, 180, 2), angles)
    # to probabilities
    probs = []
    for angle_def, bin_idx in zip(angle_defs, bins):
        if angle_def not in ANGLE_CONFIG: continue
        probs.append(ANGLE_CONFIG[angle_def][bin_idx])
    return probs

    
def bond_angle_probs_from_mol(mol):
    defs, angles = bond_angle_from_mol(mol)
    return get_bond_angle_probs(defs, angles)


def check_twisted_bond(mol, atom_coords=None):
    if atom_coords is not None:
        num_atoms = mol.GetNumAtoms()
        conformer = Chem.Conformer(num_atoms)
        for i, (x, y, z) in enumerate(atom_coords):
            conformer.SetAtomPosition(i, Point3D(x, y, z))

        # Attach the conformer to the molecule
        mol.AddConformer(conformer)
    bond_len_probs = bond_length_probs_from_mol(mol)
    bond_angle_probs = bond_angle_probs_from_mol(mol)
    num_twist_bond = sum([1 if p == 0 else 0 for p in bond_len_probs])
    num_twist_angle = sum([1 if p == 0 else 0 for p in bond_angle_probs])
    return (num_twist_bond, len(bond_len_probs)), (num_twist_angle, len(bond_angle_probs))



if __name__ == '__main__':
    import sys
    from rdkit import Chem

    mol = Chem.SDMolSupplier(sys.argv[1])[0]

    defs, angles = bond_angle_from_mol(mol)
    print(get_angle_probs(defs, angles))
