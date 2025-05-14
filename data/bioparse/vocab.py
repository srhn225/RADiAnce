#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from rdkit import Chem

from . import const
from .hierarchy import BondType
from .tokenizer.tokenize_3d import TOKENIZER
from .utils import bond_type_from_rdkit

class MoleculeVocab:

    def __init__(self):

        # load fragments (manually append single atoms)
        frags = []
        # add principal subgraphs
        for smi in TOKENIZER.get_frag_smiles():
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if len(mol.GetAtoms()) == 1: continue
            frags.append((f'f{len(frags)}', smi))
        # add atoms
        for element in const.periodic_table:
            frags.append((f'f{len(frags)}', f'[{element}]'))

        # block level vocab
        self.block_dummy = ('X', 'UNK')
        self.idx2block = [self.block_dummy] + const.aas + frags
        self.symbol2idx, self.abrv2idx = {}, {}
        self.aa_mask = []
        for i, (symbol, abrv) in enumerate(self.idx2block):
            self.symbol2idx[symbol] = i
            self.abrv2idx[abrv] = i
            self.aa_mask.append(True if abrv in const.AA_GEOMETRY else False)

        # atom level vocab
        self.atom_dummy = 'dummy'
        self.idx2atom = [self.atom_dummy] + const.periodic_table
        self.atom2idx = {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        
        # integrate physical information
        self.idx2atom_comp = [self.atom_dummy]
        for _, aa in const.aas:
            for atom in const.AA_GEOMETRY[aa]['atoms'] + const.backbone_atoms:
                atom_comp = const.AA_GEOMETRY[aa]['types'][atom]
                if atom_comp not in self.idx2atom_comp:
                    self.idx2atom_comp.append(atom_comp)
        self.atom_comp2idx = {}
        for i, atom in enumerate(self.idx2atom_comp):
            self.atom_comp2idx[atom] = i

        # atomic canonical orders & chemical bonds in each fragment
        self.atom_canonical_orders, self.element_canonical_orders = {}, {}
        self.chemical_bonds = {}
        for symbol, _ in const.aas:
            self.atom_canonical_orders[symbol] = const.backbone_atoms + const.sidechain_atoms[symbol]
            self.element_canonical_orders[symbol] = [name[0] for name in const.backbone_atoms + const.sidechain_atoms[symbol]] # only C, N, O, S
            atom2order = { a: i for i, a in enumerate(self.atom_canonical_orders[symbol]) }
            self.chemical_bonds[symbol] = [
                (atom2order[bond[0]], atom2order[bond[1]], BondType(bond[2])) for bond in const.aa_bonds[symbol]
            ]
        for symbol, smi in frags:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            self.atom_canonical_orders[symbol] = [atom.GetSymbol() for atom in mol.GetAtoms()] 
            self.element_canonical_orders[symbol] = [atom.GetSymbol() for atom in mol.GetAtoms()]
            self.chemical_bonds[symbol] = [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type_from_rdkit(bond)) for bond in mol.GetBonds()
            ]

    # block level APIs
    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.idx2block[idx][0]

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.idx2block[idx][1]

    def abrv_to_idx(self, abrv):
        return self.abrv2idx.get(abrv, self.abrv2idx['UNK'])

    def symbol_to_idx(self, symbol):
        return self.symbol2idx.get(symbol, self.abrv2idx['UNK'])
    
    def idx_to_symbol(self, idx):
        return self.idx2block[idx][0]

    def idx_to_abrv(self, idx):
        return self.idx2block[idx][1]
    
    def get_block_dummy_idx(self):
        return self.symbol_to_idx(self.block_dummy[0])

    # atom level APIs 
    def get_atom_dummy_idx(self):
        return self.atom2idx[self.atom_dummy]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx.get(atom, self.atom2idx[self.atom_dummy])

    # complicated atom type APIs
    def idx_to_atom_comp(self, idx):
        return self.idx2atom_comp[idx]

    def atom_comp_to_idx(self, atom_comp):
        return self.atom_comp2idx[atom_comp]
    
    def abrv_atomname_to_idx(self, abrv, atomname):
        if abrv not in const.AA_GEOMETRY:
            # TODO: bug what if Ca is passed in (Calthium)
            return self.atom_comp_to_idx(atomname[0]) # do not know this amino acid
        atom_comp = const.AA_GEOMETRY[abrv]['types'][atomname]
        return self.atom_comp_to_idx(atom_comp)
    
    def symbol_atomname_to_idx(self, symbol, atomname):
        abrv = self.symbol_to_abrv(symbol)
        return self.abrv_atomname_to_idx(abrv, atomname)

    # canonical order
    def abrv_to_atoms(self, abrv):
        symbol = self.abrv_to_symbol(abrv)
        return self.atom_canonical_orders.get(symbol, [])
    
    def abrv_to_elements(self, abrv):
        symbol = self.abrv_to_symbol(abrv)
        return self.element_canonical_orders.get(symbol, [])
    
    def abrv_to_bonds(self, abrv):
        symbol = self.abrv_to_symbol(abrv)
        return self.chemical_bonds.get(symbol, [])
    
    # sizes
    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_atom_comp_type(self):
        return len(self.idx2atom_comp)

    def get_num_block_type(self):
        return len(self.idx2block)

    def __len__(self):
        return len(self.symbol2idx)

    # others
    @property
    def ca_channel_idx(self):
        return const.backbone_atoms.index('CA')


VOCAB = MoleculeVocab()

