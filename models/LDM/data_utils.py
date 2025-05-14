#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from copy import deepcopy
from typing import List
from dataclasses import dataclass

import torch
from tqdm import tqdm
from rdkit import Chem

from data.bioparse import Complex, Block, Atom, BondType, VOCAB
from data.bioparse.utils import overwrite_block, is_standard_aa, index_to_numerical_index, bond_type_to_rdkit, bond_type_from_rdkit
from data.bioparse.tokenizer.tokenize_3d import TOKENIZER
from data.bioparse.hierarchy import remove_mols, add_dummy_mol, merge_cplx
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse.writer.rdkit_mol_to_sdf import rdkit_mol_to_sdf
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.base import transform_data
from evaluation.geom.check_twisted_bond import check_twisted_bond
from utils.chem_utils import valence_check, cycle_check, connect_fragments
from utils import reconstruct
from utils.logger import print_log


@dataclass
class OverwriteTask:

    cplx: Complex
    # necessary information
    select_indexes: list
    generate_mask: list
    target_chain_ids: list
    ligand_chain_ids: list
    # generated part
    S: list
    X: list
    A: list
    ll: list
    inter_bonds: tuple
    intra_bonds: list
    # output
    out_path: str

    def get_generated_seq(self):
        gen_seq = ''.join([VOCAB.idx_to_symbol(block_S) for block_S in self.S])
        return gen_seq

    def get_total_likelihood(self):
        flat_ll = []
        for block_ll in self.ll: flat_ll.extend(block_ll)
        return sum(flat_ll) / len(flat_ll)
    
    def get_overwritten_results(
            self,
            inter_block_bonds_obabel: bool=False,   # whether to use openbabel for inter-block bonds
            check_validity: bool=False,             # whether to check validity of small molecules
            expect_atom_num: int=None,              # discard very small molecules when checking validity
            filters: list=None                      # other filters (using the cplx as input)
        ):

        cplx = deepcopy(self.cplx)

        overwrite_indexes = [i for i, is_gen in zip(self.select_indexes, self.generate_mask) if is_gen]
        if len(overwrite_indexes) != len(self.X): # length change, need to modify the complex
            cplx, overwrite_indexes = modify_gen_length(cplx, len(self.X), self.ligand_chain_ids)

        assert len(overwrite_indexes) == len(self.X)
        assert len(self.X) == len(self.A)
        assert len(self.A) == len(self.ll)
    
        explicit_bonds, atom_idx_map, gen_mol, all_atom_coords = [], {}, Chem.RWMol(), []
        for i, index in enumerate(overwrite_indexes):
            block_S, block_X, block_A, block_ll, block_bonds = self.S[i], self.X[i], self.A[i], self.ll[i], self.intra_bonds[i]
            block_name = 'UNK' if block_S is None else VOCAB.idx_to_abrv(block_S)
            # construct a new block
            atoms, local2global = [], {}
            if block_bonds is None or is_standard_aa(block_name): # use canonical order
                canonical_order = VOCAB.abrv_to_atoms(block_name)
            else:
                canonical_order = []    # use the input order, which might be different from the canonical order
            for atom_order, (x, a, l) in enumerate(zip(block_X, block_A, block_ll)):
                atom_element = VOCAB.idx_to_atom(a)
                atoms.append(Atom(
                    # TODO: for structure prediction, the input order might be different from the canonical order
                    # However, atom names are not passed to the model
                    # Therefore, the only solution is that, for structure prediction, if there is a standard
                    # amino acid, then the input order of the atoms should align with the canonical order,
                    # which should be handled in the data processing logic, as the users only need to input the
                    # amino acid sequences
                    name=canonical_order[atom_order] if atom_order < len(canonical_order) else atom_element,
                    coordinate=x,
                    element=atom_element,
                    id=-1,  # normally this should be positive integer, set to -1 for later renumbering
                    properties={'bfactor': round(l, 2), 'occupancy': 1.0 }
                ))
                # update atom global index mapping to (block index, intra-block order)
                atom_idx_map[len(atom_idx_map)] = (index, atom_order)
                # update RWMol
                gen_mol.AddAtom(Chem.Atom(atom_element))
                all_atom_coords.append(x)
                # update local2global
                local2global[atom_order] = len(atom_idx_map) - 1
            # overwrite block
            overwrite_block(cplx, index, Block(
                name=block_name,
                atoms=atoms,
                id=index[1],
            ))
            # prepare intra-block bonds
            if block_bonds is None: block_bonds = VOCAB.abrv_to_bonds(block_name)
            else: block_bonds = [(src, dst, BondType(t)) for src, dst, t in zip(*block_bonds)]
            # add explicit bonds to record in PDB if the block is not a canonical amino acid
            if not is_standard_aa(block_name):
                numerical_index = index_to_numerical_index(cplx, index)
                for bond in block_bonds:
                    explicit_bonds.append((
                        (numerical_index[0], numerical_index[1], bond[0]),
                        (numerical_index[0], numerical_index[1], bond[1]),
                        bond[2]
                    ))
            # update bonds for RWMol
            for bond in block_bonds:
                begin, end = local2global[bond[0]], local2global[bond[1]]
                gen_mol.AddBond(begin, end, bond_type_to_rdkit(bond[2]))

        # processing inter-block bonds
        def format_prob_tuple(prob):
            conf, dist = prob
            dist_level = int(dist / 0.5) # [0, 0.5) - 0, [0.5, 1.0) - 1, [1.0, 1.5) - 2
            uncertainty = 1 - conf
            return (dist_level, uncertainty)

        if inter_block_bonds_obabel:
            try:
                print_log('using obabel to determine inter-block bonds')
                gen_mol = reconstruct.reconstruct_from_generated(all_atom_coords, [VOCAB.atom_to_idx(atom.GetSymbol()) for atom in gen_mol.GetAtoms()])
            except reconstruct.MolReconError:
                return None

        else: # using model predicted bonds
            if self.inter_bonds is not None:
                bond_tuples = []
                for atom_idx1, atom_idx2, prob, bond_type in zip(*self.inter_bonds): # idxs are global idxs
                    prob = format_prob_tuple(prob)
                    if bond_type == 4 and TOKENIZER.kekulize: continue # no aromatic bonds
                    bond_tuples.append((atom_idx1, atom_idx2, prob, BondType(bond_type))) # prob: confidence, distance
                bond_tuples = sorted(bond_tuples, key=lambda tup: tup[2]) # sorted by confidence
                for atom_idx1, atom_idx2, prob, bond_type in bond_tuples:
                    rdkit_bond = bond_type_to_rdkit(bond_type)
                    # bond_len = np.linalg.norm(np.array(all_atom_coords[atom_idx1]) - np.array(all_atom_coords[atom_idx2]))
                    if valence_check(gen_mol, atom_idx1, atom_idx2, rdkit_bond) and cycle_check(gen_mol.GetMol(), atom_idx1, atom_idx2, bond_type_to_rdkit(bond_type)): # and sp2_check(gen_mol.GetMol(), atom_idx1, atom_idx2, all_atom_coords):
                        # pass valence check and cycle check
                        gen_mol.AddBond(atom_idx1, atom_idx2, rdkit_bond)
                        # add to explicit bonds
                        index1, atom_order1 = atom_idx_map[atom_idx1]
                        numerical_index1 = index_to_numerical_index(cplx, index1)
                        index2, atom_order2 = atom_idx_map[atom_idx2]
                        numerical_index2 = index_to_numerical_index(cplx, index2)
                        explicit_bonds.append((
                            (numerical_index1[0], numerical_index1[1], atom_order1),
                            (numerical_index2[0], numerical_index2[1], atom_order2),
                            bond_type
                        ))
            # connect disconnected fragments
            gen_mol, added_bonds = connect_fragments(gen_mol, all_atom_coords)
            for atom_idx1, atom_idx2, rdkit_bond in added_bonds:
                # add to explicit bonds
                index1, atom_order1 = atom_idx_map[atom_idx1]
                numerical_index1 = index_to_numerical_index(cplx, index1)
                index2, atom_order2 = atom_idx_map[atom_idx2]
                numerical_index2 = index_to_numerical_index(cplx, index2)
                explicit_bonds.append((
                    (numerical_index1[0], numerical_index1[1], atom_order1),
                    (numerical_index2[0], numerical_index2[1], atom_order2),
                    bond_type_from_rdkit(rdkit_bond)
                ))

            gen_mol = gen_mol.GetMol()

        try: Chem.SanitizeMol(gen_mol)
        except Exception: pass
        smiles = Chem.MolToSmiles(gen_mol)

        if check_validity and (not validate_small_mol(gen_mol, smiles, all_atom_coords, expect_atom_num)):
            return None, None, None
        
        if filters is not None:
            for func in filters:
                if not func(cplx):
                    return None, None, None
        
        # save pdb
        complex_to_pdb(cplx, self.out_path, selected_chains=self.target_chain_ids + self.ligand_chain_ids, explict_bonds=explicit_bonds)
        # save sdf
        rdkit_mol_to_sdf(gen_mol, all_atom_coords, self.out_path.rstrip('.pdb') + '.sdf')
    
        return cplx, gen_mol, overwrite_indexes


def modify_gen_length(cplx: Complex, new_len: int, replace_ids: List[str]):
    cplx = remove_mols(cplx, replace_ids)
    cplx = add_dummy_mol(cplx, new_len, replace_ids[0])
    indexes = [(replace_ids[0], block.id) for block in cplx[replace_ids[0]]]
    return cplx, indexes


def validate_small_mol(mol, smiles, coords, expect_atom_num=None):
    if '.' in smiles: return False
    mol_size = mol.GetNumAtoms()
    if expect_atom_num is not None:
        if mol_size < expect_atom_num - 5:
            print_log(f'mol size {mol_size}, far below expectation {expect_atom_num}')
            return False # sometimes the model will converge to single blocks (like one indole)
    # if mol_size < 15: return False  # sometimes the model will converge to single blocks (like one indole)
    # validate bond length and angles. As we are predicting bonds by model,
    # there might be a few failed cases with many abnormal bond length and angles
    # between fragments. Such results should be discarded.
    (num_twist_bond, num_total_bond), (num_twist_angle, num_total_angle) = check_twisted_bond(mol, coords)
    rel_bond, rel_angle = num_twist_bond / mol_size, num_twist_angle / mol_size
    # rel_bond = num_twist_bond / (num_total_bond + 1e-10)
    # rel_angle = num_twist_angle / (num_total_angle + 1e-10)
    print_log(f'twist bond: {num_twist_bond}/{num_total_bond}, twist angle: {num_twist_angle}/{num_total_angle}, mol size: {mol_size}', level='DEBUG')
    return (rel_bond + rel_angle) < 0.1
    # return (rel_bond < 0.05) & (rel_angle < 0.05)


def _get_item(pdb_path, sdf_path, tgt_chains):
    # WARNING: only for small molecule (position ids are all zero for the generated part)
    pocket = pdb_to_complex(pdb_path, selected_chains=tgt_chains)
    ligand = sdf_to_complex(sdf_path)
    cplx = merge_cplx(pocket, ligand)
    pocket_block_ids = []
    for mol in pocket:
        for block in mol: pocket_block_ids.append((mol.id, block.id))
    ligand_block_ids = []
    ligand_chain = ligand.molecules[0].id
    for block in cplx[ligand_chain]: ligand_block_ids.append((ligand_chain, block.id))
    data = transform_data(cplx, pocket_block_ids + ligand_block_ids)
    data['generate_mask'] = torch.tensor([0 for _ in pocket_block_ids] + [1 for _ in ligand_block_ids], dtype=torch.bool)
    data['center_mask'] = torch.tensor([1 for _ in pocket_block_ids] + [0 for _ in ligand_block_ids], dtype=torch.bool)
    data['position_ids'][data['generate_mask']] = 0
    return data


class Recorder:
    def __init__(self, test_set, n_samples, save_dir):
        self.pbar = tqdm(total=n_samples * len(test_set))
        self.waiting_list = [(i, n) for n in range(n_samples) for i in range(len(test_set))]
        self.num_generated, self.num_failed = 0, 0
        self.fout = open(os.path.join(save_dir, 'results.jsonl'), 'w')

    def is_finished(self):
        return len(self.waiting_list) == 0

    def get_next_batch_list(self, batch_size):
        batch_list = self.waiting_list[:batch_size]
        self.waiting_list = self.waiting_list[batch_size:]
        return batch_list

    def check_and_save(self, log, item_idx, n, struct_only=False):
        self.num_generated += 1
        if log is None:
            self.num_failed += 1
            self.waiting_list.append((item_idx, n))
        else:
            log.update({
                'n': n,
                'struct_only': struct_only
            })
            self.fout.write(json.dumps(log) + '\n')
            self.fout.flush()
            self.pbar.update(1)

    def __del__(self):
        self.fout.close()