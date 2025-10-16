#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm
from copy import deepcopy
from typing import List

import yaml
import torch
from rdkit import Chem
import numpy as np

import models
from utils.config_utils import overwrite_values
# from utils.chem_utils import valence_check, cycle_check, connect_fragments
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
# from data.bioparse.writer.rdkit_mol_to_sdf import rdkit_mol_to_sdf
# from data.bioparse.parser.pdb_to_complex import pdb_to_complex
# from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.bioparse import Complex, Block, Atom, VOCAB, BondType
# from data.bioparse.utils import overwrite_block, is_standard_aa, index_to_numerical_index, bond_type_to_rdkit, bond_type_from_rdkit
# from data.bioparse.tokenizer.tokenize_3d import TOKENIZER
# from data.bioparse.hierarchy import remove_mols, add_dummy_mol, merge_cplx
from data.base import Summary, transform_data
from data import create_dataloader, create_dataset
from utils.logger import print_log
from utils.random_seed import setup_seed
# from utils import reconstruct
# from evaluation.geom.check_twisted_bond import check_twisted_bond
from models.LDM.data_utils import Recorder, OverwriteTask, _get_item


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k,v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k,v))

    # ckpts = sorted(ckpts, key=lambda x:x[0])
    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def clamp_coord(coord):
    # some models (e.g. diffab) will output very large coordinates (absolute value >1000) which will corrupt the pdb file
    new_coord = []
    for val in coord:
        if abs(val) >= 1000:
            val = 0
        new_coord.append(val)
    return new_coord


def generate_wrapper(model, sample_opt={}):
    # if isinstance(model, models.CondAutoEncoder):
    #     def wrapper(batch):
    #         X, S, confs = model.generate(**batch, **sample_opt) # WARN: outdated
    #         return X, S, confs, [None for _ in X]
    # elif isinstance(model, models.CondARAutoEncoder):
    #     def wrapper(batch):
    #         batch_X, batch_A, batch_ll = model.generate(**batch) # WARN: outdated
    #         return batch_X, batch_A, batch_ll, [None for _ in batch_X]
    # elif isinstance(model, models.CondIterAutoEncoder) or isinstance(model, models.CondIterAutoEncoderClean): # or isinstance(model, models.CondIterAutoEncoderKL) or isinstance(model, models.CondIterAutoEncoderDiff):
    #     def wrapper(batch):
    #         batch_S, batch_X, batch_A, batch_ll, batch_bonds = model.generate(**batch)
    #         batch_intra_bonds = []
    #         for s in batch_S:
    #             batch_intra_bonds.append([None for _ in s])
    #         return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds
    if isinstance(model, models.CondIterAutoEncoderEdge):
        def wrapper(batch):
            batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = model.generate(**batch)
            return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds
    elif isinstance(model, models.LDMMolDesignClean) or isinstance(model, models.LDMMolDesignRAG):# or isinstance(model, models.LFMMolDesign):
        def wrapper(batch):
            res_tuple = model.sample(sample_opt=sample_opt, **batch)
            if len(res_tuple) == 6:
                batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = res_tuple
            else:
                batch_S, batch_X, batch_A, batch_ll, batch_bonds = res_tuple
                batch_intra_bonds = []
                for s in batch_S:
                    batch_intra_bonds.append([None for _ in s])
            return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds
    else:
        raise NotImplementedError(f'Wrapper for {type(model)} not implemented')
    return wrapper


# def modify_gen_length(cplx: Complex, new_len: int, replace_ids: List[str]):
#     cplx = remove_mols(cplx, replace_ids)
#     cplx = add_dummy_mol(cplx, new_len, replace_ids[0])
#     indexes = [(replace_ids[0], block.id) for block in cplx[replace_ids[0]]]
#     return cplx, indexes
#     
# 
# def validate_small_mol(mol, smiles, coords, expect_atom_num=None):
#     if '.' in smiles: return False
#     mol_size = mol.GetNumAtoms()
#     if expect_atom_num is not None:
#         if mol_size < expect_atom_num - 5:
#             print_log(f'mol size {mol_size}, far below expectation {expect_atom_num}')
#             return False # sometimes the model will converge to single blocks (like one indole)
#     # if mol_size < 15: return False  # sometimes the model will converge to single blocks (like one indole)
#     # validate bond length and angles. As we are predicting bonds by model,
#     # there might be a few failed cases with many abnormal bond length and angles
#     # between fragments. Such results should be discarded.
#     (num_twist_bond, num_total_bond), (num_twist_angle, num_total_angle) = check_twisted_bond(mol, coords)
#     rel_bond, rel_angle = num_twist_bond / mol_size, num_twist_angle / mol_size
#     # rel_bond = num_twist_bond / (num_total_bond + 1e-10)
#     # rel_angle = num_twist_angle / (num_total_angle + 1e-10)
#     print_log(f'twist bond: {num_twist_bond}/{num_total_bond}, twist angle: {num_twist_angle}/{num_total_angle}, mol size: {mol_size}', level='DEBUG')
#     return (rel_bond + rel_angle) < 0.1
#     # return (rel_bond < 0.05) & (rel_angle < 0.05)


def overwrite(cplx: Complex, summary: Summary, S: list, X: list, A: list, ll: list, bonds: tuple, intra_bonds: list, out_path: str, sdf_obabel: bool=False, check_validity: bool=True, expect_atom_num=None):
    '''
        Args:
            bonds: [row, col, prob, type], row and col are atom index, prob has confidence and distance
    '''

    task = OverwriteTask(
        cplx = cplx,
        select_indexes = summary.select_indexes,
        generate_mask = summary.generate_mask,
        target_chain_ids = summary.target_chain_ids,
        ligand_chain_ids = summary.ligand_chain_ids,
        S = S,
        X = X,
        A = A,
        ll = ll,
        inter_bonds = bonds,
        intra_bonds = intra_bonds,
        out_path = out_path
    )

    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(
        inter_block_bonds_obabel = sdf_obabel,
        check_validity = check_validity,
        expect_atom_num = expect_atom_num
    )

    if cplx is None or gen_mol is None:
        return None

    return {
        'id': summary.id,
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': task.get_generated_seq(),
        'target_chains_ids': summary.target_chain_ids,
        'ligand_chains_ids': summary.ligand_chain_ids,
        'gen_block_idx': overwrite_indexes, # TODO: in pdb, (1, '0') will be saved as (1, 'A')
        'gen_pdb': os.path.abspath(out_path),
        'ref_pdb': os.path.abspath(summary.ref_pdb),
    }


    # cplx = deepcopy(cplx)
    # overwrite_indexes = [i for i, is_gen in zip(summary.select_indexes, summary.generate_mask) if is_gen]
    # if len(overwrite_indexes) != len(X): # length change, need to modify the complex
    #     cplx, overwrite_indexes = modify_gen_length(cplx, len(X), summary.ligand_chain_ids)

    # assert len(overwrite_indexes) == len(X)
    # assert len(X) == len(A)
    # assert len(A) == len(ll)
    # gen_seq = ''
    
    # flat_ll, explict_bonds, atom_idx_map, gen_mol, all_atom_coords = [], [], {}, Chem.RWMol(), []
    # for i, index in enumerate(overwrite_indexes):
    #     block_S, block_X, block_A, block_ll, block_bonds = S[i], X[i], A[i], ll[i], intra_bonds[i]
    #     block_name = 'UNK' if block_S is None else VOCAB.idx_to_abrv(block_S)
    #     gen_seq += VOCAB.idx_to_symbol(block_S)
    #     # construct a new block
    #     atoms, local2global = [], {}
    #     if block_bonds is None or is_standard_aa(block_name): # use canonical order
    #         canonical_order = VOCAB.abrv_to_atoms(block_name)
    #     else:
    #         canonical_order = []    # use the input order, which might be different from the canonical order
    #     for atom_order, (x, a, l) in enumerate(zip(block_X, block_A, block_ll)):
    #         flat_ll.append(l)
    #         atom_element = VOCAB.idx_to_atom(a)
    #         atoms.append(Atom(
    #             # TODO: for structure prediction, the input order might be different from the canonical order
    #             name=canonical_order[atom_order] if atom_order < len(canonical_order) else atom_element,
    #             coordinate=x,
    #             element=atom_element,
    #             id=-1,  # normally this should be positive integer, set to -1 for later renumbering
    #             properties={'bfactor': round(l, 2), 'occupancy': 1.0 }
    #         ))
    #         # update atom global index mapping to (block index, intra-block order)
    #         atom_idx_map[len(atom_idx_map)] = (index, atom_order)
    #         # update RWMol
    #         gen_mol.AddAtom(Chem.Atom(atom_element))
    #         all_atom_coords.append(x)
    #         # update local2global
    #         local2global[atom_order] = len(atom_idx_map) - 1
    #     # overwrite block
    #     overwrite_block(cplx, index, Block(
    #         name=block_name,
    #         atoms=atoms,
    #         id=index[1],
    #     ))
    #     # add explict bonds if the block is not a canonical amino acid
    #     if block_bonds is None:
    #         block_bonds = VOCAB.abrv_to_bonds(block_name)
    #     else:
    #         block_bonds = [(src, dst, BondType(t)) for src, dst, t in zip(*block_bonds)]
    #     if not is_standard_aa(block_name):
    #         numerical_index = index_to_numerical_index(cplx, index)
    #         for bond in block_bonds:
    #             explict_bonds.append((
    #                 (numerical_index[0], numerical_index[1], bond[0]),
    #                 (numerical_index[0], numerical_index[1], bond[1]),
    #                 bond[2]
    #             ))
    #     # update bonds for RWMol
    #     # print(local2global)
    #     for bond in block_bonds:
    #         begin, end = local2global[bond[0]], local2global[bond[1]]
    #         # print(begin, end, bond[2])
    #         gen_mol.AddBond(begin, end, bond_type_to_rdkit(bond[2]))

    
    # # inter-block bonds
    # def format_prob_tuple(prob):
    #     conf, dist = prob
    #     dist_level = int(dist / 0.5) # [0, 0.5) - 0, [0.5, 1.0) - 1, [1.0, 1.5) - 2
    #     uncertainty = 1 - conf
    #     return (dist_level, uncertainty)

    # # print(len(intra_bonds), len(bonds))
    # # print(intra_bonds)
    # # print('start adding bonds')
    # # for bond in gen_mol.GetBonds():
    # #     print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())

    # finish_flag = 0
    # if sdf_obabel:
    #     try:
    #         print('using obabel')
    #         gen_mol = reconstruct.reconstruct_from_generated(all_atom_coords, [VOCAB.atom_to_idx(atom.GetSymbol()) for atom in gen_mol.GetAtoms()])
    #         finish_flag = 1
    #     except reconstruct.MolReconError:
    #         return None
    # if finish_flag == 0: # using model predicted bonds
    #     if bonds is not None:
    #         bond_tuples = []
    #         for atom_idx1, atom_idx2, prob, bond_type in zip(*bonds): # idxs are global idxs
    #             prob = format_prob_tuple(prob)
    #             if bond_type == 4 and TOKENIZER.kekulize: continue # no aromatic bonds
    #             bond_tuples.append((atom_idx1, atom_idx2, prob, BondType(bond_type))) # prob: confidence, distance
    #         bond_tuples = sorted(bond_tuples, key=lambda tup: tup[2]) # sorted by confidence
    #         for atom_idx1, atom_idx2, prob, bond_type in bond_tuples:
    #             # print(atom_idx1, atom_idx2, bond_type)
    #             rdkit_bond = bond_type_to_rdkit(bond_type)
    #             # bond_len = np.linalg.norm(np.array(all_atom_coords[atom_idx1]) - np.array(all_atom_coords[atom_idx2]))
    #             if valence_check(gen_mol, atom_idx1, atom_idx2, rdkit_bond) and cycle_check(gen_mol.GetMol(), atom_idx1, atom_idx2, bond_type_to_rdkit(bond_type)): # and sp2_check(gen_mol.GetMol(), atom_idx1, atom_idx2, all_atom_coords):
    #                 # pass valence check and cycle check
    #                 gen_mol.AddBond(atom_idx1, atom_idx2, rdkit_bond)
    #                 # add to explicit bonds
    #                 index1, atom_order1 = atom_idx_map[atom_idx1]
    #                 numerical_index1 = index_to_numerical_index(cplx, index1)
    #                 index2, atom_order2 = atom_idx_map[atom_idx2]
    #                 numerical_index2 = index_to_numerical_index(cplx, index2)
    #                 explict_bonds.append((
    #                     (numerical_index1[0], numerical_index1[1], atom_order1),
    #                     (numerical_index2[0], numerical_index2[1], atom_order2),
    #                     bond_type
    #                 ))
    #     # connect disconnected fragments
    #     gen_mol, added_bonds = connect_fragments(gen_mol, all_atom_coords)
    #     for atom_idx1, atom_idx2, rdkit_bond in added_bonds:
    #         # add to explicit bonds
    #         index1, atom_order1 = atom_idx_map[atom_idx1]
    #         numerical_index1 = index_to_numerical_index(cplx, index1)
    #         index2, atom_order2 = atom_idx_map[atom_idx2]
    #         numerical_index2 = index_to_numerical_index(cplx, index2)
    #         explict_bonds.append((
    #             (numerical_index1[0], numerical_index1[1], atom_order1),
    #             (numerical_index2[0], numerical_index2[1], atom_order2),
    #             bond_type_from_rdkit(rdkit_bond)
    #         ))

    #     gen_mol = gen_mol.GetMol()

    # try: Chem.SanitizeMol(gen_mol)
    # except Exception: pass
    # smiles = Chem.MolToSmiles(gen_mol)

    # if check_validity and (not validate_small_mol(gen_mol, smiles, all_atom_coords, expect_atom_num)):
    #     return None
    
    # cplx_ll = sum(flat_ll) / len(flat_ll)
    # complex_to_pdb(cplx, out_path, selected_chains=summary.target_chain_ids + summary.ligand_chain_ids, explict_bonds=explict_bonds)
    # # save sdf
    # rdkit_mol_to_sdf(gen_mol, all_atom_coords, out_path.rstrip('.pdb') + '.sdf')
    # return {
    #         'id': summary.id,
    #         'pmetric': cplx_ll,
    #         'smiles': smiles,
    #         'gen_seq': gen_seq,
    #         'target_chains_ids': summary.target_chain_ids,
    #         'ligand_chains_ids': summary.ligand_chain_ids,
    #         'gen_block_idx': overwrite_indexes, # TODO: in pdb, (1, '0') will be saved as (1, 'A')
    #         'gen_pdb': os.path.abspath(out_path),
    #         'ref_pdb': os.path.abspath(summary.ref_pdb),
    # }


def format_id(summary: Summary):
    # format saving id for cross dock
    # e.g. BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb|BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf
    if '|' in summary.id:
        summary.id = summary.id.split('|')[0].strip('.pdb')


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    mode = config.get('sample_opt', {}).get('mode', 'codesign')
    struct_only = mode == 'fixseq'
    # load model
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
    print(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    _, _, test_set = create_dataset(config['dataset'])
    
    # save path
    if args.save_dir is None:
        save_dir = os.path.join(ckpt_dir, 'results')
    else:
        save_dir = args.save_dir
    ref_save_dir = os.path.join(save_dir, 'references')
    cand_save_dir = os.path.join(save_dir, 'candidates')
    tmp_cand_save_dir = os.path.join(save_dir, 'tmp_candidates')
    for directory in [ref_save_dir, cand_save_dir, tmp_cand_save_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    

    # fout = open(os.path.join(save_dir, 'results.jsonl'), 'w')

    n_samples = config.get('n_samples', 1)
    n_cycles = config.get('n_cycles', 0)

    recorder = Recorder(test_set, n_samples, save_dir)
    
    # pbar = tqdm(total=n_samples * len(test_set))
    # waiting_list = [(i, n) for n in range(n_samples) for i in range(len(test_set))]

    batch_size = config['dataloader']['batch_size']
    # num_generated, num_failed = 0, 0

    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = [test_set[i] for i, _ in batch_list]
        batch = test_set.collate_fn(batch)
        batch = to_device(batch, device)
        
        with torch.no_grad():
            batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generate_wrapper(model, deepcopy(config.get('sample_opt', {})))(batch)

        vae_batch_list = []
        for S, X, A, ll, bonds, intra_bonds, (item_idx, n) in zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, batch_list):
            cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
            summary: Summary = deepcopy(test_set.get_summary(item_idx))
            # revise id
            format_id(summary)
            summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
            if n == 0: # the first round
                os.makedirs(os.path.dirname(summary.ref_pdb), exist_ok=True)
                complex_to_pdb(cplx, summary.ref_pdb, summary.target_chain_ids + summary.ligand_chain_ids)
                os.makedirs(os.path.join(cand_save_dir, summary.id), exist_ok=True)
                os.makedirs(os.path.join(tmp_cand_save_dir, summary.id), exist_ok=True)
                complex_to_pdb(cplx, os.path.join(tmp_cand_save_dir, summary.id, 'pocket.pdb'), summary.target_chain_ids)
            if n_cycles == 0: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
            else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}.pdb')
            log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path, args.sdf_obabel, check_validity=False)
            if n_cycles == 0: recorder.check_and_save(log, item_idx, n, struct_only)
            else:
                vae_batch_list.append(
                    _get_item(
                        os.path.join(tmp_cand_save_dir, summary.id, f'pocket.pdb'),
                        save_path.rstrip('.pdb') + '.sdf',
                        summary.target_chain_ids
                    )
                )

        for cyc_i in range(n_cycles):
            print_log(f'Cycle: {cyc_i}', level='DEBUG')
            final_cycle = cyc_i == n_cycles - 1
            batch = test_set.collate_fn(vae_batch_list)
            batch = to_device(batch, device)
            vae_batch_list = []
            model_autoencoder = getattr(model, 'autoencoder', model)
            with torch.no_grad():
                if final_cycle: batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
                batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generate_wrapper(model_autoencoder, deepcopy(config.get('sample_opt', {})))(batch)
            for S, X, A, ll, bonds, intra_bonds, (item_idx, n) in zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, batch_list):
                cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
                summary: Summary = deepcopy(test_set.get_summary(item_idx))
                # revise id
                format_id(summary)
                summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
                if final_cycle: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
                else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}_cyc{cyc_i}.pdb')
                # get expect atom number
                if hasattr(test_set, 'get_expected_atom_num'):
                    expect_atom_num = test_set.get_expected_atom_num(item_idx)
                else: expect_atom_num = None
                log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path, args.sdf_obabel, check_validity=final_cycle, expect_atom_num=expect_atom_num)
                if final_cycle: recorder.check_and_save(log, item_idx, n, struct_only)
                else:
                    vae_batch_list.append(
                        _get_item(
                            os.path.join(tmp_cand_save_dir, summary.id, f'pocket.pdb'),
                            save_path.rstrip('.pdb') + '.sdf',
                            summary.target_chain_ids
                        )
                    )

        print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')
    return    


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated peptides')
    parser.add_argument('--sdf_obabel', action='store_true', help='Using openbabel to deduce bonds in SDF')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)