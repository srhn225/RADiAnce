#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')

from data.mmap_dataset import create_mmap
from data.bioparse.parser.sdf_to_complex import sdf_to_complex

from utils.parallel_func import parallel_func


def parse():
    parser = argparse.ArgumentParser(description='Single molecular conformations')
    parser.add_argument('--smiles', type=str, required=True, help='Path to the smiles (.csv)')
    parser.add_argument('--save_sdf_dir', type=str, default=None, help='Directory to save sdf')
    parser.add_argument('--out_dir', type=str, required=True, help='Processed output directory')
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


def worker(i, item, sdf_out_dir):
    id_supplier, smiles = item

    # generate conformation (reference: https://github.com/deepmodeling/Uni-Mol/blob/main/unimol_tools/unimol_tools/data/conformer.py#L99)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None # failed to parse smiles
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)   # MMFF optimize
    except Exception:
        return None
    
    # save sdf
    out_path = os.path.join(sdf_out_dir, f'{i}.sdf')
    w = Chem.SDWriter(out_path)
    w.write(mol)
    w.close()

    # load sdf and change to data
    cplx = sdf_to_complex(out_path)
    data = cplx.to_tuple()

    lig_chain = cplx.molecules[0].id
    lig_blocks = cplx[lig_chain]

    props = {
        'id@supplier': id_supplier,
        'pocket_num_blocks': 0,
        'ligand_num_blocks': len(lig_blocks),
        'pocket_num_atoms': 0,
        'ligand_num_atoms': sum([len(block) for block in lig_blocks]),
        'target_chain_ids': [],
        'ligand_chain_ids': [lig_chain],
        'target_sequences': [],
        'ligand_sequences': [smiles],
        'pocket_block_id': [],
    }

    return str(i), data, props


def process_iterator(items, sdf_out_dir, num_cpus):

    # unordered for OOM problem
    generator = parallel_func(worker, [(i, item, sdf_out_dir) for i, item in enumerate(items)], n_cpus=num_cpus, unordered=True)

    cnt = 0
    for outputs in generator:
        cnt += 1
        if outputs is None: continue
        _id, data, properties = outputs
        yield _id, data, properties, cnt


def main(args):

    # create output directory
    if args.save_sdf_dir is not None:
        os.makedirs(args.save_sdf_dir, exist_ok=True)

    # get smiles
    with open(args.smiles, 'r') as fin:
        lines = fin.readlines()
    items = []
    for line in lines[1:]: # skip header
        smi, id_supplier = line.strip().split(',')
        items.append((id_supplier, smi))
    
    # generate conformations with RDKit
    create_mmap(
        process_iterator(items, args.save_sdf_dir, args.num_workers),
        args.out_dir, len(items), commit_batch=10000, abbr_desc_len=30)


if __name__ == '__main__':
    main(parse())