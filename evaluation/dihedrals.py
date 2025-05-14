#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np # to solve deprecated notions in numpy
np.bool = np.bool_
np.int = np.int32

import os
import json
import warnings
import collections

warnings.filterwarnings("ignore")

from tqdm.contrib.concurrent import process_map
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy.spatial.distance import jensenshannon


def dihedrals(pdb, selected_chains=None, selected_residues=None):
    '''
        Args:
            pdb: str, path
            selected_chains: e.g. ['A', 'B']
            selected_residues: e.g. [('B', (192, '')), ('A', (101, 'C'))]
    '''
    # Load the structure
    parser = PDBParser()
    structure = parser.get_structure('x', pdb)
    structure.atom_to_internal_coordinates()

    bb_angles, sc_angles = {}, {}

    # cannot specify both selections
    assert not (selected_chains is not None and selected_residues is not None)

    if selected_residues is not None:
        residues = []
        for (chain, res_id) in selected_residues:
            insert_code = ' ' if res_id[1] == '' else res_id[1]
            res_id = (' ', res_id[0], insert_code)
            try:
                residue = structure[0][chain][res_id]
            except KeyError: continue
            residues.append(residue)
    elif selected_chains is not None:
        residues = []
        for chain in structure[0]:
            if chain.id in selected_chains:
                residues.extend(chain.get_residues())
    else:
        residues = []
        for chain in structure[0]: residues.extend(chain.get_residues())
    
    for residue in residues:

        res_name = residue.resname
        if res_name not in protein_letters_3to1: continue # non-standard

        # backbone dihedrals
        if res_name not in bb_angles: bb_angles[res_name] = {}
        for angle in ['psi', 'phi', 'omg']:
            if angle not in bb_angles[res_name]: bb_angles[res_name][angle] = []
            try:
                val = residue.internal_coord.get_angle(angle)
                if val is None: continue
            except KeyError:
                continue # start/end
            bb_angles[res_name][angle].append(round(val, 2))

        # sidechain dihedrals
        if res_name not in sc_angles: sc_angles[res_name] = {}
        for chi in [f'chi{i}' for i in range(1, 5)]:
            try:
                val = residue.internal_coord.get_angle(chi)
                if val is None: continue
            except KeyError:
                continue # start/end
            if chi not in sc_angles[res_name]: sc_angles[res_name][chi] = []
            sc_angles[res_name][chi].append(round(val, 2))
    
    return bb_angles, sc_angles


def discretized_distribution(angles, bins=np.arange(-180, 180, 10)) -> np.ndarray:

    '''
    10 degree for each bin
    https://en.wikipedia.org/wiki/Backbone-dependent_rotamer_library
    '''

    bin_counts = collections.Counter(np.searchsorted(bins, angles))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins) + 1)]
    bin_counts = np.array(bin_counts) / np.sum(bin_counts)

    return bin_counts


def _worker(inputs):
    pdb, selected_chains, selected_residues = inputs
    try:
        bb, sc = dihedrals(pdb, selected_chains, selected_residues)
    except Exception:
        return {}
    return {
        'backbone': bb,
        'sidechain': sc
    }


def dihedral_distribution(pdbs, all_selected_chains=None, all_selected_residues=None, num_cpus=None):
    angles_all = {}

    inputs = []
    for i, pdb in enumerate(pdbs):
        if all_selected_chains is None: selected_chains = None
        else: selected_chains = all_selected_chains[i]
        if all_selected_residues is None: selected_residues = None
        else: selected_residues = all_selected_residues[i]

        inputs.append((pdb, selected_chains, selected_residues))

    results = process_map(_worker, inputs, max_workers=num_cpus, chunksize=10)

    for res in results:
        for t in res:
            if t not in angles_all: angles_all[t] = {}
            for aa in res[t]:
                if aa not in angles_all[t]: angles_all[t][aa] = {}
                for angle in res[t][aa]:
                    if angle not in angles_all[t][aa]: angles_all[t][aa][angle] = []
                    angles_all[t][aa][angle].extend(res[t][aa][angle])

    # discretize
    for t in angles_all:
        for aa in angles_all[t]:
            for angle in angles_all[t][aa]:
                angles_all[t][aa][angle] = discretized_distribution(angles_all[t][aa][angle]).tolist()
    
    return angles_all


def _jsd_angle(model, reference):
    results = {}
    if type(model) == list:
        if len(model) == 0:
            return float('nan')
        return jensenshannon(reference, model)
    assert type(results) == dict
    for key in reference:
        ref = reference[key]
        results[key] = _jsd_angle(model.get(key, type(ref)()), ref)
    return results


def jsd_angle_profile(model, reference_type: str = 'peptide'):
    '''
        Args:
            reference_type: str, can be peptide or antibody
    '''
    if reference_type == 'peptide':
        ref_file = os.path.join(os.path.dirname(__file__), 'dihedral_data', 'pep.json')
    elif reference_type == 'antibody':
        ref_file = os.path.join(os.path.dirname(__file__), 'dihedral_data', 'ab.json')
    else:
        raise ValueError(f'Reference type {reference_type} not implemented')
    reference = json.load(open(ref_file, 'r'))

    jsd = _jsd_angle(model, reference)
    metrics = {}

    for t in jsd: # bb or sc
        for aa in jsd[t]:
            for angle in jsd[t][aa]:
                name = f'{t}_{aa}_{angle}'
                metrics[name] = jsd[t][aa][angle]

    overall_keys = ['backbone', 'sidechain', 'phi', 'psi', 'omg'] + [f'chi{i}' for i in range(1, 5)]
    overall_metrics = {}
    for key in overall_keys:
        vals = [metrics[name] for name in metrics if key in name and not np.isnan(metrics[name])]
        if len(vals) == 0: res = 0
        else: res = sum(vals) / len(vals)
        overall_metrics[key + '_overall'] = res
    
    metrics.update(overall_metrics)

    return metrics



if __name__ == '__main__':
    import os
    import sys
    import json

    # dist = dihedral_distribution([sys.argv[1]], [['G']])

    # print(jsd_angle_profile(dist))

    print(jsd_angle_profile(json.load(open(os.path.join(os.path.dirname(__file__), 'dihedral_data', 'pep.json'), 'r'))))