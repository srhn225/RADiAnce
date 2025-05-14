#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    Binding Site Recovery
    From https://github.com/Ced3-han/PepFlowww/blob/main/eval/geometry.py

    CA within 10A distance define the binding site
'''

from Bio.PDB import PDBParser, NeighborSearch


def get_bind_site(pdb,chain_id):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    peps = [atom for res in structure[chain_id] for atom in res if atom.get_name() == 'CA']
    recs = [atom for chain in structure if chain.get_id()!=chain_id for res in chain for atom in res if atom.get_name() == 'CA']
    # print(recs)
    search = NeighborSearch(recs)
    near_res = []
    for atom in peps:
        near_res += search.search(atom.get_coord(), 10.0, level='R')
    near_res = set([res.get_id()[1] for res in near_res])
    return near_res


def get_bind_ratio(pdb1, pdb2, chain_id1, chain_id2):
    near_res1,near_res2 = get_bind_site(pdb1,chain_id1),get_bind_site(pdb2,chain_id2)
    # print(near_res1)
    # print(near_res2)
    return len(near_res1.intersection(near_res2))/(len(near_res2)+1e-10) # last one is gt