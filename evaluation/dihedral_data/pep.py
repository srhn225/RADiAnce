#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from data.bioparse.const import aas
from evaluation.dihedrals import dihedrals, discretized_distribution


LNR_dir, pepbench_dir = sys.argv[1:]


bb_all, sc_all = {}, {}


def add_data(src, overall):
    for aa in src:
        if aa not in overall: overall[aa] = {}
        for angle in src[aa]:
            if angle not in overall[aa]: overall[aa][angle] = []
            overall[aa][angle].extend(src[aa][angle])
    return overall


def get_data(inputs):
    line, root_dir = inputs
    line = line.split('\t')
    _id, pep_chain = line[0], line[2]
    pdb_path = os.path.join(root_dir, 'pdbs', _id + '.pdb')
    bb, sc = dihedrals(pdb_path, selected_chains=[pep_chain])

    return (bb, sc)

    bb_all = add_data(bb, bb_all)
    sc_all = add_data(sc, sc_all)


with open(os.path.join(LNR_dir, 'test.txt'), 'r') as fin:
    lines = fin.read().strip().split('\n')


# for line in tqdm(lines):
#     line = line.split('\t')
#     _id, pep_chain = line[0], line[2]
#     pdb_path = os.path.join(LNR_dir, 'pdbs', _id + '.pdb')
#     bb, sc = dihedrals(pdb_path, selected_chains=[pep_chain])
# 
#     bb_all = add_data(bb, bb_all)
#     sc_all = add_data(sc, sc_all)

res = process_map(get_data, [(line, LNR_dir) for line in lines], max_workers=16, chunksize=10)

for bb, sc in res:
    bb_all = add_data(bb, bb_all)
    sc_all = add_data(sc, sc_all)

with open(os.path.join(pepbench_dir, 'all.txt'), 'r') as fin:
    lines = fin.read().strip().split('\n')

res = process_map(get_data, [(line, pepbench_dir) for line in lines], max_workers=16, chunksize=10)

for bb, sc in res:
    bb_all = add_data(bb, bb_all)
    sc_all = add_data(sc, sc_all)


def discretize(overall):
    dis = {}
    for _, abrv in aas:
        dis[abrv] = {}
        for angle in sorted(list(overall[abrv].keys())):
            dis[abrv][angle] = discretized_distribution(overall[abrv][angle]).tolist()
    return dis

# discretize
out_path = os.path.join(os.path.dirname(__file__), 'pep.json')
with open(out_path, 'w') as fout:
    json.dump({
        'backbone': discretize(bb_all),
        'sidechain': discretize(sc_all)
    }, fout, indent=2)