#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from data.bioparse.const import aas
from evaluation.dihedrals import dihedrals, discretized_distribution
from scripts.data_process.antibody.sabdab import Chothia


ab_dir = sys.argv[1]


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
    line = line.split('_')
    _id, hchain, lchain = line[0], line[2], line[3]
    pdb_path = os.path.join(root_dir, '..', 'all_structures', 'chothia', _id + '.pdb')

    selected_residues = []
    if hchain != '':
        for l, r in [Chothia.H1, Chothia.H2, Chothia.H3]:
            for i in range(l, r + 1):
                for code in ['', 'A', 'B', 'C', 'D', 'E']: # sufficient to cover all possible insert code
                    selected_residues.append((hchain, (i, code)))
    if lchain != '':
        for l, r in [Chothia.L1, Chothia.L2, Chothia.L3]:
            for i in range(l, r + 1):
                for code in ['', 'A', 'B', 'C', 'D', 'E']: # sufficient to cover all possible insert code
                    selected_residues.append((hchain, (i, code)))


    bb, sc = dihedrals(pdb_path, selected_residues=selected_residues)

    return (bb, sc)


lines = []
for name in ['train', 'valid', 'test']:
    with open(os.path.join(ab_dir, name + '.txt'), 'r') as fin:
        lines.extend(fin.read().strip().split('\n'))


res = process_map(get_data, [(line, ab_dir) for line in lines], max_workers=16, chunksize=10)

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
out_path = os.path.join(os.path.dirname(__file__), 'ab.json')
with open(out_path, 'w') as fout:
    json.dump({
        'backbone': discretize(bb_all),
        'sidechain': discretize(sc_all)
    }, fout, indent=2)