#!/usr/bin/python
# -*- coding:utf-8 -*-
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces


def merge_chains(model, chains_to_merge):
    for chain in chains_to_merge[1:]:
        for i, res in enumerate(model[chain]):
            res.id = (chain, res.id[1], res.id[2])
            model[chains_to_merge[0]].add(res)
        model.detach_child(chain)
    model[chains_to_merge[0]].id = "".join(chains_to_merge)
    return model


def dockq(model_pdb, native_pdb, model_rec_chains, model_lig_chains, native_rec_chains=None, native_lig_chains=None):
    if native_rec_chains is None: native_rec_chains = model_rec_chains
    if native_lig_chains is None: native_lig_chains = model_lig_chains
    model, native = load_PDB(model_pdb), load_PDB(native_pdb)

    if len(model_rec_chains) > 1: model = merge_chains(model, model_rec_chains)
    if len(model_lig_chains) > 1: model = merge_chains(model, model_lig_chains)

    if len(native_rec_chains) > 1: native = merge_chains(native, native_rec_chains)
    if len(native_lig_chains) > 1: native = merge_chains(native, native_lig_chains)

    # native:model chain map dictionary for two interfaces
    chain_map = {
        ''.join(native_rec_chains): ''.join(model_rec_chains),
        ''.join(native_lig_chains): ''.join(model_lig_chains)
    }
    # returns a dictionary containing the results and the total DockQ score
    res = run_on_all_native_interfaces(model, native, chain_map=chain_map, no_align=True)

    return res[1]


if __name__ == '__main__':
    import sys
    print(dockq(sys.argv[1], sys.argv[2], [sys.argv[3]], [sys.argv[4]]))