#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    From https://github.com/luost26/diffab/blob/main/diffab/tools/relax/pyrosetta_relaxer.py
'''
import time
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
# for fast relax
from pyrosetta.rosetta import protocols
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
from pyrosetta.rosetta.core.scoring import ScoreType


pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
    # below are from https://github.com/nrbennet/dl_binder_design/blob/main/mpnn_fr/dl_interface_design.py
    # '-beta_nov16',
    '-use_terminal_residues', 'true',
    '-in:file:silent_struct_type', 'binary'
]))


def current_milli_time():
    return round(time.time() * 1000)


def get_scorefxn(scorefxn_name:str):
    """
    Gets the scorefxn with appropriate corrections.
    Taken from: https://gist.github.com/matteoferla/b33585f3aeab58b8424581279e032550
    """
    import pyrosetta

    corrections = {
        'beta_july15': False,
        'beta_nov16': False,
        'gen_potential': False,
        'restore_talaris_behavior': False,
    }
    if 'beta_july15' in scorefxn_name or 'beta_nov15' in scorefxn_name:
        # beta_july15 is ref2015
        corrections['beta_july15'] = True
    elif 'beta_nov16' in scorefxn_name:
        corrections['beta_nov16'] = True
    elif 'genpot' in scorefxn_name:
        corrections['gen_potential'] = True
        pyrosetta.rosetta.basic.options.set_boolean_option('corrections:beta_july15', True)
    elif 'talaris' in scorefxn_name:  #2013 and 2014
        corrections['restore_talaris_behavior'] = True
    else:
        pass
    for corr, value in corrections.items():
        pyrosetta.rosetta.basic.options.set_boolean_option(f'corrections:{corr}', value)
    return pyrosetta.create_score_function(scorefxn_name)


class RelaxRegion(object):
    
    def __init__(self, scorefxn='ref2015', max_iter=1000, subset='nbrs', move_bb=True):
        super().__init__()

        self.scorefxn = get_scorefxn(scorefxn)
        self.fast_relax = FastRelax()
        self.fast_relax.set_scorefxn(self.scorefxn)
        self.fast_relax.max_iter(max_iter)

        assert subset in ('all', 'target', 'nbrs')
        self.subset = subset
        self.move_bb = move_bb

    def __call__(self, pdb_path, ligand_chains, flexible_residue_first=None, flexible_residue_last=None): # flexible_residue_first, flexible_residue_last):
        pose = pyrosetta.pose_from_pdb(pdb_path)
        start_t = current_milli_time()
        original_pose = pose.clone()

        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        tf.push_back(operation.RestrictToRepacking())   # Only allow residues to repack. No design at any position.

        # Create selector for the region to be relaxed
        # Turn off design and repacking on irrelevant positions
        # if flexible_residue_first[-1] == ' ': 
        #     flexible_residue_first = flexible_residue_first[:-1]
        # if flexible_residue_last[-1] == ' ':  
        #     flexible_residue_last  = flexible_residue_last[:-1]
        if self.subset != 'all':
            if flexible_residue_first is None or flexible_residue_last is None:
                chain_selectors = [selections.ChainSelector(chain) for chain in ligand_chains]
                if len(chain_selectors) == 1:
                    gen_selector = chain_selectors[0]
                else:
                    gen_selector = selections.OrResidueSelector(chain_selectors[0], chain_selectors[1])
                    for selector in chain_selectors[2:]:
                        gen_selector = selections.OrResidueSelector(gen_selector, selector)
            else:
                gen_selector = selections.ResidueIndexSelector()
                gen_selector.set_index_range(
                    pose.pdb_info().pdb2pose(*flexible_residue_first), 
                    pose.pdb_info().pdb2pose(*flexible_residue_last), 
                )
            nbr_selector = selections.NeighborhoodResidueSelector()
            nbr_selector.set_focus_selector(gen_selector)
            nbr_selector.set_include_focus_in_subset(True)

            if self.subset == 'nbrs':
                subset_selector = nbr_selector
            elif self.subset == 'target':
                subset_selector = gen_selector

            prevent_repacking_rlt = operation.PreventRepackingRLT()
            prevent_subset_repacking = operation.OperateOnResidueSubset(
                prevent_repacking_rlt, 
                subset_selector,
                flip_subset=True,
            )
            tf.push_back(prevent_subset_repacking)

        scorefxn = self.scorefxn
        fr = self.fast_relax

        pose = original_pose.clone()

        mmf = MoveMapFactory()
        if self.move_bb: 
            mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
        mmf.add_chi_action(move_map_action.mm_enable, subset_selector)
        mm  = mmf.create_movemap_from_pose(pose)

        fr.set_movemap(mm)
        fr.set_task_factory(tf)
        fr.apply(pose)

        e_before = scorefxn(original_pose)
        e_relax  = scorefxn(pose) 
        # print('\n\n[Finished in %.2f secs]' % ((current_milli_time() - start_t) / 1000))
        # print(' > Energy (before):    %.4f' % scorefxn(original_pose))
        # print(' > Energy (optimized): %.4f' % scorefxn(pose))
        return pose, e_before, e_relax


def pyrosetta_fastrelax(pdb_path, out_path, ligand_chains, flexible_residue_first=None, flexible_residue_last=None):
    minimizer = RelaxRegion()
    pose_min, _, _ = minimizer(
        pdb_path=pdb_path,
        ligand_chains=ligand_chains,
        flexible_residue_first=flexible_residue_first,
        flexible_residue_last=flexible_residue_last
    )
    pose_min.dump_pdb(out_path)


def pyrosetta_interface_energy(pdb_path, receptor_chains, ligand_chains, return_dict=False, relax=False, relax_opt={}, relax_save_pdb=None):
    if relax:
        minimizer = RelaxRegion()
        pose, _, _ = minimizer(
            pdb_path=pdb_path,
            ligand_chains=ligand_chains,
            **relax_opt
        )
        if relax_save_pdb is not None: pose.dump_pdb(relax_save_pdb)
    else:
        pose = pyrosetta.pose_from_pdb(pdb_path)
    interface = ''.join(ligand_chains) + '_' + ''.join(receptor_chains)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    if return_dict:
        return pose.scores
    return pose.scores['dG_separated']


if __name__ == '__main__':
    import sys
    import json
    # example args
    # args = {
    #     'pdb_path': '/path/to/pdb',
    #     'receptor_chains': ['A'],
    #     'ligand_chains': ['B'],
    #     'relax': False
    # }
    args = json.loads(sys.argv[1])
    dG = pyrosetta_interface_energy(**args)
    print(dG)