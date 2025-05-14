import os
import logging
import ray
from plip.structure.preparation import PDBComplex
from plip.basic import config
logging.getLogger('plip').setLevel(logging.ERROR)
from Bio.Align import substitution_matrices, PairwiseAligner
from tqdm import tqdm
import json
from collections import defaultdict
from collections import Counter
import pdb
import re
def load_json(json_path: str) -> dict:

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data
def invert_interactions(interactions_dict):
    inverted_dict = {}
    for key, values in interactions_dict.items():
        for target in values:
            new_key = (target[0], target[1])
            new_value = (key[0], key[1], target[2])
            
            if new_key not in inverted_dict:
                inverted_dict[new_key] = []
            
            inverted_dict[new_key].append(new_value)
    
    return inverted_dict
class Chothia:
    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)
    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

def filter_cdr_interactions(all_interactions, selected_cdrs):
    if selected_cdrs==[None]:
        return invert_interactions(all_interactions)
    cdr_mapping = {
        'HCDR1': ('H', Chothia.H1),
        'HCDR2': ('H', Chothia.H2),
        'HCDR3': ('H', Chothia.H3),
        'LCDR1': ('L', Chothia.L1),
        'LCDR2': ('L', Chothia.L2),
        'LCDR3': ('L', Chothia.L3)
    }
    invalid_cdrs = [cdr for cdr in selected_cdrs if cdr not in cdr_mapping]
    if invalid_cdrs:
        raise ValueError(f"Invalid CDR names: {invalid_cdrs}")
    
    filtered = {}
    for (reschain, resnr), interactions in all_interactions.items():
        for cdr_name in selected_cdrs:
            chain_type, (start, end) = cdr_mapping[cdr_name]
            if start <= resnr <= end:
                filtered[(reschain, resnr)] = interactions
                break
    return filtered

def plip_interactions(pdb_path, tgt_chains, lig_chains, selected_interactions=None):
    if selected_interactions is None:
        selected_interactions = [
            'saltbridge_lneg', 'saltbridge_pneg',
            'hbonds_ldon', 'hbonds_pdon',
            'pication_laro', 'pication_paro',
            'pistacking', 'hydrophobic_contacts'
        ]
    
    config.PEPTIDES = lig_chains
    try:
        cplx = PDBComplex()
        cplx.load_pdb(pdb_path)
        cplx.analyze()
    except Exception as e:
        logging.error(f"PLIP analysis failed: {pdb_path} - {str(e)}")
        return {}

    all_interactions = {}
    for key in cplx.interaction_sets:
        _, chain, _ = key.split(':')
        if chain not in lig_chains:
            continue
        for itype in selected_interactions:
            interactions = getattr(cplx.interaction_sets[key], itype, [])
            for i in interactions:
                if i.reschain not in tgt_chains:
                    continue
                _id = (i.reschain, i.resnr)
                if _id not in all_interactions:
                    all_interactions[_id] = []
                all_interactions[_id].append((i.reschain_l, i.resnr_l, itype))
    
    return all_interactions

def interaction_recovery_count_cdr(ref_pdb, gen_pdb, ref_tgt_chains, ref_lig_chains, 
                                  gen_tgt_chains, gen_lig_chains, selected_interactions=None,
                                  selected_cdrs_ref=[], selected_cdrs_gen=[]):
    if selected_interactions is None:
        selected_interactions = [
            'saltbridge_lneg', 'saltbridge_pneg',
            'hbonds_ldon', 'hbonds_pdon',
            'pication_laro', 'pication_paro',
            'pistacking', 'hydrophobic_contacts'
        ]
    
    ref_interactions = filter_cdr_interactions(
        plip_interactions(ref_pdb, ref_tgt_chains, ref_lig_chains, selected_interactions),
        selected_cdrs_ref
    )
    gen_interactions = filter_cdr_interactions(
        plip_interactions(gen_pdb, gen_tgt_chains, gen_lig_chains, selected_interactions),
        selected_cdrs_gen
    )
    
    ref_counts = Counter(itype for res in ref_interactions.values() for _, _, itype in res)
    gen_counts = Counter(itype for res in gen_interactions.values() for _, _, itype in res)
    
    overlap = {}
    total_hit = 0
    for itype, count in ref_counts.items():
        gen_count = gen_counts.get(itype, 0)
        overlap[itype] = min(count, gen_count)
        total_hit += overlap[itype]
    
    total_ref = sum(ref_counts.values())
    recovery_rate = total_hit / total_ref if total_ref > 0 else float('nan')
    
    return recovery_rate, overlap,gen_interactions
def similarity_func(sequence_A, sequence_B, **kwargs):
    """
    Performs a global pairwise alignment between two sequences
    using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
    as implemented in Biopython. Returns the alignment, the sequence
    identity and the residue mapping between both original sequences.
    """
    
    sub_matrice = substitution_matrices.load('BLOSUM62')
    aligner = PairwiseAligner()
    aligner.substitution_matrix = sub_matrice
    alns = aligner.align(sequence_A, sequence_B)
    
    best_aln = alns[0]

    aligned_A, aligned_B = best_aln
    
    # Calculate sequence identity based on aligned residues
    matches = sum(1 for a, b in zip(aligned_A, aligned_B) if a == b)
    seq_id = matches / len(aligned_A)
    
    return seq_id
@ray.remote
def process_key(key, data_key, pdb_dir, n, index_dict: dict):
    logging.getLogger('plip').setLevel(logging.ERROR)
    results_for_key = []
    
    try:


        if key not in index_dict:
            raise KeyError(f"Key {key} not found in index_dict")
        
        item = index_dict[key]
        if "CDR" in key:
            pdb_id_ref=key.split("_")[0]#antibody
            ref_tgt_chains = item["ligand_chain_ids"]
            ref_lig_chains = item["target_chain_ids"]            
            
        else:
            pdb_id_ref = item['id']
            ref_tgt_chains = item["target_chain_ids"]
            ref_lig_chains = item["ligand_chain_ids"]
        cdr = item['cdr']
        ref_seq=re.sub(r'f\d+', '', item['ref_seq'])


        for gen_id in data_key[:n]:
            try:


                if gen_id not in index_dict:
                    raise KeyError(f"Generated ID {gen_id} not found in index_dict")
                
                gen_item = index_dict[gen_id]
                if "CDR" in gen_id:
                    pdb_id_gen=gen_id.split("_")[0]#antibody
                    gen_tgt_chains = gen_item["ligand_chain_ids"]    
                    gen_lig_chains = gen_item["target_chain_ids"]                
                else:
                    pdb_id_gen = gen_item['id']#peptide need to reverse due to plip
                    gen_tgt_chains = gen_item["target_chain_ids"]
                    gen_lig_chains = gen_item["ligand_chain_ids"]
                gen_cdr = gen_item['cdr']
                gen_seq=re.sub(r'f\d+', '',gen_item['ref_seq'])



                ref_pdb = os.path.join(pdb_dir, f"{pdb_id_ref}.pdb")
                gen_pdb = os.path.join(pdb_dir, f"{pdb_id_gen}.pdb")



                if not os.path.exists(ref_pdb):
                    raise FileNotFoundError(f"Missing reference PDB: {ref_pdb}")
                if not os.path.exists(gen_pdb):
                    raise FileNotFoundError(f"Missing generated PDB: {gen_pdb}")



                recovery_rate, overlap, gen_interactions = interaction_recovery_count_cdr(
                    ref_pdb=ref_pdb,
                    gen_pdb=gen_pdb,
                    ref_tgt_chains=ref_tgt_chains,
                    ref_lig_chains=ref_lig_chains,
                    gen_tgt_chains=gen_tgt_chains,
                    gen_lig_chains=gen_lig_chains,
                    selected_cdrs_ref=[cdr],
                    selected_cdrs_gen=[gen_cdr]
                )
                seq_id=similarity_func(ref_seq,gen_seq)



                results_for_key.append({
                    "generated_id": gen_id,
                    "recovery_rate": recovery_rate,
                    "overlap": overlap,
                    "seq_id":seq_id,
                    "gen_interactions": gen_interactions,
                })
            except Exception as e:
                print(f"Error processing {gen_id}: {str(e)}")

        print(key + str(results_for_key))
    except Exception as e:
        print(f"Error processing key {key}: {str(e)}")

    
    return key, results_for_key
def calculate_statistics(results):

    stats = {}
    
    for key, res_list in results.items():
        recovery_rates = [item['recovery_rate'] for item in res_list if 'recovery_rate' in item]
        seq_ids = [item['seq_id'] for item in res_list if 'seq_id' in item]
        
        if not recovery_rates or not seq_ids:
            continue
        
        stats[key] = {
            'recovery_rate': {
                'mean': sum(recovery_rates) / len(recovery_rates),
                'max': max(recovery_rates),
                'min': min(recovery_rates)
            },
            'seq_id': {
                'mean': sum(seq_ids) / len(seq_ids),
                'max': max(seq_ids),
                'min': min(seq_ids)
            }
        }
    
    return stats
def main(json_path, pdb_dir, output_path, n,index_path):
    ray.init(num_cpus=32)
    
    with open(json_path) as f:
        data = json.load(f)
    with open(index_path) as f:
        index = json.load(f)
    futures = [process_key.remote(k, v, pdb_dir, n,index) for k, v in data.items()]
    results = defaultdict(list)

    with tqdm(total=len(futures)) as pbar:
        while futures:
            done, futures = ray.wait(futures, timeout=1)
            for future in done:
                try:
                    key, res = ray.get(future)
                    results[key] = res
                except:
                    pass
                pbar.update(1)

    import pickle

    folder_path = os.path.dirname(output_path)
    os.makedirs(folder_path, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)   
    
    ray.shutdown()
    print(f"Results saved to {output_path}")


    stats = calculate_statistics(results)
    


    stats_output_path = output_path+'_stats.json'
    with open(stats_output_path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topn", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, default="pdb")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("-n", type=int, required=True)
    args = parser.parse_args()
    
    main(args.topn, args.pdb_dir, args.output, args.n,args.index)
    
    # python -m  interaction.interaction_all --topn ./features/interaction_final/match_cache_protein/top50_matches.json --pdb_dir ./datasets/all_pdbs/ --output ./interaction/final/recovery_final --index ./interaction/data_index/dataset_properties.json -n 10 
