#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple
import pickle

import models
from utils.config_utils import overwrite_values
from data import create_dataset, create_dataloader
from utils.random_seed import setup_seed
import pdb

def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = [(float(l.split(':')[0]), l.split(':')[1].strip()) for l in ls]
    return os.path.join(ckpt_dir, 'checkpoint', ckpts[0][1])

def extract_all_embeddings(model, data_loader, device='cuda') -> Tuple[List[np.ndarray], List[np.ndarray]]:
    model.eval()
    all_bind_sites, all_ligands = [], []
    
    with torch.no_grad():
        try:
            for batch in tqdm(data_loader, desc="Processing batches"):
                device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                try:
                    bind_emb, ligand_emb = model.get_contrastive_repr(**device_batch)
                    all_bind_sites.extend(bind_emb.cpu().numpy())
                    all_ligands.extend(ligand_emb.cpu().numpy())
                except Exception as e:
                    print(f"Error processing batch: {e}")
        except Exception as e:
            print(f"Error : {e}")
    return all_bind_sites, all_ligands

def compute_hit_rate(test_bind: np.ndarray, test_ligand: np.ndarray, 
                    train_ligand: np.ndarray, device='cuda', N_percent=0.05) -> float:


    all_ligand = np.concatenate([train_ligand, test_ligand], axis=0)
    


    test_bind_tensor = torch.tensor(test_bind, dtype=torch.float32, device=device)
    all_ligand_tensor = torch.tensor(all_ligand, dtype=torch.float32, device=device)



    correct_indices = torch.arange(len(test_ligand), device=device) + len(train_ligand)
    k = int(N_percent * len(all_ligand))
    


    hit_count = 0
    batch_size = 256
    for i in tqdm(range(0, len(test_bind), batch_size), desc="Calculating similarities"):
        batch_bind = test_bind_tensor[i:i+batch_size]
        sim_matrix = torch.mm(batch_bind, all_ligand_tensor.T)
        


        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
        current_indices = correct_indices[i:i+batch_size].unsqueeze(1)
        ranks = (sorted_indices == current_indices).int().argmax(dim=1)
        
        hit_count += (ranks < k).sum().item()

    return hit_count / len(test_bind)

def save_cache(cache_path, data):

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_cache(cache_path):

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def main(args):



    config = yaml.safe_load(open(args.config, 'r'))
    setup_seed(12)



    device = torch.device('cuda' if args.gpu >=0 else 'cpu')
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    model = torch.load(b_ckpt, map_location='cpu').to(device).eval()
    


    train_set, _, test_set = create_dataset(config['dataset'])
    train_loader = create_dataloader(train_set, config['dataloader'])
    test_loader = create_dataloader(test_set, config['dataloader'])
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")



    cache_dir = os.path.join(os.path.dirname(args.config), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cache_dir, 'pep_train_cache.pkl')
    test_cache_path = os.path.join(cache_dir, 'pep_test_cache.pkl')
    


    # if os.path.exists(test_cache_path):
    #     print("Loading test features from cache...")
    #     test_bind, test_ligand = load_cache(test_cache_path)
    # else:
    #     print("Extracting test features...")
    test_bind, test_ligand = extract_all_embeddings(model, test_loader, device)
        # save_cache(test_cache_path, (test_bind, test_ligand))
        


    # if os.path.exists(train_cache_path):
    #     print("Loading training features from cache...")
    #     train_bind, train_ligand = load_cache(train_cache_path)
    # else:
    #     print("Extracting training features...")
    train_bind, train_ligand = extract_all_embeddings(model, train_loader, device)
        # save_cache(train_cache_path, (train_bind, train_ligand))





    print("\nCalculating hit rates:")
    for n_percent in [0.001,0.005,0.01, 0.05, 0.1,0.3]:
        hr = compute_hit_rate(
            test_bind=np.array(test_bind),
            test_ligand=np.array(test_ligand),
            train_ligand=np.array(train_ligand),
            device=device,
            N_percent=n_percent
        )
        print(f"Top-{n_percent*100:.1f}%: {hr*100:.2f}%")

def parse():

    parser = argparse.ArgumentParser(description='RAG Validation')
    parser.add_argument('--config', required=True, help='Path to the config file')
    parser.add_argument('--ckpt', required=True, help= 'Path to the checkpoint file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
    # CUDA_VISIBLE_DEVICES=1 python rag_valid.py --config ./configs/contrastive/valid/test_ab.yaml   --ckpt ./ckpts/onlyab.ckpt --gpu 0
