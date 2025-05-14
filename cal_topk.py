#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import yaml
import torch
import json
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader
import pdb
import models
from utils.config_utils import overwrite_values
from data import create_dataset, create_dataloader
from utils.random_seed import setup_seed
from data.mmap_dataset import create_mmap
def get_best_ckpt(ckpt_dir):
    

    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = [(float(l.split(':')[0]), l.split(':')[1].strip()) for l in ls]
    return os.path.join(ckpt_dir, 'checkpoint', ckpts[0][1])

def extract_ligand_features(model, dataloader, device) -> Tuple[List[np.ndarray], List[str]]:
    
    model.eval()
    all_features = []
    all_bind=[]
    all_ids = []
    
    with torch.no_grad():
        try:
            for batch in tqdm(dataloader, desc="Extracting ligand features"):
                device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                bind_emb, ligand_emb = model.get_contrastive_repr(**device_batch)
                
                all_features.extend(ligand_emb.cpu().numpy())
                all_bind.extend(bind_emb.cpu().numpy())
                all_ids.extend(batch['name'])
        except Exception as e:
            pass           
    return all_features,all_bind, all_ids

def extract_bind_features(model, dataloader, device) -> Tuple[List[np.ndarray], List[str]]:
    
    model.eval()
    all_features = []
    all_ids = []
    
    with torch.no_grad():
        try:
            for batch in tqdm(dataloader, desc="Extracting bind features"):
                device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                bind_emb, _ = model.get_contrastive_repr(**device_batch)
                all_features.extend(bind_emb.cpu().numpy())
                all_ids.extend(batch['name'])
        except Exception as e:
            pass
    return all_features, all_ids

def save_ligand_features(ligand_features: List[np.ndarray], ligand_ids: List[str], cache_dir: str):
    ligand_feature_dict = {ligand_id: feature for ligand_id, feature in zip(ligand_ids, ligand_features)}
    


    output_path = os.path.join(cache_dir, 'ligand_features.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(ligand_feature_dict, f)
    print(f"Ligand features saved to {output_path}")
def save_features_as_mmap(features: List[np.ndarray], ids: List[str], mmap_dir: str, desc: str = "Saving features"):
    iterator = (
        (str(ids[i]), features[i].tolist(), {"dim": features[i].shape[0]}, i)
        for i in range(len(features))
    )
    


    create_mmap(
        iterator=iterator,
        out_dir=mmap_dir,
        total_len=len(features),
        abbr_desc_len=10,
    )
    print(f"{desc} saved to {mmap_dir}")
def main(args):


    config = yaml.safe_load(open(args.config, 'r'))
    setup_seed(12)


    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if args.gpu >=0 else 'cpu')
    cache_dir = os.path.join(os.path.dirname(args.save_dir), 'match_cache_protein')
    os.makedirs(cache_dir, exist_ok=True)


    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    model = torch.load(b_ckpt, map_location='cpu').to(device).eval()



    train_set, valid_set, test_set = create_dataset(config['dataset'])
    


    train_loader = create_dataloader(train_set, config['dataloader'])
    train_ligand_features,train_features, train_ligand_ids = extract_ligand_features(model, train_loader, device)

    ligand_mmap_dir = os.path.join(cache_dir, "ligand_mmap")
    save_features_as_mmap(
        features=train_ligand_features,
        ids=train_ligand_ids,
        mmap_dir=ligand_mmap_dir,
        desc="Ligand features"
    )


    print("Extracting all bind features...")


    train_bind_loader = create_dataloader(train_set, config['dataloader'])
    valid_bind_loader = create_dataloader(valid_set, config['dataloader'])
    test_bind_loader = create_dataloader(test_set, config['dataloader'])
    
    train_ids=train_ligand_ids


    valid_features, valid_ids = extract_bind_features(model, valid_bind_loader, device)
    test_features, test_ids = extract_bind_features(model, test_bind_loader, device)
    


    all_bind_features = train_features + valid_features + test_features
    all_bind_ids = train_ids + valid_ids + test_ids






    print("Computing top50 matches...")
    


    ligand_tensor = torch.tensor(train_ligand_features, dtype=torch.float32, device=device)
    bind_tensor = torch.tensor(all_bind_features, dtype=torch.float32, device=device)


    batch_size = 512
    top_k = 50
    results = {}

    for i in tqdm(range(0, len(bind_tensor), batch_size), desc="Processing bind features"):


        batch_bind = bind_tensor[i:i+batch_size]
        batch_ids = all_bind_ids[i:i+batch_size]
        


        sim_matrix = torch.mm(batch_bind, ligand_tensor.T)  # (batch_size, num_ligands)
        


        _, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)  # (batch_size, top_k)
        


        for j in range(topk_indices.size(0)):
            bind_id = batch_ids[j]
            ligand_indices = topk_indices[j].cpu().numpy()
            matched_ids = [train_ligand_ids[idx] for idx in ligand_indices]
            results[bind_id] = matched_ids



    output_path = os.path.join(cache_dir, 'top50_matches.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Top50 matches saved to {output_path}")


def parse():
    parser = argparse.ArgumentParser(description='Compute Top-50 ligand matches for the binding site')
    parser.add_argument('--config', required=True, help='Path to the configuration file')
    parser.add_argument('--ckpt', required=True, help='Path to the model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use; -1 means using CPU')
    parser.add_argument('--save_dir', type=str, default="./features/testset", help='Directory to save results')

    
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
    # python -m cal_topk --config ./configs/contrastive/calculate/topk_protein.yaml --ckpt ./ckpts/epoch249_step238500.ckpt --gpu 0 --save_dir ./features_all
