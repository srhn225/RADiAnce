#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import yaml
import json
from data import create_dataset
from tqdm import tqdm
def main(args):


    config = yaml.safe_load(open(args.config, 'r'))
    


    all_items,_,_ = create_dataset(config['dataset'])
    result = {}
    for item in tqdm(all_items):

        item_id = item['name']

        result[item_id] = {
        'id':item['id'],
        'path':item['id']+'.pdb',
        'cdr':item['cdr'],
        'ref_seq':item["ref_seq"],
        "target_chain_ids":item["target_chain_ids"],
        "ligand_chain_ids":item["ligand_chain_ids"],
        }
    


    os.makedirs(args.save_dir, exist_ok=True)
    


    output_path = os.path.join(args.save_dir, 'dataset_properties.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Dataset properties saved to {output_path}")

def parse():
    parser = argparse.ArgumentParser(description='dataset_property_json')
    parser.add_argument('--config', required=True, help='path to the config file')
    parser.add_argument('--save_dir', required=True, help='directory to save the dataset properties')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
    # python dataset_property_json.py --config ./configs/json/all.yaml --save_dir ./interaction/data_index/