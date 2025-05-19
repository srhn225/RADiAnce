# RADiAnce
## Data Processing

**Suppose all the datasets are downloaded below `/path/to/data/`.**

### Peptide

All data are saved under `/path/to/data/peptide`. We set environment variable `export PREFIX=/path/to/data/peptide`.

1. LNR & PepBench & ProtFrag

Download:

```bash
# LNR
wget https://zenodo.org/records/13373108/files/LNR.tar.gz?download=1 -O ${PREFIX}/LNR.tar.gz
tar zxvf ${PREFIX}/LNR.tar.gz -C $PREFIX
# PepBench
wget https://zenodo.org/records/13373108/files/train_valid.tar.gz?download=1 -O ${PREFIX}/pepbench.tar.gz
tar zxvf $PREFIX/pepbench.tar.gz -C $PREFIX
mv ${PREFIX}/train_valid ${PREFIX}/pepbench
# ProtFrag
wget https://zenodo.org/records/13373108/files/ProtFrag.tar.gz?download=1 -O ${PREFIX}/ProtFrag.tar.gz
tar zxvf $PREFIX/ProtFrag.tar.gz -C $PREFIX
```

Processing:

```bash
# for 6mcl.pdb, there is unknown atoms in the pdb named X, which you need to manually remove
python -m scripts.data_process.peptide.pepbench --index ${PREFIX}/LNR/test.txt --out_dir ${PREFIX}/LNR/processed --remove_het
python -m scripts.data_process.peptide.pepbench --index ${PREFIX}/pepbench/all.txt --out_dir ${PREFIX}/pepbench/processed
python -m scripts.data_process.peptide.transform_index --train_index ${PREFIX}/pepbench/train.txt --valid_index ${PREFIX}/pepbench/valid.txt --all_index_for_non_standard ${PREFIX}/pepbench/all.txt --processed_dir ${PREFIX}/pepbench/processed/
python -m scripts.data_process.peptide.pepbench --index ${PREFIX}/ProtFrag/all.txt --out_dir ${PREFIX}/ProtFrag/processed
```



### Antibody

All data are saved under `/path/to/data/antibody`. We set environment variable `export PREFIX=/path/to/data/antibody`.

1. SAbDab (Downloaded at Sep 24th)

```bash
mkdir ${PREFIX}/SAbDab
wget https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/ -O ${PREFIX}/all_structures.zip
wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/ -O ${PREFIX}/SAbDab/summary.csv
```

Processing:

```bash
# process
python -m scripts.data_process.antibody.sabdab --index ${PREFIX}/SAbDab/summary.csv --out_dir ${PREFIX}/SAbDab/processed
# split
python -m scripts.data_process.antibody.split --index ${PREFIX}/SAbDab/index.txt --rabd_summary ${PREFIX}/RAbD/rabd_summary.jsonl

```

## Prepare RAG Database
```bash
python -m cal_topk --config ./configs/contrastive/calculate/topk_protein.yaml --ckpt path/to/contrastive_vae --gpu 0 --save_dir your_path
```
Important: You need to change the rag database path in the train configs(e.g.`./configs/LDM-rag/train_pt.yaml`) to your own path.

## Training

Training of the full RADiAnce requires 8 GPUs with 80G memmory each.

```bash
GPU=0,1,2,3,4,5,6,7 bash ./scripts/train.sh ./configs/contrastive/train_pt.yaml # train contrastive autoencoder
GPU=0,1,2,3,4,5,6,7 bash ./scripts/train.sh ./configs/LDM-rag/train_pt.yaml # train rag diffusion model, you should also modify the path of the rag database and checkpoints in the config file

```
## Generation on Test Sets
```bash
# peptide
python generate.py --config configs/test/test_pep.yaml --ckpt /path/to/checkpoint.ckpt --gpu 0 --save_dir ./results/pep
# antibody
python generate.py --config configs/test_rag/test_abxx.yaml --ckpt /path/to/checkpoint.ckpt --gpu 0 --save_dir ./results/abxx
```

