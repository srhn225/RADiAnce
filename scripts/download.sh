#!/bin/bash



PEPTIDE_DIR="./datasets/peptide"
MOLECULE_DIR="./datasets/molecule"
ANTIBODY_DIR="./datasets/antibody"



#######################################


#######################################
echo "Processing Peptide data..."





python -m scripts.data_process.peptide.pepbench --index ${PEPTIDE_DIR}/LNR/test.txt --out_dir ${PEPTIDE_DIR}/LNR/processed --remove_het
python -m scripts.data_process.peptide.pepbench --index ${PEPTIDE_DIR}/pepbench/all.txt --out_dir ${PEPTIDE_DIR}/pepbench/processed
python -m scripts.data_process.peptide.transform_index \
    --train_index ${PEPTIDE_DIR}/pepbench/train.txt \
    --valid_index ${PEPTIDE_DIR}/pepbench/valid.txt \
    --all_index_for_non_standard ${PEPTIDE_DIR}/pepbench/all.txt \
    --processed_dir ${PEPTIDE_DIR}/pepbench/processed/
python -m scripts.data_process.peptide.pepbench --index ${PEPTIDE_DIR}/ProtFrag/all.txt --out_dir ${PEPTIDE_DIR}/ProtFrag/processed


