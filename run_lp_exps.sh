#!/bin/bash
# run_lp_exps.sh

DATASET="wikipedia"

# Experiment 1: Default settings
python train_link_prediction_v2.py --dataset_name $DATASET --num_epochs 50 --lr 0.001 --dropout 0.1 --hidden_dim 64 --embedding_dim 64 --num_sampled_neighbors 5

python train_link_prediction_v2.py --dataset_name $DATASET --num_epochs 50 --lr 0.001 --dropout 0.1 --hidden_dim 64 --embedding_dim 64 --num_sampled_neighbors 10

python train_link_prediction_v2.py --dataset_name $DATASET --num_epochs 50 --lr 0.001 --dropout 0.1 --hidden_dim 64 --embedding_dim 64 --num_sampled_neighbors 20

echo "Link prediction experiments completed."
