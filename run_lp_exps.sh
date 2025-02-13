#!/bin/bash
# run_lp_exps.sh

DATASET="wikipedia"

# Experiment 1: Default settings
python train_link_prediction_v2.py --dataset_name $DATASET --epochs 50 --lr 0.001 --in_dim 16 --hidden_dim 64 --embedding_dim 64 --pos_dim 8 --time_dim 8

# Experiment 2: Higher learning rate
python train_link_prediction_v2.py --dataset_name $DATASET --epochs 50 --lr 0.005 --in_dim 16 --hidden_dim 64 --embedding_dim 64 --pos_dim 8 --time_dim 8

# Experiment 3: More transformer layers and larger hidden dimension
python train_link_prediction_v2.py --dataset_name $DATASET --epochs 100 --lr 0.001 --in_dim 16 --hidden_dim 128 --embedding_dim 128 --pos_dim 8 --time_dim 8

echo "Link prediction experiments completed."
