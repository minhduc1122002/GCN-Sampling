#!/bin/bash
# run_nc_exps.sh

DATASET="wikipedia"

# Experiment 1: Default hyperparameters
python train_node_classification_v2.py --dataset_name $DATASET --epochs 50 --lr 0.001 --in_dim 16 --hidden_dim 64 --pos_dim 8 --time_dim 8

# Experiment 2: Higher learning rate
python train_node_classification_v2.py --dataset_name $DATASET --epochs 50 --lr 0.005 --in_dim 16 --hidden_dim 64 --pos_dim 8 --time_dim 8

# Experiment 3: More epochs and larger hidden dimension
python train_node_classification_v2.py --dataset_name $DATASET --epochs 100 --lr 0.001 --in_dim 16 --hidden_dim 128 --pos_dim 8 --time_dim 8

# Experiment 4: Using validation split
python train_node_classification_v2.py --dataset_name $DATASET --use_validation --epochs 50 --lr 0.001 --in_dim 16 --hidden_dim 64 --pos_dim 8 --time_dim 8

echo "Node classification experiments completed."
