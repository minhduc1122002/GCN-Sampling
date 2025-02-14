# ASH-DGT Dynamic Graph Learning

This repo implements the ASH-DGT model for dynamic graph learning tasks, including node classification and link prediction.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn

## Datasets
You can find all dataset are used in this [link](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Download them and store under **./data** folder.

## Training and experiments
Feel free to explore and fine-tuning the model hyperparamters (see description in the main) and custom to search them in your own setting script (eg. bash script).
### Node classification
```python 
python train_node_classification_v2.py --dataset_name mooc --epochs 50 --lr 0.001 --in_dim 16 --hidden_dim 16 --pos_dim 8 --time_dim 8 --batch_size 128
```

### Link prediction
```python
python train_link_prediction_v2.py --dataset_name uci --num_epochs 50 --attn_heads 8 --lr 0.001 --dropout 0.1 --hidden_dim 16 --embedding_dim 16 --coarsen_method ve --seed 1024
```

Will update the readme soon!





