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

### Link prediction
Example of training GCN on UCI dataset:

```python
python train_link_prediction_v2.py --backbone "GCN" --dataset_name $DATASET --num_epochs 50 --lr 0.001 --dropout 0.1 --hidden_dim 64 --embedding_dim 64 --num_sampled_neighbors 20
```

Example of training Transformer on UCI dataset:
```python
python train_link_prediction_v2.py --backbone "Transformer" --dataset_name $DATASET --num_epochs 50 --lr 0.001 --dropout 0.1 --hidden_dim 64 --embedding_dim 64 --num_sampled_neighbors 20
```






