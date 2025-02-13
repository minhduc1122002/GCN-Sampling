import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_node_classification, Data
from model import ASH_DGT  # Your integrated ASH-DGT model

def extract_node_labels(df, num_nodes):
    """
    Extract node labels for node classification.
    For each node (given by the 'idx' column), the label from its first occurrence is used.
    Returns:
       y: torch.LongTensor of shape [num_nodes]
       num_classes: number of unique classes
    """
    label_dict = {}
    for _, row in df.iterrows():
        node = int(row["idx"])
        if node not in label_dict:
            label_dict[node] = int(row["label"])
    y = np.zeros(num_nodes, dtype=int)
    for i in range(num_nodes):
        y[i] = label_dict.get(i, 0)
    y = torch.tensor(y, dtype=torch.long)
    num_classes = int(y.max().item() + 1)
    return y, num_classes

def build_graph_from_data(data_obj, num_nodes, device):
    src = torch.tensor(data_obj.sources, dtype=torch.long, device=device)
    dst = torch.tensor(data_obj.destinations, dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)
    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0  # undirected
    ts = torch.tensor(data_obj.timestamps, dtype=torch.float, device=device)
    return edge_index, A, ts

def evaluate_split(model, split_data, y, num_nodes, device):
    unique_nodes = np.array(list(split_data.unique_nodes))
    unique_nodes = torch.tensor(unique_nodes, dtype=torch.long, device=device)
    # Ensure y is on the same device.
    y = y.to(device)
    edge_index, A, ts = build_graph_from_data(split_data, num_nodes, device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.arange(num_nodes, device=device), edge_index, A, ts)
    preds = torch.argmax(logits, dim=1)
    preds_split = preds[unique_nodes]
    y_split = y[unique_nodes]
    accuracy = (preds_split == y_split).float().mean().item()
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train Node Classification Model using ASH-DGT")
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., mooc, reddit, wiki, uci, socialevo, enron)")
    parser.add_argument("--use_validation", action="store_true", help="Use validation split")
    # Data split ratios
    parser.add_argument("--train_frac", type=float, default=0.70, help="Training fraction")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Test fraction")
    # Model hyperparameters
    parser.add_argument("--in_dim", type=int, default=16, help="Input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_neighbors", type=int, default=15, help="Number of sampled neighbors")
    parser.add_argument("--num_transformer_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--num_super_nodes", type=int, default=6, help="Number of super nodes for coarsening")
    parser.add_argument("--num_global_nodes", type=int, default=2, help="Number of global nodes")
    parser.add_argument("--coarsen_method", type=str, default="kmeans", choices=["kmeans", "ve", "vn", "jc"],
                        help="Coarsening method")
    # New hyperparameters for positional and temporal encoding dimensions
    parser.add_argument("--pos_dim", type=int, default=8, help="Positional encoding dimension")
    parser.add_argument("--time_dim", type=int, default=8, help="Temporal encoding dimension")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data using your provided function
    full_data, node_features, edge_features, train_data, val_data, test_data = get_data_node_classification(
        args.dataset_name, use_validation=args.use_validation
    )
    
    # Load full CSV to extract node labels.
    df_full = pd.read_csv(f"./data/{args.dataset_name}/ml_{args.dataset_name}.csv")
    df_full.sort_values(by="ts", inplace=True)
    num_nodes = full_data.n_unique_nodes
    y, num_classes = extract_node_labels(df_full, num_nodes)
    
    print(f"Dataset {args.dataset_name}: {num_nodes} nodes, {num_classes} classes")
    
    # Build graphs for train, validation, and test splits.
    edge_index_train, A_train, ts_train = build_graph_from_data(train_data, num_nodes, device)
    edge_index_val, A_val, ts_val = build_graph_from_data(val_data, num_nodes, device)
    edge_index_test, A_test, ts_test = build_graph_from_data(test_data, num_nodes, device)
    
    # Use full node set for transductive node classification.
    node_ids = torch.arange(num_nodes, device=device)
    
    # Instantiate the ASH-DGT model with hyperparameters.
    model = ASH_DGT(num_nodes=num_nodes,
                    in_dim=args.in_dim,
                    hidden_dim=args.hidden_dim,
                    num_classes=num_classes,
                    pos_dim=args.pos_dim,
                    time_dim=args.time_dim,
                    num_neighbors=args.num_neighbors,
                    num_transformer_layers=args.num_transformer_layers,
                    nhead=args.attn_heads,
                    dropout=args.dropout,
                    num_super_nodes=args.num_super_nodes,
                    num_global_nodes=args.num_global_nodes,
                    coarsen_method=args.coarsen_method).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits = model(node_ids, edge_index_train, A_train, ts_train)
        loss = criterion(logits, y.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "node_cls_model.pth")
    print("Training complete. Model saved as 'node_cls_model.pth'.")
    
    # Evaluate on each split.
    train_acc = evaluate_split(model, train_data, y, num_nodes, device)
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    if args.use_validation:
        val_acc = evaluate_split(model, val_data, y, num_nodes, device)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
    test_acc = evaluate_split(model, test_data, y, num_nodes, device)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
