import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_loader import get_data_node_classification_inductive, Data
from model import ASH_DGT

class NodeDataset(Dataset):
    def __init__(self, num_nodes):
        # Dataset of node indices from 0 to num_nodes-1.
        self.indices = np.arange(num_nodes)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.indices[idx]

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
    A[dst, src] = 1.0
    ts = torch.tensor(data_obj.timestamps, dtype=torch.float, device=device)
    return edge_index, A, ts

def build_subgraph_from_data(data_obj, sub_nodes, device):
    """
    Given a set of global node IDs (sub_nodes) for a mini-batch,
    re-index them to [0, len(sub_nodes)-1] and construct the induced subgraph.
    """
    # Create mapping from global id to local id.
    mapping = {global_id: local_id for local_id, global_id in enumerate(sub_nodes)}
    # Map sources and destinations.
    # Here we assume that data_obj.sources and data_obj.destinations are arrays of global IDs.
    sources = np.array([mapping[s] for s in data_obj.sources if s in mapping])
    destinations = np.array([mapping[d] for d in data_obj.destinations if d in mapping])
    n = len(sub_nodes)
    src = torch.tensor(sources, dtype=torch.long, device=device)
    dst = torch.tensor(destinations, dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)
    A = torch.zeros((n, n), device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    # For timestamps, we simply take a dummy value (or you can filter appropriately).
    ts = torch.tensor(data_obj.timestamps[:n], dtype=torch.float, device=device)
    return edge_index, A, ts

def evaluate_split(model, split_data, y, num_nodes, device):
    # Evaluate on nodes in the split (unique nodes)
    unique_nodes = np.array(list(split_data.unique_nodes))
    unique_nodes = torch.tensor(unique_nodes, dtype=torch.long, device=device)
    edge_index, A, ts = build_graph_from_data(split_data, num_nodes, device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.arange(num_nodes, device=device), edge_index, A, ts)
    preds = torch.argmax(logits, dim=1)
    preds_split = preds[unique_nodes]
    y_split = y[unique_nodes].to(device)
    accuracy = (preds_split == y_split).float().mean().item()
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train Node Classification Model using ASH-DGT with mini-batch training")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., mooc, reddit, wiki, uci, socialevo, enron)")
    parser.add_argument("--use_validation", action="store_true", help="Use validation split")
    parser.add_argument("--in_dim", type=int, default=16, help="Input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_transformer_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num_global_nodes", type=int, default=2, help="Number of global nodes")
    parser.add_argument("--num_super_nodes", type=int, default=12, help="Number of super nodes for coarsening")
    parser.add_argument("--num_sampled_neighbors", type=int, default=5, help="Number of sampled neighbors")
    parser.add_argument("--coarsen_method", type=str, default="ve", choices=["kmeans", "ve", "vn", "jc"],
                        help="Coarsening method")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for coarsening methods")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--train_frac", type=float, default=0.70, help="Training fraction")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--pos_dim", type=int, default=8, help="Positional encoding dimension")
    parser.add_argument("--time_dim", type=int, default=8, help="Temporal encoding dimension")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for mini-batch training")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for sampling neighbors")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset.
    node_features, edge_features, full_data, train_data, val_data, test_data, new_val_data, new_test_data = \
        get_data_node_classification_inductive(args.dataset_name, use_validation=args.use_validation)
    
    
    # Load CSV and extract node labels.
    df_full = pd.read_csv(f"./data/{args.dataset_name}/ml_{args.dataset_name}.csv")
    df_full.sort_values(by="ts", inplace=True)
    num_nodes = full_data.n_unique_nodes
    y, num_classes = extract_node_labels(df_full, num_nodes)
    y = y.to(device)
    
    print(f"Dataset {args.dataset_name}: {num_nodes} nodes, {num_classes} classes")
    
    # Build graphs for train, validation, and test splits.
    edge_index_train, A_train, ts_train = build_graph_from_data(train_data, num_nodes, device)
    edge_index_val, A_val, ts_val = build_graph_from_data(new_val_data, num_nodes, device)
    edge_index_test, A_test, ts_test = build_graph_from_data(new_test_data, num_nodes, device)
    
    # Prepare mini-batch dataset of node indices.
    dataset = NodeDataset(num_nodes)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Instantiate the ASH-DGT model.
    model = ASH_DGT(num_nodes=num_nodes,
                    in_dim=args.in_dim,
                    hidden_dim=args.hidden_dim,
                    num_classes=num_classes,
                    pos_dim=args.pos_dim,
                    time_dim=args.time_dim,
                    num_neighbors=args.num_sampled_neighbors,
                    num_transformer_layers=args.num_transformer_layers,
                    nhead=args.attn_heads,
                    dropout=args.dropout,
                    num_super_nodes=args.num_super_nodes,
                    num_global_nodes=args.num_global_nodes,
                    coarsen_method=args.coarsen_method,
                    threshold=args.threshold,
                    gamma=args.gamma,
                    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    cnt = 0
    prev_val_acc = 0.0
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            batch = batch.to(device)
            # Build induced subgraph for the batch.
            # Remap global indices in batch to local indices.
            sub_nodes = np.sort(batch.cpu().numpy())
            mapping = {global_id: local_id for local_id, global_id in enumerate(sub_nodes)}
            # Create a dummy Data object for the subgraph.
            # We use the batch nodes as both sources and destinations.
            sub_data = Data(
                sources=np.array([mapping[x] for x in sub_nodes]),
                destinations=np.array([mapping[x] for x in sub_nodes]),
                timestamps=[ts_train.mean().item()] * len(sub_nodes),
                edge_idxs=np.array([mapping[x] for x in sub_nodes]),
                labels=np.zeros(len(sub_nodes), dtype=int)
            )
            sub_edge_index, sub_A, sub_ts = build_subgraph_from_data(sub_data, sub_nodes, device)
            
            logits, _ = model(torch.tensor(np.arange(len(sub_nodes)), device=device),
                                       sub_edge_index, sub_A, sub_ts)
            loss = criterion(logits, y[torch.tensor(sub_nodes, device=device)])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}")
        # Evaluate on splits.
        train_acc = evaluate_split(model, train_data, y, num_nodes, device)
        # print(f"Train Accuracy: {train_acc*100:.4f}%")
        # val_acc = evaluate_split(model, val_data, y, num_nodes, device)
        # print(f"Validation Accuracy: {val_acc*100:.4f}%")
        # if val_acc > prev_val_acc:
        #     prev_val_acc = val_acc
        #     cnt = 0
        #     torch.save(model.state_dict(), "node_cls_model.pth")
        # else:
        #     cnt += 1
        # if cnt == 5:
        #     print("Early stopping...")
        #     break
    
    # torch.save(model.state_dict(), "node_cls_model.pth")
    print("Training complete. Model saved as 'node_cls_model.pth'.")
    
    # Evaluate on splits.
    test_acc = evaluate_split(model, new_test_data, y, num_nodes, device)
    print(f"Test Accuracy: {test_acc*100:.4f}%")

if __name__ == "__main__":
    main()
