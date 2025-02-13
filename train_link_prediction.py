import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from data_loader import get_data
from model import ASH_DGT

def sample_negative_edges(sources, destinations, num_nodes, num_samples):
    # Create a set of positive edges (both directions)
    pos_set = set()
    for u, v in zip(sources, destinations):
        pos_set.add((u, v))
        pos_set.add((v, u))
    negatives = []
    while len(negatives) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue
        if (u, v) in pos_set:
            continue
        negatives.append((u, v))
    neg_edge_index = np.array(negatives).T  # shape: [2, num_samples]
    return torch.tensor(neg_edge_index, dtype=torch.long)

def build_adjacency(sources, destinations, num_nodes):
    A = np.zeros((num_nodes, num_nodes))
    for u, v in zip(sources, destinations):
        A[u, v] = 1
        A[v, u] = 1
    return torch.tensor(A, dtype=torch.float)

def train_link_prediction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data using get_data; here we use the same function for link prediction.
    # We assume the CSV file and feature files follow the ml_datasetname naming convention.
    node_features, edge_features, full_data, train_data, val_data, test_data, new_val, new_test = get_data(
        args.dataset_name, args.val_ratio, args.test_ratio
    )
    num_nodes = full_data.n_unique_nodes
    print("Number of nodes:", num_nodes)
    
    # Use training data for link prediction.
    train_sources = np.array(train_data.sources)
    train_destinations = np.array(train_data.destinations)
    train_timestamps = np.array(train_data.timestamps)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Instantiate the model.
    model = ASH_DGT(num_nodes=num_nodes,
                    in_dim=args.in_dim,
                    hidden_dim=args.hidden_dim,
                    num_classes=args.embedding_dim,  # output embedding dimension
                    pos_dim=8,
                    time_dim=8,
                    num_neighbors=args.num_sampled_neighbors,
                    num_transformer_layers=args.num_transformer_layers,
                    nhead=args.attn_heads,
                    dropout=args.dropout,
                    num_super_nodes=args.num_super_nodes,
                    num_global_nodes=args.num_global_nodes,
                    coarsen_method=args.coarsen_method).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = args.num_epochs

    A = build_adjacency(full_data.sources, full_data.destinations, num_nodes).to(device)
        
    # Add train sources and destinations to A_train
    A_train = A.clone()
    A_train[train_sources, train_destinations] = 1
    A_train[train_destinations, train_sources] = 1
        
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Use full node set.
        node_ids = torch.arange(num_nodes, device=device)
        # For training, build A from train sources/destinations.
        # Build adjacency matrix for the full graph
  
        # Add val sources and destinations to A_train
        # A_train[val_sources, val_destinations] = 1
        # A_train[val_destinations, val_sources] = 1
        # A_val = A_train.clone()

        ts_train = torch.tensor(train_timestamps, dtype=torch.float).to(device)
        # Forward pass: get node embeddings.
        embeddings = model(node_ids,
                           torch.tensor([train_sources, train_destinations], dtype=torch.long).to(device),
                           A_train, ts_train)
        # Compute scores for positive edges.
        pos_edges = list(zip(train_sources, train_destinations))
        num_pos = len(pos_edges)
        pos_edges = np.array(pos_edges)
        pos_u = pos_edges[:,0]
        pos_v = pos_edges[:,1]
        pos_scores = (embeddings[pos_u] * embeddings[pos_v]).sum(dim=1)
        # Sample negative edges.
        neg_edges = sample_negative_edges(train_sources, train_destinations, num_nodes, num_pos)
        neg_u = neg_edges[0].numpy()
        neg_v = neg_edges[1].numpy()
        neg_scores = (embeddings[neg_u] * embeddings[neg_v]).sum(dim=1)
        pos_labels = torch.ones(num_pos, device=device)
        neg_labels = torch.zeros(num_pos, device=device)
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        # add validation
        
    
    torch.save(model.state_dict(), "link_pred_model.pth")
    print("Training complete. Model saved as 'link_pred_model.pth'.")
    
    # Evaluation on test split.
    test_sources = np.array(test_data.sources)
    test_destinations = np.array(test_data.destinations)
    test_timestamps = np.array(test_data.timestamps)
    ts_test = torch.tensor(test_timestamps, dtype=torch.float).to(device)
    # Add test sources and destinations to A_train
    A_train[test_sources, test_destinations] = 1
    A_train[test_destinations, test_sources] = 1
    A_test = A_train.clone()
    node_ids = torch.arange(num_nodes, device=device)
    model.eval()
    with torch.no_grad():
        embeddings = model(node_ids,
                           torch.tensor([test_sources, test_destinations], dtype=torch.long).to(device),
                           A_test, ts_test)
    emb = embeddings
    pos_edges_test = list(zip(test_sources, test_destinations))
    num_pos_test = len(pos_edges_test)
    pos_edges_test = np.array(pos_edges_test)
    pos_u_test = pos_edges_test[:,0]
    pos_v_test = pos_edges_test[:,1]
    pos_scores_test = (emb[pos_u_test] * emb[pos_v_test]).sum(dim=1).cpu().numpy()
    neg_edges_test = sample_negative_edges(test_sources, test_destinations, num_nodes, num_pos_test)
    neg_u_test = neg_edges_test[0].numpy()
    neg_v_test = neg_edges_test[1].numpy()
    neg_scores_test = (emb[neg_u_test] * emb[neg_v_test]).sum(dim=1).cpu().numpy()
    all_scores = np.concatenate([pos_scores_test, neg_scores_test])
    all_labels = np.concatenate([np.ones(num_pos_test), np.zeros(num_pos_test)])
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    print(f"Test AUC: {auc*100:.2f}%, Test AP: {ap*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Train link prediction model using dataset_name")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., mooc, reddit, wiki, uci, socialevo, enron)")
    parser.add_argument("--in_dim", type=int, default=16, help="Input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_transformer_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num_global_nodes", type=int, default=2, help="Number of global nodes")
    parser.add_argument("--num_super_nodes", type=int, default=6, help="Number of super nodes for coarsening")
    parser.add_argument("--num_sampled_neighbors", type=int, default=15, help="Number of sampled neighbors")
    parser.add_argument("--coarsen_method", type=str, default="kmeans", choices=["kmeans", "ve", "vn"], help="Coarsening method")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--train_frac", type=float, default=0.70, help="Training fraction")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    train_link_prediction(args)

if __name__ == "__main__":
    main()
