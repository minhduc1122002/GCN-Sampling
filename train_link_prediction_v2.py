import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from data_loader import get_data
from model import ASH_DGT
import time


class EdgeDataset(Dataset):
    def __init__(self, sources, destinations, timestamps):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.destinations[idx], self.timestamps[idx]


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


def evaluate_model(model, sources, destinations, timestamps, num_nodes, device):
    model.eval()
    node_ids = torch.arange(num_nodes, device=device)
    edge_index = torch.tensor([sources, destinations], dtype=torch.long).to(device)
    A = build_adjacency(sources, destinations, num_nodes).to(device)
    ts = torch.tensor(timestamps, dtype=torch.float).to(device)
    with torch.no_grad():
        _, embeddings = model(node_ids, edge_index, A, ts)
    pos_edges = np.array(list(zip(sources, destinations)))
    pos_u = pos_edges[:, 0]
    pos_v = pos_edges[:, 1]
    pos_scores = (embeddings[pos_u] * embeddings[pos_v]).sum(dim=1).cpu().numpy()
    neg_edges = sample_negative_edges(sources, destinations, num_nodes, len(sources))
    neg_u = neg_edges[0].numpy()
    neg_v = neg_edges[1].numpy()
    neg_scores = (embeddings[neg_u] * embeddings[neg_v]).sum(dim=1).cpu().numpy()
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(sources)), np.zeros(len(sources))])
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    return auc, ap


def train_link_prediction(args):
    device = "cuda"

    # Load dataset (using unified get_data function).
    (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_val,
        new_test,
    ) = get_data(args.dataset_name, args.val_ratio, args.test_ratio)
    num_nodes = full_data.n_unique_nodes
    print("Number of nodes:", num_nodes)

    # Prepare training edge data.
    train_sources = np.array(train_data.sources)
    train_destinations = np.array(train_data.destinations)
    train_timestamps = np.array(train_data.timestamps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build full adjacency from full data.
    A_full = build_adjacency(full_data.sources, full_data.destinations, num_nodes).to(
        device
    )
    # For training, use training edges (subset of A_full).
    A_train = A_full.clone()

    # Create an edge dataset and DataLoader for training.
    train_edge_dataset = EdgeDataset(
        train_sources, train_destinations, train_timestamps
    )
    train_edge_loader = DataLoader(
        train_edge_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Instantiate the model.
    model = ASH_DGT(
        num_nodes=num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.embedding_dim,  # output embedding dimension
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
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = args.num_epochs

    cnt = 0
    prev_auc = 0.0
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        node_ids = torch.arange(num_nodes, device=device)
        ts_train = torch.tensor(train_timestamps, dtype=torch.float).to(device)
        full_edge_index_train = torch.tensor(
            [train_sources, train_destinations], dtype=torch.long
        ).to(device)
        # Compute full node embeddings once per epoch.
        _, embeddings = model(node_ids, full_edge_index_train, A_train, ts_train)
        epoch_loss = 0.0
        # Accumulate loss over mini-batches.
        epoch_time = time.time()
        for batch in train_edge_loader:
            batch_sources, batch_destinations, _ = batch
            batch_sources = batch_sources.to(device)
            batch_destinations = batch_destinations.to(device)
            num_pos = batch_sources.shape[0]
            # Compute positive scores.
            pos_scores = (
                embeddings[batch_sources] * embeddings[batch_destinations]
            ).sum(dim=1)
            # Sample negative edges for this mini-batch.
            neg_edges = sample_negative_edges(
                batch_sources.cpu().numpy(),
                batch_destinations.cpu().numpy(),
                num_nodes,
                num_pos,
            )
            neg_u = neg_edges[0].to(device)
            neg_v = neg_edges[1].to(device)
            neg_scores = (embeddings[neg_u] * embeddings[neg_v]).sum(dim=1)
            pos_labels = torch.ones(num_pos, device=device)
            neg_labels = torch.zeros(num_pos, device=device)
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            # print(criterion(scores, labels))
            epoch_loss = epoch_loss + criterion(scores, labels)
        epoch_loss.backward()
        optimizer.step()
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss.item()/len(train_edge_loader):.4f}"
        )
        # Evaluate model on validation set.
        val_sources = np.array(val_data.sources)
        val_destinations = np.array(val_data.destinations)
        val_timestamps = np.array(val_data.timestamps)
        auc, ap = evaluate_model(
            model, val_sources, val_destinations, val_timestamps, num_nodes, device
        )
        print(time.time(), epoch_time)
        print(
            f"Validation AUC: {auc*100:.4f}%, Validation AP: {ap*100:.4f}%, Time: {(time.time() - epoch_time):.4f}s"
        )
        # Early stopping based on validation AUC.
        if auc > prev_auc:
            cnt = 0
            prev_auc = auc
            torch.save(model.state_dict(), "link_pred_model.pth")
        else:
            cnt += 1
            if cnt == 10:
                print("Early stopping.")
                break

    # torch.save(model.state_dict(), "link_pred_model.pth")
    print("Training complete. Model saved as 'link_pred_model.pth'.")

    # Evaluate on test split.
    # load the best model
    model.load_state_dict(torch.load("link_pred_model.pth"))
    test_sources = np.array(test_data.sources)
    test_destinations = np.array(test_data.destinations)
    test_timestamps = np.array(test_data.timestamps)
    ts_test = torch.tensor(test_timestamps, dtype=torch.float).to(device)
    full_edge_index_test = torch.tensor(
        [test_sources, test_destinations], dtype=torch.long
    ).to(device)
    model.eval()
    with torch.no_grad():
        _, embeddings = model(
            torch.arange(num_nodes, device=device),
            full_edge_index_test,
            A_full,
            ts_test,
        )
    emb = embeddings
    pos_edges_test = np.array(list(zip(test_sources, test_destinations)))
    num_pos_test = len(pos_edges_test)
    pos_u_test = pos_edges_test[:, 0]
    pos_v_test = pos_edges_test[:, 1]
    pos_scores_test = (emb[pos_u_test] * emb[pos_v_test]).sum(dim=1).cpu().numpy()
    neg_edges_test = sample_negative_edges(
        test_sources, test_destinations, num_nodes, num_pos_test
    )
    neg_u_test = neg_edges_test[0].numpy()
    neg_v_test = neg_edges_test[1].numpy()
    neg_scores_test = (emb[neg_u_test] * emb[neg_v_test]).sum(dim=1).cpu().numpy()
    all_scores = np.concatenate([pos_scores_test, neg_scores_test])
    all_labels = np.concatenate([np.ones(num_pos_test), np.zeros(num_pos_test)])
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    print(f"Test AUC: {auc*100:.4f}%, Test AP: {ap*100:.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Train Link Prediction Model using ASH-DGT with mini-batch training"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., mooc, reddit, wiki, uci, socialevo, enron)",
    )
    parser.add_argument(
        "--in_dim", type=int, default=16, help="Input feature dimension"
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="Output embedding dimension"
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--attn_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=3,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num_global_nodes", type=int, default=2, help="Number of global nodes"
    )
    parser.add_argument(
        "--num_super_nodes",
        type=int,
        default=12,
        help="Number of super nodes for coarsening",
    )
    parser.add_argument(
        "--num_sampled_neighbors",
        type=int,
        default=5,
        help="Number of sampled neighbors",
    )
    parser.add_argument(
        "--coarsen_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "ve", "vn", "jc"],
        help="Coarsening method",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for coarsening methods"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--train_frac", type=float, default=0.70, help="Training fraction"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Validation ratio"
    )
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument(
        "--pos_dim", type=int, default=8, help="Positional encoding dimension"
    )
    parser.add_argument(
        "--time_dim", type=int, default=8, help="Temporal encoding dimension"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for mini-batch training"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Gamma for bandit sampling"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_link_prediction(args)


if __name__ == "__main__":
    main()
