import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.nn.conv import GCNConv


###############################################
# Graph Coarsening Functions
###############################################
def get_one_hop_neighbors(A, node_idx, max_neighbors=50):
    # A: [num_nodes, num_nodes] dense adjacency matrix.
    neighbors = (A[node_idx] > 0).nonzero(as_tuple=False).squeeze()
    if neighbors.dim() == 0:
        neighbors = neighbors.unsqueeze(0)
    # Exclude self.
    neighbors = neighbors[neighbors != node_idx]
    if neighbors.numel() == 0:
        return None
    if neighbors.numel() > max_neighbors:
        perm = torch.randperm(neighbors.numel(), device=A.device)[:max_neighbors]
        neighbors = neighbors[perm]
    return neighbors


def graph_coarsening_kmeans(X, A, num_super_nodes, threshold=0.0):
    """
    Coarsening using K-Means clustering on node features.
    If X contains NaN values, we skip KMeans and return X and an identity matrix.
    """
    n = X.size(0)
    if num_super_nodes <= 0:
        P = torch.eye(n, device=X.device)
        return X, A, P
    # Replace NaNs with 0.
    X_np = torch.nan_to_num(X, nan=0.0).cpu().detach().numpy()
    # If there are still any NaN (shouldn't be), then skip clustering.
    if np.isnan(X_np).any():
        print("Warning: NaN detected in features. Skipping KMeans coarsening.")
        P = torch.eye(n, device=X.device)
        return X, A, P
    # Otherwise, do KMeans.
    kmeans = KMeans(n_clusters=num_super_nodes, random_state=0).fit(X_np)
    labels = kmeans.labels_
    P_prime = torch.zeros(n, num_super_nodes, device=X.device)
    for i, label in enumerate(labels):
        P_prime[i, label] = 1.0
    cluster_sizes = P_prime.sum(dim=0)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(cluster_sizes + 1e-10))
    P = P_prime @ D_inv_sqrt
    X_coarse = P.t() @ X
    A_coarse = P.t() @ A @ P
    return X_coarse, A_coarse, P


def graph_coarsening_ve(X, A, num_super_nodes, threshold=0.5):
    """
    Variation-Edge (VE) Coarsening:
    """
    n = X.size(0)
    if num_super_nodes >= n:
        P = torch.eye(n, device=X.device)
        return X, A, P

    # Compute normalized edge weight matrix (each row sums to 1)
    deg = A.sum(dim=1)  # [n]
    norm_weights = A / (deg.unsqueeze(1) + 1e-10)  # [n, n]

    # Compute pairwise L1 distances in a vectorized way.
    dist_matrix = torch.cdist(norm_weights, norm_weights, p=1)  # [n, n]
    sim_matrix = -dist_matrix  # similarity = negative L1 distance

    # Extract upper-triangular indices (move them to the same device as X)
    triu_idx = torch.triu_indices(n, n, offset=1).to(X.device)
    sim_values = sim_matrix[triu_idx[0], triu_idx[1]]

    # Filter candidate pairs with similarity above threshold.
    valid_mask = sim_values >= threshold
    valid_idx = triu_idx[:, valid_mask]
    valid_sim = sim_values[valid_mask].tolist()

    candidate_pairs = []
    for k in range(valid_idx.size(1)):
        i = valid_idx[0, k].item()
        j = valid_idx[1, k].item()
        candidate_pairs.append((valid_sim[k], i, j))
    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    # Union-find merging.
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merged = 0
    for sim, i, j in candidate_pairs:
        pi = find(i)
        pj = find(j)
        if pi != pj:
            parent[pj] = pi
            merged += 1
            if n - merged <= num_super_nodes:
                break

    P_prime = torch.zeros(n, n - merged, device=X.device)
    mapping = {}
    new_label = 0
    for i in range(n):
        pi = find(i)
        if pi not in mapping:
            mapping[pi] = new_label
            new_label += 1
        P_prime[i, mapping[pi]] = 1.0

    cluster_sizes = P_prime.sum(dim=0)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(cluster_sizes + 1e-10))
    P = P_prime @ D_inv_sqrt
    X_coarse = P.t() @ X
    A_coarse = P.t() @ A @ P
    return X_coarse, A_coarse, P


def graph_coarsening_vn(X, A, num_super_nodes, threshold=0.5):
    """
    Variation-Neighborhood (VN) Coarsening:
    """
    n = X.size(0)
    if num_super_nodes >= n:
        P = torch.eye(n, device=A.device)
        return X, A, P

    A_bin = (A > 0).float()  # [n, n]
    row_sum = A_bin.sum(dim=1)  # [n]

    # Compute pairwise intersection and union using broadcasting.
    inter = (A_bin.unsqueeze(1) * A_bin.unsqueeze(0)).sum(dim=2)  # [n, n]
    union = row_sum.unsqueeze(1) + row_sum.unsqueeze(0) - inter  # [n, n]
    union[union == 0] = 1e-10  # avoid division by zero
    jaccard_matrix = inter / union  # [n, n]

    # Extract upper-triangular indices (move to same device).
    triu_idx = torch.triu_indices(n, n, offset=1).to(X.device)
    sim_values = jaccard_matrix[triu_idx[0], triu_idx[1]]
    valid_mask = sim_values >= threshold
    valid_idx = triu_idx[:, valid_mask]
    valid_sim = sim_values[valid_mask].tolist()

    candidate_pairs = []
    for k in range(valid_idx.size(1)):
        i = valid_idx[0, k].item()
        j = valid_idx[1, k].item()
        candidate_pairs.append((valid_sim[k], i, j))
    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merged = 0
    for sim, i, j in candidate_pairs:
        pi = find(i)
        pj = find(j)
        if pi != pj:
            parent[pj] = pi
            merged += 1
            if n - merged <= num_super_nodes:
                break

    P_prime = torch.zeros(n, n - merged, device=A.device)
    mapping = {}
    new_label = 0
    for i in range(n):
        pi = find(i)
        if pi not in mapping:
            mapping[pi] = new_label
            new_label += 1
        P_prime[i, mapping[pi]] = 1.0

    cluster_sizes = P_prime.sum(dim=0)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(cluster_sizes + 1e-10))
    P = P_prime @ D_inv_sqrt
    X_coarse = P.t() @ X
    A_coarse = P.t() @ A @ P
    return X_coarse, A_coarse, P


def graph_coarsening_jc(X, A, num_super_nodes, threshold=0.5):
    n = X.size(0)
    if num_super_nodes >= n:
        P = torch.eye(n, device=X.device)
        return X, A, P

    A_bin = (A > 0).float()
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def jaccard_similarity(i, j):
        Ni = A_bin[i]
        Nj = A_bin[j]
        inter = (Ni * Nj).sum().item()
        union = (Ni + Nj - Ni * Nj).sum().item()
        return inter / union if union > 0 else 0.0

    candidate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_similarity(i, j)
            if sim < threshold:
                continue
            candidate_pairs.append((sim, i, j))
    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    merged = 0
    for sim, i, j in candidate_pairs:
        pi = find(i)
        pj = find(j)
        if pi != pj:
            parent[pj] = pi
            merged += 1
            if n - merged <= num_super_nodes:
                break
    P_prime = torch.zeros(n, n - merged, device=X.device)
    mapping = {}
    new_label = 0
    for i in range(n):
        pi = find(i)
        if pi not in mapping:
            mapping[pi] = new_label
            new_label += 1
        P_prime[i, mapping[pi]] = 1.0
    cluster_sizes = P_prime.sum(dim=0)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(cluster_sizes + 1e-10))
    P = P_prime @ D_inv_sqrt
    X_coarse = P.t() @ X
    A_coarse = P.t() @ A @ P
    return X_coarse, A_coarse, P


def graph_coarsening(X, A, num_super_nodes, method="kmeans", threshold=0.5):
    if method == "kmeans":
        return graph_coarsening_kmeans(X, A, num_super_nodes, threshold)
    elif method == "ve":
        return graph_coarsening_ve(X, A, num_super_nodes, threshold)
    elif method == "vn":
        return graph_coarsening_vn(X, A, num_super_nodes, threshold)
    elif method == "jc":
        return graph_coarsening_jc(X, A, num_super_nodes, threshold)
    else:
        raise ValueError(
            "Unknown coarsening method: choose from 'kmeans', 've', 'vn', or 'jc'."
        )


###############################################
# Laplacian Positional Encoding
###############################################
class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, pos_dim, eps=1e-5, normalized=True):
        super(LaplacianPositionalEncoding, self).__init__()
        self.pos_dim = pos_dim
        self.eps = eps
        self.normalized = normalized

    def forward(self, edge_index, num_nodes):
        device = edge_index.device
        A = torch.zeros((num_nodes, num_nodes), device=device)
        for i, j in edge_index.t():
            A[i, j] = 1.0
            A[j, i] = 1.0
        if self.normalized:
            deg = A.sum(dim=1)
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + self.eps))
            L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            deg = A.sum(dim=1)
            D = torch.diag(deg)
            L = D - A + self.eps * torch.eye(num_nodes, device=device)
        try:
            eigvals, eigvecs = torch.linalg.eigh(L)
        except Exception as e:
            print("Eigen decomposition failed:", e)
            eigvecs = torch.zeros((num_nodes, num_nodes), device=device)
        return eigvecs[:, : self.pos_dim]


###############################################
# Temporal Encoder
###############################################
class TemporalEncoder(nn.Module):
    def __init__(self, time_dim):
        super(TemporalEncoder, self).__init__()
        self.time_dim = time_dim
        self.linear = nn.Linear(1, time_dim)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cos(self.linear(t.float()))


###############################################
# AdaptiveSampler: Returns sampled neighbor indices using EXP3-style sampling.
###############################################
class AdaptiveSampler(nn.Module):
    def __init__(self, embed_dim, num_neighbors, gamma=0.1):
        super(AdaptiveSampler, self).__init__()
        self.num_neighbors = num_neighbors
        self.gamma = gamma
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**0.5

    def forward(self, target_embed, candidate_embeds):
        # Check if target_embed is empty
        if target_embed.numel() == 0:
            return torch.empty(
                0, self.num_neighbors, device=target_embed.device, dtype=torch.long
            )
        # Ensure target_embed is 2D: [B, D]
        if target_embed.dim() > 2:
            target_embed = target_embed.view(target_embed.size(0), -1)
        # target_embed: [B, D]; candidate_embeds: [B, N, D]
        Q = self.query_proj(target_embed)  # [B, D]
        K = self.key_proj(candidate_embeds)  # [B, N, D]
        scores = torch.einsum("bd,bnd->bn", Q, K) / self.scale  # [B, N]
        probs = torch.softmax(scores, dim=-1)
        uniform = torch.full_like(probs, self.gamma / candidate_embeds.size(1))
        probs = (1 - self.gamma) * probs + uniform
        sampled_idx = torch.multinomial(probs, self.num_neighbors, replacement=False)
        return sampled_idx


###############################################
# ASH-DGT Model for Link Prediction
###############################################


class ASH_DGT(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        hidden_dim,
        num_classes,
        pos_dim=8,
        time_dim=8,
        num_neighbors=15,
        num_transformer_layers=3,
        nhead=8,
        dropout=0.2,
        num_super_nodes=6,
        num_global_nodes=2,
        coarsen_method="kmeans",
        threshold=0.5,
        gamma=0.4,
    ):
        super(ASH_DGT, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_super_nodes = num_super_nodes
        self.num_global_nodes = num_global_nodes
        self.coarsen_method = coarsen_method
        self.threshold = threshold

        self.node_embedding = nn.Embedding(num_nodes, in_dim)
        self.pos_encoder = LaplacianPositionalEncoding(pos_dim)
        self.temp_encoder = TemporalEncoder(time_dim)
        self.input_proj = nn.Linear(in_dim + pos_dim + time_dim, hidden_dim)

        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self.adaptive_sampler = AdaptiveSampler(
            embed_dim=hidden_dim, num_neighbors=num_neighbors, gamma=gamma
        )
        self.edge_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_ids, edge_index, A, edge_times):
        device = node_ids.device
        # n = self.num_nodes
        n = node_ids.size(0)

        # Compute initial node embeddings.
        x = self.node_embedding(node_ids)  # [n, in_dim]
        pos_enc = self.pos_encoder(edge_index, n)  # [n, pos_dim]
        avg_time = torch.full((n, 1), edge_times.mean().item(), device=device)
        temp_enc = self.temp_encoder(avg_time)  # [n, time_dim]
        x_cat = torch.cat([x, pos_enc, temp_enc], dim=1)  # [n, in_dim+pos_dim+time_dim]
        h = self.input_proj(x_cat)  # [n, hidden_dim]

        # Transformer encoding over all nodes.
        # h_seq = h.unsqueeze(0)  # [1, n, hidden_dim]
        # h_trans = self.transformer_encoder(h_seq)
        # h = h_trans.squeeze(0)  # [n, hidden_dim]
        h = self.gcn(h, edge_index)
        # Precompute aggregated super and global features.
        if self.num_super_nodes > 0:
            X_coarse, A_coarse, _ = graph_coarsening(
                h,
                A,
                self.num_super_nodes,
                method=self.coarsen_method,
                threshold=self.threshold,
            )
            super_nodes = X_coarse  # [num_super_nodes, hidden_dim]
            agg_super = super_nodes.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            agg_super = h.mean(dim=0, keepdim=True)
        agg_global = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        # Vectorized neighbor retrieval:
        E = edge_index.size(1)
        max_nbr = 50
        # Precompute one-hop neighbors for all nodes.
        neighbors_list = []
        for i in range(n):
            nbrs = get_one_hop_neighbors(A, i, max_neighbors=max_nbr)
            if nbrs is None or nbrs.numel() == 0:
                padded = torch.full((max_nbr,), i, device=device, dtype=torch.long)
            else:
                pad_len = max_nbr - nbrs.numel()
                if pad_len > 0:
                    padded = torch.cat(
                        [
                            nbrs,
                            torch.full((pad_len,), i, device=device, dtype=torch.long),
                        ]
                    )
                else:
                    padded = nbrs[:max_nbr]
            neighbors_list.append(padded)
        neighbors_tensor = torch.stack(neighbors_list, dim=0)  # [n, max_nbr]

        # For each edge, get candidate neighbor indices for u and v.
        u_idx = edge_index[0]  # [E]
        v_idx = edge_index[1]  # [E]
        cand_u = neighbors_tensor[u_idx]  # [E, max_nbr]
        cand_v = neighbors_tensor[v_idx]  # [E, max_nbr]
        candidates_u = h[cand_u]  # [E, max_nbr, hidden_dim]
        candidates_v = h[cand_v]  # [E, max_nbr, hidden_dim]
        target_u = h[u_idx].unsqueeze(1)  # [E, 1, hidden_dim]
        target_v = h[v_idx].unsqueeze(1)  # [E, 1, hidden_dim]

        # Use AdaptiveSampler in batch.
        sampled_idx_u = self.adaptive_sampler(
            target_u, candidates_u
        )  # [E, num_neighbors]
        sampled_idx_v = self.adaptive_sampler(
            target_v, candidates_v
        )  # [E, num_neighbors]

        E_range = torch.arange(E, device=device).unsqueeze(1)  # [E, 1]
        sampled_neighbors_u = candidates_u[E_range, sampled_idx_u]
        sampled_neighbors_v = candidates_v[E_range, sampled_idx_v]

        chunk_size = 64  # adjust as needed
        E_total = sampled_neighbors_u.size(0)
        agg_u_list = []
        agg_v_list = []
        for i in range(0, E_total, chunk_size):
            chunk_u = sampled_neighbors_u[i : i + chunk_size, :, :]
            chunk_v = sampled_neighbors_v[i : i + chunk_size, :, :]
            agg_u = self.edge_conv(chunk_u.transpose(1, 2)).transpose(1, 2)
            agg_v = self.edge_conv(chunk_v.transpose(1, 2)).transpose(1, 2)
            agg_u = chunk_u.mean(dim=1)
            agg_v = chunk_v.mean(dim=1)
            agg_u_list.append(agg_u)
            agg_v_list.append(agg_v)
        agg_u = torch.cat(agg_u_list, dim=0)
        agg_v = torch.cat(agg_v_list, dim=0)
        agg_super_exp = agg_super.expand(E, -1)  # [E, hidden_dim]
        agg_global_exp = agg_global.expand(E, -1)  # [E, hidden_dim]
        # Build edge sequence for each edge:
        # [h[u], agg_u, agg_v, h[v], agg_super, agg_global]
        seq_u = h[u_idx]  # [E, hidden_dim]
        seq_v = h[v_idx]  # [E, hidden_dim]
        updated_h = h.clone()
        updated_u = self.mlp(seq_u + agg_u + agg_super_exp + agg_global_exp)
        updated_v = self.mlp(seq_v + agg_v + agg_super_exp + agg_global_exp)
        updated_h.index_copy_(0, u_idx, updated_u)
        updated_h.index_copy_(0, v_idx, updated_v)

        logits = self.classifier(updated_h)
        return logits, updated_h