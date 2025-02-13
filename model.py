import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

###############################################
# Graph Coarsening Functions
###############################################

def graph_coarsening_kmeans(X, A, num_super_nodes, threshold=0.0):
    """
    Coarsening using K-Means clustering on node features.
    The 'threshold' parameter is accepted but not used by default.
    """
    n = X.size(0)
    if num_super_nodes <= 0:
        P = torch.eye(n, device=X.device)
        return X, A, P
    X_np = X.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=num_super_nodes, random_state=0).fit(X_np)
    labels = kmeans.labels_
    P_prime = torch.zeros(n, num_super_nodes, device=X.device)
    for i, label in enumerate(labels):
        P_prime[i, label] = 1.0
    # (Optional: one could check cluster inertia here and filter clusters based on threshold)
    cluster_sizes = P_prime.sum(dim=0)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(cluster_sizes + 1e-10))
    P = P_prime @ D_inv_sqrt
    X_coarse = P.t() @ X
    A_coarse = P.t() @ A @ P
    return X_coarse, A_coarse, P

def graph_coarsening_ve(X, A, num_super_nodes, threshold=0.5):
    """
    Coarsening using Variation-Edge (VE) based method.
    Nodes are merged based on the similarity of their normalized edge weights.
    Only candidate pairs with similarity >= threshold are considered.
    """
    n = X.size(0)
    if num_super_nodes >= n:
        P = torch.eye(n, device=X.device)
        return X, A, P

    degrees = A.sum(dim=1)
    norm_weights = A / (degrees.unsqueeze(1) + 1e-10)
    candidate_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            sim = -torch.norm(norm_weights[i] - norm_weights[j], p=1).item()
            if sim < threshold:
                continue
            candidate_pairs.append((sim, i, j))
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
    Coarsening using Variation-Neighborhood (VN) based method.
    Groups nodes with high Jaccard similarity of their neighborhoods.
    """
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
    def jaccard(i, j):
        Ni = A_bin[i]
        Nj = A_bin[j]
        inter = (Ni * Nj).sum().item()
        union = (Ni + Nj - Ni * Nj).sum().item()
        return inter / union if union > 0 else 0.0

    candidate_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            sim = jaccard(i, j)
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

def graph_coarsening_jc(X, A, num_super_nodes, threshold=0.5):
    """
    Coarsening using the Jaccard Coefficient (JC) method.
    Groups nodes whose neighborhoods have high Jaccard similarity.
    """
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
        for j in range(i+1, n):
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
    """
    Unified graph coarsening function.
    method: 'kmeans', 've', 'vn', or 'jc'
    The threshold parameter is applied in VE, VN, and JC methods.
    """
    if method == "kmeans":
        return graph_coarsening_kmeans(X, A, num_super_nodes, threshold)
    elif method == "ve":
        return graph_coarsening_ve(X, A, num_super_nodes, threshold)
    elif method == "vn":
        return graph_coarsening_vn(X, A, num_super_nodes, threshold)
    elif method == "jc":
        return graph_coarsening_jc(X, A, num_super_nodes, threshold)
    else:
        raise ValueError("Unknown coarsening method: choose from 'kmeans', 've', 'vn', or 'jc'.")

###############################################
# (The rest of your model code remains unchanged)
###############################################
# Example: AdaptiveNodeSampler, LaplacianPositionalEncoding, TemporalEncoder, and ASH_DGT are defined below.
# For brevity, here is a minimal stub for the ASH_DGT model that uses the graph coarsening function.

class AdaptiveNodeSampler(nn.Module):
    def __init__(self, embed_dim, num_neighbors, gamma=0.1):
        super(AdaptiveNodeSampler, self).__init__()
        self.num_neighbors = num_neighbors
        self.gamma = gamma
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.cached_scores = None

    def cache_attention_scores(self, attn_scores):
        self.cached_scores = attn_scores

    def forward(self, target_embed, candidate_embeds):
        B, N, _ = candidate_embeds.size()
        if self.cached_scores is not None:
            p = self.cached_scores
        else:
            Q = self.query_proj(target_embed)
            K = self.key_proj(candidate_embeds)
            scores = torch.einsum('bd,bnd->bn', Q, K) / self.scale
            p = torch.softmax(scores, dim=-1)
        uniform = torch.full_like(p, self.gamma / N)
        p = (1 - self.gamma) * p + uniform
        sampled_idx = torch.multinomial(p, self.num_neighbors, replacement=False)
        self.cached_scores = None
        return sampled_idx

class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, pos_dim, eps=1e-5, normalized=True):
        """
        Args:
            pos_dim (int): Number of eigenvectors to use.
            eps (float): Stability constant.
            normalized (bool): Use normalized Laplacian if True.
        """
        super(LaplacianPositionalEncoding, self).__init__()
        self.pos_dim = pos_dim
        self.eps = eps
        self.normalized = normalized

    def forward(self, edge_index, num_nodes):
        device = edge_index.device
        # Build dense adjacency matrix from edge_index.
        A = torch.zeros((num_nodes, num_nodes), device=device)
        for i, j in edge_index.t():
            A[i, j] = 1.0
            A[j, i] = 1.0  # assume undirected
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
        return eigvecs[:, :self.pos_dim]

class TemporalEncoder(nn.Module):
    def __init__(self, time_dim):
        super(TemporalEncoder, self).__init__()
        self.time_dim = time_dim
        self.linear = nn.Linear(1, time_dim)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cos(self.linear(t.float()))

class ASH_DGT(nn.Module):
    def __init__(self,
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
                 threshold=0.5):
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
        self.adaptive_sampler = AdaptiveNodeSampler(embed_dim=hidden_dim, num_neighbors=num_neighbors)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, node_ids, edge_index, A, edge_times):
        device = node_ids.device
        n = self.num_nodes
        x = self.node_embedding(node_ids)
        pos_enc = self.pos_encoder(edge_index, n)
        avg_time = torch.full((n, 1), edge_times.mean().item(), device=device)
        temp_enc = self.temp_encoder(avg_time)
        x_cat = torch.cat([x, pos_enc, temp_enc], dim=-1)
        x_hidden = self.input_proj(x_cat)
        
        # Adaptive node sampling (for demonstration, we simulate candidate neighbor sampling).
        B = n
        N_candidates = 50
        candidate_indices = torch.randint(0, n, (B, N_candidates), device=device)
        candidate_embeds = x_hidden[candidate_indices]
        sampled_idx = self.adaptive_sampler(x_hidden, candidate_embeds)
        sampled_neighbors = torch.gather(candidate_embeds, 1,
                                         sampled_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        transformer_input = torch.cat([x_hidden.unsqueeze(1), sampled_neighbors], dim=1)
        
        # Graph coarsening and global node generation.
        if self.num_super_nodes > 0:
            X_coarse, A_coarse, _ = graph_coarsening(x_hidden, A, self.num_super_nodes,
                                                     method=self.coarsen_method,
                                                     threshold=self.threshold)
            # For simplicity, global nodes are generated using kmeans on X_coarse.
            X_np = X_coarse.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=self.num_global_nodes, random_state=0).fit(X_np)
            centroids = torch.tensor(kmeans.cluster_centers_, device=x_hidden.device, dtype=x_hidden.dtype)
            global_nodes = centroids
        else:
            global_nodes = x_hidden.mean(dim=0, keepdim=True)
        global_nodes_exp = global_nodes.unsqueeze(0).expand(B, -1, -1)
        transformer_input = torch.cat([transformer_input, global_nodes_exp], dim=1)
        
        transformer_input = transformer_input.transpose(0, 1)
        transformer_output = self.transformer_encoder(transformer_input)
        final_representation = transformer_output[0]
        logits = self.classifier(final_representation)
        return logits
