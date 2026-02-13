import torch
import torch.nn as nn
import numpy as np

def normalize_matrix(mat):
    return mat / (mat.norm() + 1e-8)

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def signed_attention(self, Q, K, V, attention_weights):
        batch_size, num_nodes = Q.size(0), Q.size(1)
        
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        QK = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_scores = torch.zeros_like(QK)

        for b in range(batch_size):
            for w in attention_weights[b]:
                # w is (N, N). We broaden to (1, 1, N, N) for broadcasting
                w_expanded = w.unsqueeze(0).unsqueeze(0)
                scaled_QK = QK[b:b+1] * w_expanded
                
                sign = torch.sign(scaled_QK)
                abs_scaled_QK = torch.abs(scaled_QK)
                exp_abs = torch.exp(abs_scaled_QK)
                
                # Softmax-like normalization preserving sign
                attention = sign * exp_abs / (torch.sum(exp_abs, dim=-1, keepdim=True) + 1e-10)
                attention_scores[b:b+1] += attention

            if len(attention_weights[b]) > 0:
                attention_scores[b:b+1] /= len(attention_weights[b])

        attention_scores = self.dropout(attention_scores)
        output = torch.matmul(attention_scores, V) # (B, H, N, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        output = self.out_linear(output)

        return output, attention_scores

    def forward(self, x, attention_weights):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        attention_output, attention_scores = self.signed_attention(Q, K, V, attention_weights)
        x = self.layer_norm1(x + attention_output)
        
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.layer_norm2(x + ffn_output)

        return x, attention_scores

class SignedGraphTransformer(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, num_layers, num_heads, dropout=0.1, 
                 attention_type='first_hop', max_hops=2, diffusion_theta=0.1, 
                 pos_encoding_type="learnable"):
        super(SignedGraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.attention_type = attention_type
        self.max_hops = max_hops
        self.diffusion_theta = diffusion_theta
        self.pos_encoding_type = pos_encoding_type
        
        # Projections
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.positional_proj = nn.Linear(30, hidden_dim) # Assuming 30 gradients

        if self.pos_encoding_type == "learnable":
            self.pos_embed = nn.Parameter(torch.randn(num_nodes, hidden_dim))

        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.edge_weight_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def compute_laplacian_positional_encoding(self, adj_matrix):
        # adj_matrix: (N, N)
        adj_matrix = adj_matrix.float()
        norm = adj_matrix.norm(dim=-1, keepdim=True) + 1e-8
        normed_fc = adj_matrix / norm
        cos_sim = torch.matmul(normed_fc, normed_fc.T).clamp(-1.0, 1.0)
        A = 1 - (1 / torch.pi) * torch.arccos(cos_sim)

        degrees = torch.sum(A, dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-10))
        L_sym = torch.eye(self.num_nodes, device=adj_matrix.device) - D_inv_sqrt @ A @ D_inv_sqrt # Note: logic adjusted slightly for stability
        
        # Eigendecomposition
        try:
            # simple fix for hermitian requirement
            L_sym = (L_sym + L_sym.T) / 2
            eigvals, eigvecs = torch.linalg.eigh(L_sym)
            eigvecs = eigvecs / (eigvecs.norm(dim=0, keepdim=True) + 1e-8)
            # Use smallest non-trivial eigenvectors? Code used top indices via argsort descending
            # We follow code:
            top_idx = torch.argsort(eigvals, descending=True)[:30]
            G = eigvecs[:, top_idx]
            
            # If we don't have 30 eigenvectors, pad?
            if G.shape[1] < 30:
                padding = torch.zeros(self.num_nodes, 30 - G.shape[1], device=adj_matrix.device)
                G = torch.cat([G, padding], dim=1)
                
            return self.positional_proj(G)
        except Exception as e:
            # Fallback for numerical instability
            return torch.zeros(self.num_nodes, 30, device=adj_matrix.device) @ self.positional_proj.weight.T

    def compute_diffusion_matrix(self, adj_matrix):
        degrees = torch.sum(torch.abs(adj_matrix), dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-10))
        A_norm = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        
        S = torch.eye(self.num_nodes, device=adj_matrix.device)
        A_k = A_norm
        for k in range(1, 10):
            S += (self.diffusion_theta ** k) * A_k
            A_k = A_k @ A_norm
        return S

    def compute_multi_hop_weights(self, adj_matrix):
        weights = [adj_matrix]
        A_k = adj_matrix
        for k in range(1, self.max_hops):
            A_k = A_k @ adj_matrix
            A_k = normalize_matrix(A_k)
            weights.append(A_k)
        return weights

    def get_masked_edge_predictions(self, h, adj_matrix, mask_edges):
        """
        h: (B, N, D)
        mask_edges: (B, N, N) 0=masked
        """
        B, N, D = h.shape
        all_preds = []
        all_targets = []

        for b in range(B):
            # Indices where mask is 0
            mask_indices = (mask_edges[b] == 0).nonzero(as_tuple=False)
            if mask_indices.shape[0] == 0:
                continue
            
            i_idx, j_idx = mask_indices[:, 0], mask_indices[:, 1]
            h_i = h[b, i_idx]
            h_j = h[b, j_idx]
            
            edge_feat = torch.cat([h_i, h_j], dim=-1)
            pred = self.edge_weight_predictor(edge_feat).squeeze(-1)
            target = adj_matrix[b, i_idx, j_idx]
            
            all_preds.append(pred)
            all_targets.append(target)

        if len(all_preds) == 0:
            return torch.tensor([]), torch.tensor([])
            
        return torch.cat(all_preds), torch.cat(all_targets)

    def forward(self, x, adj_matrix, mask_edges=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj_matrix = adj_matrix.unsqueeze(0)
            if mask_edges is not None: mask_edges = mask_edges.unsqueeze(0)

        batch_size = x.size(0)
        h = self.input_proj(x)

        # Positional Encoding
        if self.pos_encoding_type == "learnable":
            pos_encoding = self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            pos_encoding = torch.stack([
                self.compute_laplacian_positional_encoding(adj_matrix[b])
                for b in range(batch_size)
            ])
        h = h + pos_encoding

        # Prepare Attention Weights
        attention_weights_batch = []
        for b in range(batch_size):
            if self.attention_type == 'first_hop':
                attention_weights_batch.append([adj_matrix[b]])
            elif self.attention_type == 'multi_hop':
                attention_weights_batch.append(self.compute_multi_hop_weights(adj_matrix[b]))
            elif self.attention_type == 'diffusion':
                attention_weights_batch.append([self.compute_diffusion_matrix(adj_matrix[b])])
            else:
                attention_weights_batch.append([adj_matrix[b]]) # Default

        all_attn_maps = []
        for layer in self.layers:
            h, scores = layer(h, attention_weights_batch)
            all_attn_maps.append(scores)

        # Self-supervision prediction (only if mask provided, else return None for preds)
        if mask_edges is not None:
            pred_weights = self.edge_weight_predictor(
                torch.cat([
                    h.unsqueeze(2).repeat(1, 1, self.num_nodes, 1), 
                    h.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
                ], dim=-1)
            ).squeeze(-1)
        else:
            pred_weights = None

        return h, pred_weights, all_attn_maps
