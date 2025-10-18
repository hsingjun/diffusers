import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
    def forward(self, x):
        #print(f"<><><><> dtype of x at the start of MLP forward: {x.dtype}") # dtype of x at the start of MLP forward: torch.float32
        return self.net(x)

class ProteinDeepSetCLIPHead(nn.Module):
    """
    Input:  variable-length set of protein embeddings (B, N, 1024)
    Output:
      - last_hidden_state: (B, 77, 768)  # matches CLIP text encoder used by SD v1.x
      - logits:            (B, 368)      # multi-class species prediction
    """
    def __init__(self,
                 in_dim=1024,
                 phi_hidden=1024,
                 out_dim=768,
                 num_tokens=77,
                 num_classes=368,
                 p_drop=0.1):
        super().__init__()

        # Per-element feature mapper φ: 1024 -> 768
        self.phi = MLP(in_dim, phi_hidden, out_dim, p_drop=p_drop)

        # Learned token queries (77, 768) that pull information from the set
        # via permutation-invariant attention pooling.
        self.token_queries = nn.Parameter(torch.randn(num_tokens, out_dim) / math.sqrt(out_dim))

        # Final LayerNorm (mirrors CLIPTextModel’s final_layer_norm)
        self.final_layer_norm = nn.LayerNorm(out_dim)

        # Classifier (e.g., use token 0 or mean over tokens; we choose mean for stability)
        self.classifier = nn.Linear(out_dim, num_classes)

        # optional dropout on tokens before classification
        self.cls_drop = nn.Dropout(p_drop)

        self.scale = 1.0 / math.sqrt(out_dim)

    def forward(self, x, mask=None, return_tokens=True):
        """
        x:    (B, N, 1024) protein embeddings (ProtTrans)
        mask: (B, N) with 1 for valid, 0 for pad (optional; if None, all valid)

        Returns:
          last_hidden_state: (B, 77, 768)
          logits:            (B, 368)
        """
        
        #print(f"<><><><> dtype of x at the start of forward: {x.dtype}") # dtype of x at the start of forward: torch.float32
        x = x.to(torch.bfloat16)
        B, N, _ = x.shape

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

        # φ maps each protein embedding to 768-d token space (perm-invariant step is later)
        z = self.phi(x)                         # (B, N, 768)

        # Attention-based Deep Sets pooling (still permutation-invariant)
        # For each of 77 learned queries, compute weights over the N elements.
        Q = self.token_queries                 # (77, 768)
        Q = Q.unsqueeze(0).expand(B, -1, -1)   # (B, 77, 768)

        # attn_logits[b, t, n] = <Q[b,t], z[b,n]> * scale
        attn_logits = torch.einsum('btd,bnd->btn', Q, z) * self.scale  # (B, 77, N)

        # mask invalid positions
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # softmax over set dimension (N)
        attn_weights = attn_logits.softmax(dim=-1)                     # (B, 77, N)

        # weighted sum to produce 77 token embeddings
        tokens = torch.einsum('btn,bnd->btd', attn_weights, z)         # (B, 77, 768)

        # CLIP-style final LayerNorm
        last_hidden_state = self.final_layer_norm(tokens)              # (B, 77, 768)

        # Simple pooled representation for classification
        pooled = last_hidden_state.mean(dim=1)                         # (B, 768)
        pooled = self.cls_drop(pooled)
        logits = self.classifier(pooled)                               # (B, 368)

        if return_tokens:
            return last_hidden_state, logits
        return logits

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Suppose each protein embedding from ProtTrans is 1024-d
    prot_dim = 1024
    num_proteins = 3000   # e.g., human proteome tokens
    batch_size = 4

    # Create random "ProtTrans" embeddings for demo (B, N, prot_dim)
    x = torch.randn(batch_size, num_proteins, prot_dim).to(device) #.to(torch.bfloat16)

    # Create model: map proteome -> 128-d vector
    model = ProteinDeepSetCLIPHead(
            in_dim=prot_dim,
            phi_hidden=1024,
            out_dim=768,
            num_tokens=77,
            num_classes=368,
            p_drop=0.1
    ).to(device)  # Move model to GPU

    mask = torch.ones(batch_size, num_proteins, dtype=torch.bool).to(device)
    tokens, logits = model(x, mask=mask)
    print(tokens.shape)
    print(logits.shape)
