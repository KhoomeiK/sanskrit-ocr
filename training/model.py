"""CNN patch embed + causal Transformer decoder using native PyTorch MultiheadAttention."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from data import PATCH

IN_CH  = 3   # RGB
EMBED_DIM = 768

class PatchEmbeddingCNN(nn.Module):
    """
    Run a tiny ConvNet on each (3 x PATCH x PATCH) patch separately,
    producing one EMBED_DIM vector per patch.
    """
    def __init__(self,
                 in_ch: int = IN_CH,
                 patch_size: int = PATCH,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = 512):
        super().__init__()
        self.P = patch_size
        # A simple 3‐layer ConvNet that ends in a 1×1 feature map:
        self.conv = nn.Sequential(
            # [B*L, in_ch, P, P] → [B*L, hidden_dim, P, P]
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # [B*L, hidden_dim, P, P] → [B*L, hidden_dim, P/2, P/2]
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # [B*L, hidden_dim, P/2, P/2] → [B*L, embed_dim, 1, 1]
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=patch_size // 2, stride=patch_size // 2),
            nn.GELU(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: [B, L, D] where
          L = (H/PATCH)*(W/PATCH) = # of patches per image
          D = in_ch * PATCH * PATCH

        returns: [B, L, embed_dim]
        """
        b, L, D = patches.shape
        P = self.P
        C = D // (P * P)
        # 1) Treat each vector as its own image:
        x = patches.view(b * L, C, P, P)         # → [B*L, in_ch, P, P]
        # 2) Run the ConvNet on each mini‐image:
        x = self.conv(x)                         # → [B*L, embed_dim, 1, 1]
        # 3) Squeeze spatial dims and reshape back:
        x = x.view(b, L, -1)                     # → [B, L, embed_dim]
        return x

class CausalSelfAttention(nn.Module):
    """Causal self-attention using nn.MultiheadAttention."""
    def __init__(self, dim, n_head):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=n_head, batch_first=True)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # print(f'{x.shape=} {causal_mask.shape=} {padding_mask.shape=}')
        attn_out, _ = self.mha(x, x, x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        return attn_out
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_head, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class CausalTransformer(nn.Module):
    def __init__(self, dim=768, depth=12, n_head=12):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_head) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, padding_mask)
        return self.norm(x)

class MultiModalLM(nn.Module):
    def __init__(self, vocab_size: int, dim=768, depth=12, n_head=12):
        super().__init__()
        self.patch_cnn = PatchEmbeddingCNN(in_ch=3, embed_dim=dim)
        self.txt_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, dim))
        self.tr = CausalTransformer(dim, depth, n_head)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, patches: torch.Tensor, txt_ids: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        img_tok = self.patch_cnn(patches)
        txt_tok = self.txt_embed(txt_ids)
        x = torch.cat([img_tok, txt_tok], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        h = self.tr(x, padding_mask)
        return self.head(h[:, -txt_ids.size(1):])