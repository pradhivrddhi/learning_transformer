import torch

from .scaled_dot_product_attention import scaled_dot_product_attention

class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = torch.nn.Linear(embed_dim, head_dim)
        self.k = torch.nn.Linear(embed_dim, head_dim)
        self.v = torch.nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )
        return attn_outputs
