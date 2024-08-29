import torch

from .scaled_dot_product_attention import scaled_dot_product_attention

class AttentionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads

        self.q = torch.nn.Linear(embed_dim, head_dim)
        self.k = torch.nn.Linear(embed_dim, head_dim)
        self.v = torch.nn.Linear(embed_dim, head_dim)
        self.register_buffer('tril', torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings)))

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state),
            mask = self.tril,
        )
        return attn_outputs
