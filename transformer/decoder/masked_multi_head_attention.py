import torch

from .attention_head import AttentionHead

class MaskedMultiHeadAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        
        self.heads = torch.nn.ModuleList(
            [
                AttentionHead(config)
                for _ in range(num_heads)
            ]
        )
        
        self.output_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

