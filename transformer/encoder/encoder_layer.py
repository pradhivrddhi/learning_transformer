import torch


from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward

class EncoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = torch.nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection and normalization
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x