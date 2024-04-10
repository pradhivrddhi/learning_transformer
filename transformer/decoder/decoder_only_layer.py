import torch


from .masked_multi_head_attention import MaskedMultiHeadAttention
from .feed_forward import FeedForward

class DecoderOnlyLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = torch.nn.LayerNorm(config.hidden_size)
        self.attention = MaskedMultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, decoder_output):
        # Apply attention with a skip connection
        hidden_state = decoder_output + self.attention(self.layer_norm_1(decoder_output))
        # Apply feed-forward layer with a skip connection and normalization
        hidden_state = hidden_state + self.feed_forward(self.layer_norm_2(hidden_state))
        return hidden_state
