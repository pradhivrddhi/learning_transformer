import torch


from .masked_multi_head_attention import MaskedMultiHeadAttention
from .encoder_decoder_attention import EncoderDecoderAttention
from .feed_forward import FeedForward

class DecoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm_2_encoder = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm_2_decoder = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm_3 = torch.nn.LayerNorm(config.hidden_size)
        self.attention = MaskedMultiHeadAttention(config)
        self.encoder_decoder_attention = EncoderDecoderAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, decoder_output, encoder_output):
        # Apply attention with a skip connection
        hidden_state = decoder_output + self.attention(self.layer_norm_1(decoder_output))
        # Apply encoder decoder attention with a skip connection and normalization
        hidden_state = hidden_state + self.encoder_decoder_attention(self.layer_norm_2_decoder(hidden_state), self.layer_norm_2_encoder(encoder_output))
        # Apply feed-forward layer with a skip connection and normalization
        hidden_state = hidden_state + self.feed_forward(self.layer_norm_3(hidden_state))
        return hidden_state