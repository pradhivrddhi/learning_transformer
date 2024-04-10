import torch


from .encoder_decoder_attention_head import EncoderDecoderAttentionHead

class EncoderDecoderAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        
        self.heads = torch.nn.ModuleList(
            [
                EncoderDecoderAttentionHead(embed_dim, head_dim)
                for _ in range(num_heads)
            ]
        )
        
        self.output_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, decoder_hidden_state, encoder_hidden_state):
        x = torch.cat([h(decoder_hidden_state, encoder_hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
