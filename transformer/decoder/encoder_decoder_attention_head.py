import torch


from .scaled_dot_product_attention import scaled_dot_product_attention

class EncoderDecoderAttentionHead(torch.nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = torch.nn.Linear(embed_dim, head_dim)
        self.k = torch.nn.Linear(embed_dim, head_dim)
        self.v = torch.nn.Linear(embed_dim, head_dim)
        self.register_buffer('tril', torch.tril(torch.ones(embed_dim*8, embed_dim*8)))


    def forward(self, decoder_hidden_state, encoder_hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(decoder_hidden_state), self.k(encoder_hidden_state), self.v(encoder_hidden_state),
            mask = self.tril,
        )
        return attn_outputs
