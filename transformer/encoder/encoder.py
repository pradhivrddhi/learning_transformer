import torch


from .embeddings import Embeddings
from .encoder_layer import EncoderLayer


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = torch.nn.ModuleList([
            EncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x