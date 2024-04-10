import torch


from ..encoder.encoder import Encoder

from .embeddings import Embeddings
from .decoder_layer import DecoderLayer


class Decoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.layers = torch.nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, targets, query):
        decoder_output = self.embeddings(targets)
        encoder_output = self.encoder(query)
        for layer in self.layers:
            decoder_output = layer(decoder_output, encoder_output)
        decoder_output = self.linear(decoder_output)
        return decoder_output
