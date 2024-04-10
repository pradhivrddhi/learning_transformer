import torch



from .embeddings import Embeddings
from .decoder_only_layer import DecoderOnlyLayer


class DecoderOnly(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = torch.nn.ModuleList([
            DecoderOnlyLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, targets):
        decoder_output = self.embeddings(targets)
        for layer in self.layers:
            decoder_output = layer(decoder_output)
        decoder_output = self.linear(decoder_output)
        return decoder_output

