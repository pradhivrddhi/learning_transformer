import torch


from ..encoder.encoder import Encoder

class SequenceClassification(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :] # select the hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x