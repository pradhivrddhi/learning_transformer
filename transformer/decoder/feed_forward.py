import torch


class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
