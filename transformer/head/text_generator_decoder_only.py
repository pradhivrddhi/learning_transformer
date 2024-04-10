import torch


from copy import deepcopy

from ..decoder.decoder_only import DecoderOnly

class TextGenerator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.block_size = config.max_position_embeddings

        self.decoder = DecoderOnly(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)


    def forward(self, query, targets=None, tokenizer=None):
        if targets is None:
            output = self.decoder.to(self.device).forward(query)
            logits = self.lm_head.to(self.device)(output)
            loss = None
        else:
            output = self.decoder.to(self.device).forward(query)
            logits = self.lm_head.to(self.device)(output)
            B, T, C = logits.shape
            
            loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to obtain probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1).to(self.device) # (B, C)
            index_next = torch.multinomial(probs, num_samples=1) # (B, C)

            if tokenizer is None:
                print(index_next.squeeze(-1))
                print(targets[:, -1])
            else:
                print(tokenizer.decode(index_next.squeeze(-1)))
                print(tokenizer.decode(targets[:, -1]))

        return logits, loss

    def generate(self, index, max_new_tokens):
        with torch.no_grad():
            iteration = dict()
            for _ in range(max_new_tokens):
                index_cond = index[:, -self.block_size:]
                iteration['input'] = index_cond[0]

                # get the predictions
                logits, _ = self.forward(index_cond.to(self.device))

                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)

                # apply softmax to obtain probabilities
                probs = torch.nn.functional.softmax(logits.cpu(), dim=-1).to(self.device) # (B, C)

                # sample from the distribution
                index_next = torch.multinomial(probs, num_samples=1).to(self.device) # (B, C)

                # append sampled index to the running sequence
                index = torch.cat((index.to(self.device), index_next.to(self.device)), dim=1) # (B, T+1)
            
            return index
