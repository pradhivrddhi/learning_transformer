from transformers import AutoTokenizer

class FetchFromPretrained:
    def __init__(self, model_ckpt):
        self.model_ckpt = model_ckpt

    def fetch(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        return tokenizer
