from transformers import AutoConfig

class FetchFromPretrained:
    def __init__(self, model_ckpt):
        self.model_ckpt = model_ckpt

    def fetch(self):
        config = AutoConfig.from_pretrained(self.model_ckpt)
        return config
