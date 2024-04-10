from transformers import BertModel

class FetchFromPretrained:
    def __init__(self, model_ckpt):
        self.model_ckpt = model_ckpt

    def fetch(self):
        model = BertModel.from_pretrained(self.model_ckpt)
        return model
