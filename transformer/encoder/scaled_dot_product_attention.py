from math import sqrt

import torch



def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.bmm(weights, value)