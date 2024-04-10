from math import sqrt

import torch



def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask[:scores.size()[-2], :scores.size()[-1]] == 0, float('-inf'))
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return weights.bmm(value)
