import torch
import torch.nn as nn

#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # X (batch_size, num_steps)
    # valid_len (batch_size, valid_length_int)
    result = torch.ones_like(X)
    for index, _ in enumerate(X):
        result[index][~valid_len[index]] = value
        index += 1
    return result

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    pass