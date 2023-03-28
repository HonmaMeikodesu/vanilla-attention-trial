import torch
import torch.nn as nn


#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # X (batch_size, num_steps)
    # valid_len (batch_size)
    max_len = torch.arange(X.shape[1], dtype=float, device=X.device)[None, :]
    # max_len (1, num_steps) with a sequence of (1,2,3,4,.....)
    col = valid_len[:, None]
    # col (batch_size, 1)
    mask = max_len < col
    # mask (batch_size, num_steps) with boolean
    result = torch.ones_like(X)
    result[~mask] = value
    return result

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    pass