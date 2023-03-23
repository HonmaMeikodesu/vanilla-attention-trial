import torch
import torch.nn as nn
from utils import sequence_mask

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        # pred (batch_size, num_steps, vocab_size)
        # label (batch_size, num_steps)
        # valid_len(batch_size, int)
        pass