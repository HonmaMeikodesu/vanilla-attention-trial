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
        pred = pred.permute(0, 2, 1)
        # pred (batch_size, num_of_classes, multiple dimension of loss)
        # label (batch_size)
        self.reduction = "none"
        # 阻止CrossEntropyLoss对batch_size维度进行loss聚合
        unmasked_loss = super(MaskedSoftmaxCELoss).forward(pred, label)
        # unmasked_loss (batch_size, multiple dimension of loss)