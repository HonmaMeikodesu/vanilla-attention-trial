import torch
import torch.nn as nn

from utils import sequence_mask


#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        # pred (batch_size, num_steps, vocab_size)
        # label (batch_size, num_steps)
        # valid_len(batch_size)
        pred = pred.permute(0, 2, 1)
        # pred (batch_size, num_of_classes, multiple dimension of loss)
        # label (batch_size, multiple dimension of loss)
        self.reduction = "none"
        # 阻止CrossEntropyLoss进行loss聚合
        unmasked_loss = super(MaskedSoftmaxCELoss, self).forward(pred, label)
        # unmasked_loss (batch_size, multiple dimension of loss)
        mask = sequence_mask(unmasked_loss, valid_len)
        return (unmasked_loss * mask).mean(dim=1)
        # return value (batch_size) 自带squeeze效果