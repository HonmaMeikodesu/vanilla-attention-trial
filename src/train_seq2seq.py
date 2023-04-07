
import d2l.torch as d2l
import torch
import torch.nn as nn

from infer import predict_seq2seq
from loss import MaskedSoftmaxCELoss


#@save
def train_seq2seq(net, data_iter, lr, num_epochs, src_vocab, tgt_vocab, device, num_steps):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()

    for x in range(num_epochs):
            net.train()
            total_loss = 0
            for batch in data_iter:
                optimizer.zero_grad()
                X, X_valid_len, Y, Y_valid_len = (x.to(device) for x in batch)
                # X, Y (batch_size, vocab_size)
                # X_valid_len, Y_valid_len (batch_size, integer)
                bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1).to(device)
                # bos (batch_size, 1)
                dec_input = torch.concat([bos, Y[:, :-1]], dim=1)
                output, __ = net(X, dec_input, X_valid_len, Y_valid_len)
                l = loss(output, Y, Y_valid_len)
                with torch.no_grad():
                    total_loss += l.sum().data
                l.sum().backward()
                d2l.grad_clipping(net, 1)
                optimizer.step()
            if (x + 1) % 10 == 0:
                print(f"epoch: {x + 1}, loss {total_loss}")
                engs = ['go on .', "i lost .", 'he\'s calm .', 'i\'m home .']
                fras = ['poursuis .', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
                for eng, fra in zip(engs, fras):
                    translation = predict_seq2seq(
                        net, eng, src_vocab, tgt_vocab, torch.LongTensor([3]).to(device), num_steps, device)
                    if len(translation.split(" ")) == 1:
                        arr = translation.split(" ")
                        arr.append(".")
                        translation = " ".join(arr)
                    print(f'{eng} => {translation}, ',
                        f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
