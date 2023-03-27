import d2l.torch as d2l
import torch
import torch.nn as nn

from model import EncoderDecoder, Seq2SeqDecoder, Seq2SeqEncoder
from loss import MaskedSoftmaxCELoss


#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
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
    net.train()

    for x in range(num_epochs):
            for batch in data_iter:
                optimizer.zero_grad()
                X, X_valid_len, Y, Y_valid_len = (x.to(device) for x in batch)
                # X, Y (batch_size, vocab_size)
                # X_valid_len, Y_valid_len (batch_size, integer)
                bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1).to(device)
                # bos (batch_size, 1)
                dec_input = torch.concat([bos, Y[:, :-1]], dim=1)
                output = net(X, dec_input)
                # output ()
                output
                l = loss(output, Y, Y_valid_len)
                total_loss = l.sum()
                total_loss.backward()
                d2l.grad_clipping(net, 1)
                optimizer.step()
                with torch.no_grad():
                    if (x + 1) % 10 == 0:
                        print(f"loss {total_loss}")

batch_size = 20

emb_size = 64

num_steps = 10

num_hiddens = 64

num_epoch = 400

lr = 0.01

device = d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
enc = Seq2SeqEncoder(len(src_vocab), emb_size, num_hiddens, 2, 0.1)
dec = Seq2SeqDecoder(len(tgt_vocab), emb_size, num_hiddens, 2, 0.1)
net = EncoderDecoder(enc, dec)

train_seq2seq(net, train_iter, lr, num_epoch, tgt_vocab, device)