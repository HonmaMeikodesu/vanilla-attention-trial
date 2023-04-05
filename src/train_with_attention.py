import d2l.torch as d2l

from model import EncoderDecoder, Seq2SeqAttentionDecoder, Seq2SeqEncoder
from train_seq2seq import train_seq2seq

batch_size = 20

emb_size = 64

num_steps = 10

num_hiddens = 64

num_epoch = 400

lr = 0.01

device = d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
enc = Seq2SeqEncoder(len(src_vocab), emb_size, num_hiddens, 2, 0.1)
dec = Seq2SeqAttentionDecoder(len(tgt_vocab), emb_size, num_hiddens, 2, 0.1)
net = EncoderDecoder(enc, dec)

train_seq2seq(net, train_iter, lr, num_epoch, src_vocab, tgt_vocab, device, num_steps)