import d2l.torch as d2l
import torch
import torch.nn as nn

from utils import masked_softmax


#@save
class Seq2SeqEncoder(nn.Module):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.emb = nn.Embedding(vocab_size, num_hiddens)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X (batch_size, num_steps, vocab)
        X: torch.Tensor = self.emb(X)
        # X (batch_size, num_steps, embed_size)
        X = X.permute(1, 0, 2)
        # X (num_steps, batch_size, embed_size)
        output, hidden_state = self.rnn(X)
        # output (batch_size, num_steps, num_hidden)
        # hidden_state (num_layers, batch_size, num_hidden)
        return output, hidden_state

class Seq2SeqDecoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.emb = nn.Embedding(vocab_size, num_hiddens) # 用于强制教学
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.ffn = nn.Linear(num_hiddens, vocab_size) # 最终决策层

    def init_state(self, enc_outputs, *args):
        __, hidden_state = enc_outputs
        return hidden_state

    def forward(self, X, state):
        X: torch.Tensor = self.emb(X)
        X = X.permute(1, 0, 2)
        enc_context: torch.Tensor = state[-1]
        # X (num_steps, batch_size, embed_size)
        # enc_context (batch_size, num_hidden)
        enc_context = enc_context.repeat(X.shape[0], 1, 1) # 广播维度，为了与X的embed_size维进行concat
        # enc_context (num_steps, batch_size, num_hidden)
        X_concat_context = torch.concat([X, enc_context], dim=2)
        # X_concat_context (num_steps, batch_size, num_hidden + embed_size)
        
        output, __ = self.rnn(X_concat_context)
        # output (batch_size, num_steps, num_hidden)
        # hidden_state (num_layers, batch_size, num_hidden)

        final_output = self.ffn(output).permute(1, 0, 2)
        # final_output (batch_size, num_steps, vocal_size)

        return final_output
        

        

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X)
        dec_init_state = self.decoder.init_state(enc_outputs)
        dec_outputs = self.decoder(dec_X, dec_init_state)
        return dec_outputs
    

class AdditiveAttention(nn.Module):
    """Additive attention.

    Defined in :numref:`sec_attention-scoring-functions`"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        pass

    def forward(self, queries, keys, values, valid_lens):
        pass

class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        pass

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        pass

    def forward(self, X, state):
        pass