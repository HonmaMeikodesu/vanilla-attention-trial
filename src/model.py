import math

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
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X: torch.Tensor = self.emb(X)
        # X (batch_size, num_steps, embed_size)
        X = X.permute(1, 0, 2)
        # X (num_steps, batch_size, embed_size)
        output, hidden_state = self.rnn(X)
        output = output.permute(1, 0 , 2)
        # output (batch_size, num_steps, num_hidden)
        # hidden_state (num_layers, batch_size, num_hidden)
        return output, hidden_state

class Seq2SeqDecoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.emb = nn.Embedding(vocab_size, embed_size) # 用于强制教学
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
        
        output, hidden_state = self.rnn(X_concat_context)
        # output (num_steps, batch_size, num_hidden)
        # hidden_state (num_layers, batch_size, num_hidden)

        final_output = self.ffn(output).permute(1, 0, 2)
        # final_output (batch_size, num_steps, vocal_size)

        return final_output, hidden_state
        

        

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_X_valid_len, dec_X_valid_len, *args):
        enc_outputs = self.encoder(enc_X)
        dec_init_state = self.decoder.init_state(enc_outputs, enc_X_valid_len)
        dec_outputs = self.decoder(dec_X, dec_init_state)
        return dec_outputs
    

class DotProductAttention(nn.Module):
    """DotProduct attention.

    Defined in :numref:`sec_attention-scoring-functions`"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, queries, keys, values, valid_lens):
        # queries (batch_size, 1, num_hidden)
        # keys (batch_size, num_steps, num_hidden)
        # values (batch_size, num_steps, num_hidden)
        scores = torch.bmm(queries, keys.permute(0, 2, 1)) / math.sqrt(queries.shape[-1])
        scores = scores.squeeze(dim=1)
        # scores (batch_size, num_steps)
        attention_weights = masked_softmax(scores, valid_lens)
        attention_weights = attention_weights.unsqueeze(dim=1)
        # attention_weights (batch_size, 1, num_steps)
        result = torch.bmm(self.dropout(attention_weights), values)
        # result (batch, 1, num_hidden)
        return result


class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = DotProductAttention(dropout)
        self.emb = nn.Embedding(vocab_size, embed_size) # 用于强制教学
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.ffn = nn.Linear(num_hiddens, vocab_size) # 最终决策层

    def init_state(self, enc_outputs, enc_valid_len, *args):
        enc_output, hidden_state = enc_outputs
        return enc_output, hidden_state, enc_valid_len

    def forward(self, X, state):
        X: torch.Tensor = self.emb(X)
        X = X.permute(1, 0, 2)
        enc_output, hidden_state, enc_valid_len = state
        # X (num_steps, batch_size, embed_size)
        # enc_context (batch_size, num_hidden)
        final_output: torch.Tensor = None
        for x in X:
            x = x.unsqueeze(dim=1)
            # x (batch_size, 1, embed_size)
            hidden_state = hidden_state[-1].unsqueeze(dim=1)
            # hidden_state (batch_size, 1, num_hidden)
            attention_context = self.attention(hidden_state, enc_output, enc_output, enc_valid_len)
            # attention_context (batch_size, 1, num_hidden)
            x_concat_context = torch.transpose(torch.concat([x, attention_context], dim=2), 0, 1)
            # x_concat_context (1, batch_size, embed_size + num_hidden)
            output, hidden_state = self.rnn(x_concat_context)
            # output (1, batch_size, num_hidden)
            # hidden_state (num_layers, batch_size, num_hidden)
            current_output = self.ffn(output)
            # current_output (1, batch_size, vocal_size)
            if final_output is None:
                final_output = current_output
            else:
                final_output = torch.concat([final_output, current_output], dim=0)
            # final_output (num_steps, batch_size, vocal_size)

        return final_output.transpose(0, 1), hidden_state
        # final_output (batch_size, num_steps, vocal_size)