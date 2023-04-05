import d2l.torch as d2l
import torch


#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, enc_valid_len, num_steps,
                    device):
    """序列到序列模型的预测"""
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(" ")] + [src_vocab["<eos>"]]
    # src_tokens (sentence_length)
    src_tokens = torch.tensor(d2l.truncate_pad(src_tokens, num_steps, src_vocab["<pad>"]), device=device).type(torch.long).unsqueeze(dim=0)
    # src_tokens (batch_size=1, num_steps)
    enc_outputs = net.encoder(src_tokens)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    
    tgt_tokens = torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device).unsqueeze(dim=0)
    # tgt_tokens (batch_size=1, num_steps=1)
    output_sentences = []
    for _ in range(num_steps):
        dec_output, dec_state = net.decoder(tgt_tokens, dec_state)
        # dec_output (batch_size=1, num_steps=1, vocal_size)
        tgt_tokens = torch.argmax(dec_output, dim=2)
        # tgt_tokens (batch_size=1, num_steps=1)
        Y = tgt_tokens.squeeze(dim=0)
        if tgt_vocab["<eos>"] == Y.item():
            break
        output_sentences.append(tgt_vocab.idx_to_token[Y.item()])
    print(" ".join(output_sentences))

