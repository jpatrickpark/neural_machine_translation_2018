import numpy as np
import torch
class beam_search():
    def __init__(self, encoder, decoder, attention, max_length, beam_size):
        """
        Args:
            encoder: the encoder network
            decoder: the decoder network
            attention: boolean. True if using attention
            max_length: int. max sentence length produced
            beam_size: int.
        """    
        super(beam_search, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.max_length = max_length
        self.beam_size = beam_size
        
        
    def search(self, encoder_output, decoder_hidden):
    """
    Args:
        encoder_output: output of encoder, used for attention. shape: 1 x 1 x hidden_size
        decoder_hidden: last encoder hidden vector. 
    """
    decoder_input_cand = {}
    decoder_output_cand = {}
    decoder_hidden_cand = {}
    decoded_words_cand = {k:[] for k in range(beam_size)}
    final_sent = []
    final_score = []
    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    
    ## INIT
    if self.attention == True:
        decoder_attentions = torch.zeros(max_length, max_length)
        
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
    else: 
        
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
    topv, topi = decoder_output.data.topk(beam_size)
    for i in range(beam_size):
        decoded_words_cand[i].append(output_lang.index2word[topi.squeeze()[i].item()])
        decoder_input_cand[i] = topi.squeeze()[i].detach()
        decoder_hidden_cand[i] = decoder_hidden
        
    ## BEAM-SEARCH
    word_cnt = 0
    while (bool(decoder_hidden_cand)) & (word_cnt <= max_length):
        word_cnt += 1
        topi = {}
        avail_keys = list(decoder_hidden_cand.keys())
        for b in avail_keys:
            if self.attention == True:
                decoder_output_cand[b], decoder_hidden_cand[b], decoder_attention = decoder(decoder_input_cand[b], 
                                                                                 decoder_hidden_cand[b], 
                                                                                 encoder_outputs)
            else:
                decoder_output_cand[b], decoder_hidden_cand[b] = decoder(decoder_input_cand[b], 
                                                                                 decoder_hidden_cand[b])
            
            topv, topi[b] = decoder_output_cand[b].data.topk(beam_size)

            max_cand = score_all.argsort()[-beam_size:][::-1]
            decoded_sent_score = score_all[max_cand]
            #print(topv, topi[b], decoder_output_cand[b])

        cand_sentences = {}
        cand_hiddens = {}
        keys_to_rm = []
        for j in range(len(max_cand)):
            prev_cand_id = avail_keys[int(np.floor(max_cand[j]/beam_size))]

            next_id = topi[prev_cand_id].squeeze()[max_cand[j] % beam_size]
            s_cand = decoded_words_cand[prev_cand_id].copy()
            s_cand.append(output_lang.index2word[next_id.item()])
            cand_sentences[j] = s_cand
            h_cand = decoder_hidden_cand[prev_cand_id]
            cand_hiddens[j] = h_cand
            decoder_input_cand[j] = next_id.detach()    
        decoded_words_cand = cand_sentences
        decoder_hidden_cand = cand_hiddens
        for key, s in decoded_words_cand.items():
            if 'EOS' in s:
                final_sent.append(s)
                final_score.append(decoded_sent_score[key])
                keys_to_rm.append(key)
        for k in keys_to_rm:
            decoder_hidden_cand.pop(k)
            decoded_words_cand.pop(k)
    
    return final_sent, final_score
