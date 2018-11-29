import config
import numpy as np
import torch
import torch.nn.functional as F

class beam_search():
    def __init__(self, encoder, decoder, max_length, beam_size, attention = False):
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
        
        
    def search(self, encoder_outputs, decoder_input, decoder_hidden, decoder_cell_state):
        """
        Args:
            encoder_output: output of encoder, used for attention. shape: 1 x 1 x hidden_size
            decoder_input: SOS token (e.g. torch.tensor([[SOS_token]], device=device))
            decoder_hidden: last encoder hidden vector. 
            decoder_cell_state: last encoder cell state.
        """
        decoder_input_cand = {}
        decoder_output_cand = {}
        decoder_hidden_cand = {}
        decoder_cell_state_cand = {}
        decoded_words_cand = {k:[] for k in range(self.beam_size)}
        decoded_sentences_prob = {k:0 for k in range(self.beam_size)} #JP: create decoded_sentences_prob
        final_sent = []
        final_score = []
        
        ## INIT
        if self.attention == True:
            #decoder_attn = torch.zeros(self.max_length, self.max_length) #JP: this line is actually unnecessary
          
            #decoder_output, decoder_attn, decoder_hidden, decoder_cell_state = self.decoder(decoder_hidden, decoder_cell_state, decoder_input, encoder_outputs)
            decoder_output, decoder_attn, decoder_hidden, decoder_cell_state = self.decoder(decoder_hidden.contiguous(), decoder_cell_state, decoder_input.contiguous(), encoder_outputs)
        else: 
            decoder_output, decoder_hidden, decoder_cell_state = self.decoder(decoder_hidden, decoder_cell_state, decoder_input)
            
        decoder_output = F.log_softmax(decoder_output, dim=1)
        topv, topi = decoder_output.data.topk(self.beam_size)
        for i in range(self.beam_size):
            decoded_words_cand[i].append(topi.squeeze()[i].item())
            decoder_input_cand[i] = topi.squeeze()[i].detach()
            decoder_hidden_cand[i] = decoder_hidden
            decoder_cell_state_cand[i] = decoder_cell_state
            decoded_sentences_prob[i] += topv.squeeze()[i].detach() #JP: calculate log probability (multiplication becomes addition)
            
        ## BEAM-SEARCH
        word_cnt = 0
        while (bool(decoder_hidden_cand)) & (word_cnt <= self.max_length):
            word_cnt += 1
            topi = {}
            avail_keys = list(decoder_hidden_cand.keys())
            score_all = []
            for b in avail_keys:
                if self.attention == True:
                    decoder_output, decoder_attn, decoder_hidden_cand[b], decoder_cell_state_cand[b] = self.decoder(decoder_hidden_cand[b], decoder_cell_state_cand[b], decoder_input_cand[b].unsqueeze(0),  encoder_outputs)
                    decoder_output_cand[b] = F.log_softmax(decoder_output, dim=1)
                else:
                    decoder_output, decoder_hidden_cand[b], decoder_cell_state_cand[b] = self.decoder(decoder_hidden_cand[b], decoder_cell_state_cand[b], decoder_input_cand[b])
                    decoder_output_cand[b] = F.log_softmax(decoder_output, dim=1)
                
                topv, topi[b] = decoder_output_cand[b].data.topk(len(decoder_hidden_cand))
                score_all.extend((topv+decoded_sentences_prob[b]).tolist()[0]) #JP: multiply (add in log) conditional probability topv to previous ones
                
            score_all = np.array(score_all)   
            max_cand = score_all.argsort()[-len(decoder_hidden_cand):][::-1]
            decoded_sent_score = score_all[max_cand]

            cand_sentences = {}
            cand_hiddens = {}
            cand_cell_states = {}
            keys_to_rm = []
            
            for j in range(len(max_cand)):
                prev_cand_id = avail_keys[int(np.floor(max_cand[j]/len(decoder_hidden_cand)))]
                if topi[prev_cand_id].squeeze().dim() == 0:
                    next_id = topi[prev_cand_id].squeeze()
                else:
                    next_id = topi[prev_cand_id].squeeze()[max_cand[j] % len(decoder_hidden_cand)]
                s_cand = decoded_words_cand[prev_cand_id].copy()
                s_cand.append(next_id.item())
                cand_sentences[j] = s_cand
                
                h_cand = decoder_hidden_cand[prev_cand_id]
                cand_hiddens[j] = h_cand
                
                decoder_input_cand[j] = next_id.detach()   
                
                c_cand = decoder_cell_state_cand[prev_cand_id]
                cand_cell_states[j] = c_cand

                decoded_sentences_prob[j] = decoded_sent_score[j] # update decoded_sentences_prob

                
            decoded_words_cand = cand_sentences
            decoder_hidden_cand = cand_hiddens
            decoder_cell_state_cand = cand_cell_states
            
            #print(decoded_sentences_prob)
            for key, s in decoded_words_cand.items():
                if config.EOS_TOKEN in s:
                    final_sent.append(s)
                    #final_score.append(decoded_sent_score[key])
                    final_score.append(decoded_sentences_prob[key]) #JP: use the joint probability. actually, same as using decoded_sent_score..
                    keys_to_rm.append(key)
                    
            for k in keys_to_rm:
                decoder_hidden_cand.pop(k)
                decoded_words_cand.pop(k)
                decoder_cell_state_cand.pop(k)

        if len(final_score) == 0:
            max_prob = 0
            max_prob_id = None
            for k in decoded_sentences_prob.keys():
                if decoded_sentences_prob[k] > max_prob: 
                    max_prob_id = k
                    max_prob = decoded_sentences_prob[k]
            final_score.append(max_prob)
            final_sent.append(decoded_words_cand[max_prob_id])
                
        return final_sent, final_score
