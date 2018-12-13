import string
def detok(ind_input, ind2word, remove_punc=True):
    '''
    Turn indices back to string
    
    Arg:
    ind_input: Each column is a sentence, element value is word indices. 
               shape = (sequence length * batch size)
    in2word: a numpy array that match ind back to word. len = # of vocab
    '''
    detok_output = []
    for i in range(ind_input.shape[1]):
        tok_trg = ind_input[:,i]
        tok_trg = tok_trg[(tok_trg>3) | (tok_trg==0)] #remove padding, SOS, EOS
        items_list = ind2word[tok_trg]
        
        if len(items_list) == 0:
            # not sure if this case ever happens yet.
            detok_str = ""
        elif isinstance(items_list, str):
            # sometimes items_list is just one string. return it.
            if items_list in string.punctuation:
                detok_str = ""
            else:
                detok_str = items_list
        else:
            # if it is multi-word list, then join them as a string
            if remove_punc:
                detok_str = " ".join([s for s in items_list if s not in string.punctuation])
            else:
                detok_str = " ".join(items_list)
        detok_output.append(detok_str)
    return detok_output
        
