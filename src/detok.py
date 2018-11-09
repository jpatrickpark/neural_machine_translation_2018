def detok(ind_input, ind2word):
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
        detok_str = " ".join(ind2word[tok_trg])
        detok_output.append(detok_str)
    return detok_output
        