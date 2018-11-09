import sacrebleu
from detok import detok
import numpy as np

def bleu(itos, translation_output, reference):
    '''
    Args:
        trg.vocab.itos: a list the match indices to string.
        translation_output: 2D tensor of tranlation output. shape: N x B
        reference: 1D list of reference sentences (words, not indices). len(reference) = B
    '''
    EN_ind2word = np.array(itos)
    detok_translation = detok(translation_output, EN_ind2word)
    bleu_score = sacrebleu.raw_corpus_bleu(detok_translation, [reference], .01).score
    

    return bleu_score