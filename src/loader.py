from torchtext import data
from torchtext import datasets

import re
import nltk
import jieba
from functools import partial
import config

from nltk.tokenize.toktok import ToktokTokenizer

def chinese_tokenizer(line):
    # Currently using jieba to tokenize Chinese
    # TODO: use Stanford tokenizer on prince
    return list(jieba.cut(line, cut_all=False))

def load_chinese_english_data(data_dir, njobs, split_chinese_into_characters=False):
    '''
    Loads the following files:
        data_dir/train.cn, data_dir/train.en,
        data_dir/val.cn, data_dir/val.en,
        data_dir/test.cn, data_dir/test.en
    '''
    toktok = ToktokTokenizer()

    jieba.enable_parallel(njobs)

    # TODO: if split_chinese_into_characters:
    # TODO: preprocessing and postprocessing.
    ZH = data.Field(
        tokenize=chinese_tokenizer, 
        init_token=config.SOS_TOKEN, 
        eos_token=config.EOS_TOKEN
    )
    
    EN = data.Field(
        tokenize=toktok.tokenize, 
        init_token=config.SOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        lower=True
    )

    train, val, test = datasets.TranslationDataset.splits(
        path=data_dir, 
        train='train', validation='val', test='test', 
        exts=('.cn', '.en'), fields=(ZH, EN)
    )

    # TODO: fine-tune this
    ZH.build_vocab(train.src, min_freq=3, max_size=60000)
    EN.build_vocab(train.trg, min_freq=3, max_size=60000)

    # Some debug statements.
    # TODO: use logger to save this in file
    print("Most common chinese vocabs:", ZH.vocab.freqs.most_common(10))
    print("Chinese vocab size:", len(ZH.vocab))
    print("Most common english vocabs:", EN.vocab.freqs.most_common(10))
    print("English vocab size:", len(EN.vocab))
    
    return train, val, test, ZH, EN
    
if __name__ == '__main__':
    # Test scripts
    train, val, test, ZH, EN = load_chinese_english_data('../data/neu2017/', 4)
    
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_size=4)

    batch = next(iter(train_iter))
    print(batch.src)
    print(batch.trg)