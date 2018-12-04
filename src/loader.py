from torchtext import data
from torchtext import datasets

import re
import nltk
import jieba
from functools import partial
import config

from nltk.tokenize.toktok import ToktokTokenizer
import io
import os
import string
from functools import partial

import sacrebleu

class myTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))
    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('idx', data.LabelField(use_vocab=False))]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for i, (src_line, trg_line) in enumerate(zip(src_file, trg_file)):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line, i], fields))

        super(myTranslationDataset, self).__init__(examples, fields, **kwargs)
    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
    
def chinese_tokenizer(line):
    # Currently using jieba to tokenize Chinese
    # TODO: use Stanford tokenizer on prince
    return list(jieba.cut(line, cut_all=False))

def load_chinese_english_data(data_dir, njobs, split_chinese_into_characters=False):
    '''
    DEPRECATED
    '''
    toktok = ToktokTokenizer()

    jieba.enable_parallel(njobs)

    # TODO: if split_chinese_into_characters:
    # TODO: preprocessing and postprocessing.
    ZH = data.Field(
        tokenize=chinese_tokenizer, 
        init_token=config.SOS_TOKEN, 
        eos_token=config.EOS_TOKEN,
        fix_length=args.max_sentence_length
    )
    
    EN = data.Field(
        tokenize=toktok.tokenize, 
        init_token=config.SOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        lower=True,
        fix_length=args.max_sentence_length
    )

    train, val, test = myTranslationDataset.splits(
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

def tokenize_by_character(line):
    line = line.replace(" ", "")
    return(list(line))
    

def split_after_removing_punctuations(exclude, line):
    token_list = str.split(line)
    return [tok for tok in token_list if tok not in exclude]

def transform(token):
    if token == '&amp;':
        return 'and'
    elif token.startswith('&apos;'):
        return "'" + token[6:]
    return token

def transform_v3(token):
    if token == '&amp;':
        return '&'
    if token == '&quot;':
        return '"'
    if token == '&apos;':
        return "'"
    if token == '&#91;':
        return '('
    if token == '&#93;':
        return ')'
    if token.startswith('&apos;'):
        return "'" + token[6:]
    return token

def split_after_removing_punctuations_and_replace_special_words(exclude, line):
    token_list = str.split(line)
    return [transform(tok) for tok in token_list if tok not in exclude]

def v3_tokenize(line):
    token_list = str.split(line)
    return [transform_v3(tok) for tok in token_list]

def sacrebleu_tokenize_zh(line):
    token_list =  str.split(sacrebleu.tokenize_zh(line))
    return [tok for tok in token_list if tok not in ["amp","quot"]]
    

def load_data(args):
    '''
    Loads the following files:
        data_dir/train.cn, data_dir/train.en,
        data_dir/val.cn, data_dir/val.en,
        data_dir/test.cn, data_dir/test.en
    '''

    assert args.source_lang in ["vi", "zh"], "unsupported source language: {}".format(args.source_lang)
    if args.source_lang != 'zh':
        assert not args.split_chinese_into_characters, "Source lang is not chinese but split_chinese_into_characters is set to True"
        
    exclude = set(string.punctuation)
    tokenize_without_punctuations = partial(split_after_removing_punctuations, exclude)
    
    exclude_punctuations_and_ampersand_characters = exclude.union(set(['&quot;', '&#91;', '&#93;', '&apos;']))
    #print(exclude_punctuations_and_ampersand_characters)
    tokenize_without_punctuations_and_ampersand_characters = partial(split_after_removing_punctuations_and_replace_special_words, exclude_punctuations_and_ampersand_characters)


    
    if args.split_chinese_into_characters:
        if args.preprocess_version >=3:
            # we could just use the tokenization scheme from sacrebleu and use the not tokenized version of chinese data
            # also better because it doesn't split english words into characters
            tokenize_func = sacrebleu_tokenize_zh
        else:
            tokenize_func = tokenize_by_character
            
        SRC = data.Field(
            tokenize=tokenize_func, 
            init_token='<sos>', 
            eos_token='<eos>',
            include_lengths=True,
            fix_length=args.max_sentence_length
        )
    else:
        #TODO: do we need to tokenize vi and zh differently?
        if args.source_lang == 'zh':
            if args.preprocess_version >= 3:
                raise NotImplementedError("v3 Chinese tokenizer splits characters")
            else:
                tokenize_func = tokenize_without_punctuations
        else:
            if args.preprocess_version == 1:
                tokenize_func = tokenize_without_punctuations
            elif args.preprocess_version == 2:
                tokenize_func = tokenize_without_punctuations_and_ampersand_characters
            elif args.preprocess_version >= 3:
                tokenize_func = v3_tokenize
        SRC = data.Field(
            tokenize=tokenize_func, 
            init_token='<sos>', 
            eos_token='<eos>',
            include_lengths=True,
            fix_length=args.max_sentence_length
        )
        
    if args.preprocess_version == 1:
        tokenize_func = tokenize_without_punctuations
    elif args.preprocess_version == 2:
        tokenize_func = tokenize_without_punctuations_and_ampersand_characters
    elif args.preprocess_version >= 3:
        tokenize_func = v3_tokenize
    EN = data.Field(
        tokenize=tokenize_func, 
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        include_lengths=True,
        fix_length=args.max_sentence_length
    )

    if args.source_lang == 'zh':
        '''
        if args.preprocess_version == 3:
            train, val, test = myTranslationDataset.splits(
                path=args.data, 
                train='train', validation='dev', test='test', 
                exts=('.zh', '.tok.en'), fields=(SRC, EN)
            )
        else:
        '''
        train, val, test = myTranslationDataset.splits(
            path=args.data, 
            train='train', validation='dev', test='test', 
            exts=('.tok.zh', '.tok.en'), fields=(SRC, EN)
        )
    else:
        train, val, test = myTranslationDataset.splits(
            path=args.data, 
            train='train', validation='dev', test='test', 
            exts=('.tok.vi', '.tok.en'), fields=(SRC, EN)
        )

    # TODO: fine-tune this
    SRC.build_vocab(train.src, min_freq=args.min_freq, max_size=args.max_vocab_size)
    EN.build_vocab(train.trg, min_freq=args.min_freq, max_size=args.max_vocab_size)

    # Some debug statements.
    # TODO: use logger to save this in file
    print("Most common source vocabs:", SRC.vocab.freqs.most_common(10))
    print("Source vocab size:", len(SRC.vocab))
    print("Most common english vocabs:", EN.vocab.freqs.most_common(10))
    print("English vocab size:", len(EN.vocab))
    
    return train, val, test, SRC, EN
    
if __name__ == '__main__':
    # Test scripts
    
    train, val, test, ZH, EN = load_chinese_english_data('../data/neu2017/', 4)
    
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_size=4)

    batch = next(iter(train_iter))
    print(batch.src)
    print(batch.trg)
    print(batch.idx)
