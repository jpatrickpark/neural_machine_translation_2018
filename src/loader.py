from torchtext import data
from torchtext import datasets

import re
import nltk
import jieba
from functools import partial
import config

from nltk.tokenize.toktok import ToktokTokenizer
import io
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
    
if __name__ == '__main__':
    # Test scripts
    train, val, test, ZH, EN = load_chinese_english_data('../data/neu2017/', 4)
    
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_size=4)

    batch = next(iter(train_iter))
    print(batch.src)
    print(batch.trg)