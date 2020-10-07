# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:26 PM


from collections import Counter
from typing import List
from torchtext.vocab import Vocab
from pathlib import Path


class ClassVocab(Vocab):

    def __init__(self, classes, specials=['<pad>', '<unk>'], **kwargs):
        '''
        convert key to index(stoi), and get key string by index(itos)
        :param classes: list or str, key string or entity list
        :param specials: list, special tokens except <unk> (default: {['<pad>', '<unk>']})
        :param kwargs:
        '''
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes
        c = Counter(cls_list)
        self.special_count = len(specials)
        super().__init__(c, specials=specials, **kwargs)


def entities2iob_labels(entities: list):
    '''
    get all iob string label by entities
    :param entities:
    :return:
    '''
    tags = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    return tags


# Keep references to global vocabs
vocab_cls = {'keys': None, 'iob_labels': None, 'entities': None}


def set_vocab(entities_list: List[str]):
    """Sets keys, iob_labels and entities Vocabs in `vocab_cls` dict with given list of entities."""
    global vocab_cls
    vocab_cls['keys'] = ClassVocab(Path(__file__).parent.joinpath('keys.txt'), specials_first=False)
    vocab_cls['iob_labels'] = ClassVocab(entities2iob_labels(entities_list), specials_first=False)
    vocab_cls['entities'] = ClassVocab(entities_list, specials_first=False)
