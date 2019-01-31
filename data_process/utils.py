from nltk import word_tokenize
import operator
import os
import numpy as np

def nltk_tokenize(text):
    return word_tokenize(text)

PAD = '<PAD>'
UNK = '<UNK>'

PAD_INDEX = 0
UNK_INDEX = 1

INIT = 1e-2

class Vocab(object):

    def __init__(self):
        self._count_dict = dict()
        self._predefined_list = [PAD, UNK]

    def add(self, word):
        if word in self._count_dict:
            self._count_dict[word] += 1
        else:
            self._count_dict[word] = 1

    def add_list(self, words):
        for word in words:
            self.add(word)

    def get_vocab(self, max_size=None, min_freq=0):
        sorted_words = sorted(self._count_dict.items(), key=operator.itemgetter(1), reverse=True)
        word2index = {}
        for word in self._predefined_list:
            word2index[word] = len(word2index)
        for word, freq in sorted_words:
            if (max_size is not None and len(word2index) >= max_size) or freq < min_freq:
                word2index[word] = word2index[UNK]
            else:
                word2index[word] = len(word2index)
        index2word = {}
        index2word[word2index[UNK]] = UNK
        for word, index in word2index.items():
            if index == word2index[UNK]:
                continue
            else:
                index2word[index] = word
        return word2index, index2word

def load_word_embeddings(fname, vocab_size, embed_size, word2index):
    if not os.path.isfile(fname):
        raise IOError('Not a file', fname)

    word2vec = np.random.uniform(-INIT, INIT, [vocab_size, embed_size])
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                word2vec[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    word2vec[PAD_INDEX, :] = 0
    return word2vec
