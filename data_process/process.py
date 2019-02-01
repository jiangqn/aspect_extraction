from utils import *
import numpy as np
import os
import pickle

data_path = '../data/official_data/processed_data/sentences_term_restaurant.txt'
target_path = '../data/official_data/processed_data/restaurant/'
glove_path = '../data/glove.840B.300d.txt'

dev_rate = 0.2

file = open(data_path)
data_dict = {}

for line in file.readlines():
    line = line.rstrip().split('__split__')
    if line[0] in data_dict:
        data_dict[line[0]].append([int(line[3]), int(line[4])])
    else:
        data_dict[line[0]] = [[int(line[3]), int(line[4])]]

sentences = []
labels = []

vocab = Vocab()

max_len = 0

for sentence, positions in data_dict.items():
    positions = sorted(positions)
    start = 0
    end = None
    result = []
    label = []
    for position in positions:
        end = position[0]
        tmp = nltk_tokenize(sentence[start: end])
        result += tmp
        label += [0] * len(tmp)
        start, end = position
        tmp = nltk_tokenize(sentence[start: end])
        result += tmp
        label += [1] * len(tmp)
        start = position[1]
    end = len(sentence)
    tmp = nltk_tokenize(sentence[start: end])
    result += tmp
    label += [0] * len(tmp)
    sentences.append(result)
    labels.append(label)
    vocab.add_list(result)
    max_len = max(max_len, len(result))

word2index, index2word = vocab.get_vocab()
num = len(sentences)
glove = load_word_embeddings(glove_path, len(index2word), 300, word2index)

sentences_np = np.zeros((num, max_len), dtype=np.int32)
labels_np = np.zeros((num, max_len), dtype=np.int32)

for i, (sentence, label) in enumerate(zip(sentences, labels)):
    for j, word in enumerate(sentence):
        sentences_np[i, j] = word2index[word]
    for j, flag in enumerate(label):
        labels_np[i, j] = flag

dev_num = int(num * dev_rate)
train_num = num - dev_num

train_sentences = sentences_np[0: train_num].copy()
train_labels = labels_np[0: train_num].copy()
dev_sentences = sentences_np[train_num: num].copy()
dev_labels = labels_np[train_num: num].copy()

np.savez(target_path + 'extraction/train.npz', sentences=train_sentences, labels=train_labels)
np.savez(target_path + 'extraction/dev.npz', sentences=dev_sentences, labels=dev_labels)

np.save(target_path + 'glove.npy', glove)

with open(os.path.join(target_path + 'word2index.pickle'), 'wb') as handle:
    pickle.dump(word2index, handle)
with open(os.path.join(target_path + 'index2word.pickle'), 'wb') as handle:
    pickle.dump(index2word, handle)

file = open(data_path)

sentences = []
terms = []
labels = []

max_len = 0
max_term_len = 0

for line in file.readlines():
    line = line.rstrip().split('__split__')
    sentence = nltk_tokenize(line[0])
    term = nltk_tokenize(line[1])
    label = line[2]
    if label == 'positive':
        label = 0
    elif label == 'negative':
        label = 1
    else:
        continue
    sentences.append(sentence)
    terms.append(term)
    labels.append(label)
    max_len = max(max_len, len(sentence))
    max_term_len = max(max_term_len, len(term))

num = len(sentences)
dev_num = int(num * dev_rate)
train_num = num - dev_num

sentences_np = np.zeros((num, max_len), dtype=np.int32)
terms_np = np.zeros((num, max_term_len), dtype=np.int32)
labels_np = np.zeros(num, dtype=np.int32)

for i, (sentence, term, label) in enumerate(zip(sentences, terms, labels)):
    for j, word in enumerate(sentence):
        sentences_np[i, j] = word2index[word] if word in word2index else UNK_INDEX
    for j, word in enumerate(term):
        terms_np[i, j] = word2index[word] if word in word2index else UNK_INDEX
    labels_np[i] = label

train_sentences = sentences_np[0: train_num].copy()
train_terms = terms_np[0: train_num].copy()
train_labels = labels_np[0: train_num].copy()

dev_sentences = sentences_np[train_num: num].copy()
dev_terms = terms_np[train_num: num].copy()
dev_labels = labels_np[train_num: num].copy()

np.savez(target_path + 'classification/train.npz', sentences=train_sentences, terms=train_terms, labels=train_labels)
np.savez(target_path + 'classification/dev.npz', sentences=dev_sentences, terms=dev_terms, labels=dev_labels)