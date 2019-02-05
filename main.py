import torch
import pickle
import argparse
from data_process.utils import nltk_tokenize

word2index_path = './data/official_data/processed_data/restaurant/word2index.pickle'
decnn_path = './data/official_data/processed_data/restaurant/extraction/decnn.pkl'
gcae_path = './data/official_data/processed_data/restaurant/classification/gcae.pkl'
count_dict_path = './data/restaurant/count_dict.pickle'

with open(word2index_path, 'rb') as handle:
    word2index = pickle.load(handle)
decnn = torch.load(decnn_path)
gcae = torch.load(gcae_path)

with open(count_dict_path, 'rb') as handle:
    count_dict = pickle.load(handle)

def text2tensor(text):
    tensor = []
    for word in text:
        tensor.append(word2index[word] if word in word2index else word2index['<UNK>'])
    tensor = torch.tensor(tensor).unsqueeze(0).long().cuda()
    return tensor

def extract_terms(text, predicts):
    text = text + ['']
    predicts = predicts + [0]
    terms = []
    term = []
    s = len(text)
    state = 0
    for i in range(s):
        if state == 0:
            if predicts[i] == 0:
                continue
            else:
                term.append(text[i])
                state = 1
        else:
            if predicts[i] == 0:
                terms.append(term)
                term = []
                state = 0
            else:
                term.append(text[i])
    return terms

def word_list2text(word_list):
    text = ''
    for word in word_list:
        text += word + ' '
    text = text.strip()
    return text

def process(sentence_text, wdata):
    sentence_text = nltk_tokenize(sentence_text.strip().lower())
    sentence = text2tensor(sentence_text)
    logit = decnn(sentence)
    predicts = logit.max(dim=-1)[1]
    predicts = predicts[0].tolist()
    terms = extract_terms(sentence_text, predicts)
    count = 0
    for term_text in terms:
        print('sentence:', word_list2text(sentence_text))
        print('term:', word_list2text(term_text))
        sentence = text2tensor(sentence_text)
        term = text2tensor(term_text)
        logit = gcae(sentence, term)
        predict = logit.max(dim=-1, keepdim=False)[1]
        if predict[0] == 0:
            sentiment = 'positive'
        else:
            sentiment = 'negative'
        print('sentiment:', sentiment)
        option = input()
        if option == 'y':
            wdata.write(word_list2text(sentence_text) + '\n')
            wdata.write(word_list2text(term_text) + '\n')
            wdata.write(sentiment + '\n')
            count += 1
        elif option == 'n':
            wdata.write(word_list2text(sentence_text) + '\n')
            wdata.write(word_list2text(term_text) + '\n')
            if sentiment == 'positive':
                sentiment = 'negative'
            else:
                sentiment = 'positive'
            wdata.write(sentiment + '\n')
            count += 1
        else:
            continue
    return count

parser = argparse.ArgumentParser()
parser.add_argument('--data_index', type=int, default=0)
config = parser.parse_args()
data_index = config.data_index

data_path = './data/restaurant/reviews/%04d.txt' % (data_index)
target_path = './data/restaurant/processed/%04d.txt' % (data_index)
rdata = open(data_path, 'r', encoding=u'utf-8')
wdata = open(target_path, 'w', encoding=u'utf-8')

count = 0
for sentence in rdata:
    count += process(sentence, wdata)

count_dict[data_index] = count
total = 0
for index in count_dict:
    total += count_dict[index]
print('this file: %d' % count)
print('total samples: %d' % total)

with open(count_dict_path, 'wb') as handle:
    pickle.dump(count_dict, handle)