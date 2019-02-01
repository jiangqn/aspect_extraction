import torch
import pickle
from data_process.utils import nltk_tokenize

word2index_path = './data/official_data/processed_data/restaurant/word2index.pickle'
decnn_path = './data/official_data/processed_data/restaurant/extraction/decnn.pkl'
gcae_path = './data/official_data/processed_data/restaurant/classification/gcae.pkl'
data_path = './data/restaurant/train.txt'

with open(word2index_path, 'rb') as handle:
    word2index = pickle.load(handle)
decnn = torch.load(decnn_path)
gcae = torch.load(gcae_path)

src = 'This place is a great deal for the price and the food they give you'

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

def process(sentence_text):
    sentence_text = nltk_tokenize(sentence_text.strip().lower())
    sentence = text2tensor(sentence_text)
    logit = decnn(sentence)
    predicts = logit.max(dim=-1)[1]
    predicts = predicts[0].tolist()
    terms = extract_terms(sentence_text, predicts)
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

# process(src)

file = open(data_path)

count = 0
for line in file.readlines():
    process(line)
    count += 1
    if count >= 10:
        break