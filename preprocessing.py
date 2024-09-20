import numpy as np
import pandas as pd
import re
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Glove
def WordEmbedding(emb_size):
    wordVectors = [np.zeros(emb_size, dtype=np.float32), np.random.randn(emb_size).astype(np.float32)]
    wordsList = ['-pad-', '-oov-']
    with open('../Embedding/' + 'glove.6B.' + str(emb_size) + 'd.txt', encoding="utf8") as glove_file:
        for line in glove_file:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            wordsList.append(word)
            wordVectors.append(coefs)

    word_idx = {w: i for i, w in enumerate(wordsList)}
    return wordsList, word_idx, wordVectors

def Transform2WordEmbedding(Sentence, wordsList, word_idx, maxlen=20):
    wordembedding = []
    temp = []
    words = nltk.word_tokenize(Sentence)
    for word in words:
        word = str.lower(word)
        if word in wordsList:
            temp.append(word_idx[word])
        else:
            temp.append(1)  # 使用 1 作為 '-oov-' 的對應索引
    wordembedding.append(temp)
    wordembedding = pad_sequences(wordembedding, padding='post', maxlen=maxlen)
    return wordembedding

#BERT
def tokenize(sentence, maxlen=20):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokens = tokenizer.encode_plus(sentence, max_length=maxlen, truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False,
                                   return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

def PosEmbedding():
    posList = ['-pad-']
    posVectors = [np.zeros(20, dtype=np.float32)]
    with open('./Embedding/pos_emb_win5_size20.txt', encoding="utf8") as pos_file:
        for line in pos_file:
            pos, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            posList.append(pos)
            posVectors.append(coefs)

    pos_idx = {p: i for i, p in enumerate(posList)}
    return posList, pos_idx, posVectors

def Transform2PosEmbedding(Sentence, posList, pos_idx, maxlen=20):
    x_pos = []
    words = nltk.word_tokenize(Sentence)
    pos_tags = nltk.pos_tag(words)
    temp = [pos for word, pos in pos_tags]
    x_pos.append(temp)
    
    posembedding = []
    for line in x_pos:
        temp = [pos_idx.get(tag, 0) for tag in line]
        posembedding.append(temp)
    
    posembedding = pad_sequences(posembedding, padding='post', maxlen=maxlen)
    return posembedding
