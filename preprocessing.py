import os
import numpy as np
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

# 初始化 NLTK 資源
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# 初始化 BERT Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# GloVe Embedding
def WordEmbedding(emb_size):
    """
    加載 GloVe 詞嵌入。
    :param emb_size: 詞嵌入的維度大小 (50, 100, 200, 300)
    :return: 詞列表、詞索引字典、詞向量
    """
    embedding_path = os.path.join(os.path.dirname(__file__), '../Embedding/glove.6B.' + str(emb_size) + 'd.txt')
    wordVectors = [np.zeros(emb_size, dtype=np.float32), np.random.randn(emb_size).astype(np.float32)]
    wordsList = ['-pad-', '-oov-']
    try:
        with open(embedding_path, encoding="utf8") as glove_file:
            for line in glove_file:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                wordsList.append(word)
                wordVectors.append(coefs)
    except FileNotFoundError:
        print(f"Error: File not found at {embedding_path}")
        return None, None, None

    word_idx = {w: i for i, w in enumerate(wordsList)}
    return wordsList, word_idx, wordVectors

def Transform2WordEmbedding(Sentence, wordsList, word_idx, maxlen=20):
    """
    將句子轉換為詞嵌入索引。
    :param Sentence: 輸入句子
    :param wordsList: 詞列表
    :param word_idx: 詞索引字典
    :param maxlen: 最大長度
    :return: 詞嵌入索引的序列
    """
    wordembedding = []
    temp = []
    words = nltk.word_tokenize(Sentence)
    for word in words:
        word = str.lower(word)
        temp.append(word_idx.get(word, 1))  # 使用 1 作為 '-oov-' 的對應索引
    wordembedding.append(temp)
    wordembedding = pad_sequences(wordembedding, padding='post', maxlen=maxlen)
    return wordembedding

# BERT Tokenization
def tokenize(sentence, maxlen=20):
    """
    使用 BERT Tokenizer 將句子轉換為編碼。
    :param sentence: 輸入句子
    :param maxlen: 最大長度
    :return: 編碼的輸入 ID 和注意力遮罩
    """
    tokens = tokenizer.encode_plus(sentence, max_length=maxlen, truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False,
                                   return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

# POS Embedding
def PosEmbedding():
    """
    加載 POS 標籤嵌入。
    :return: POS 標籤列表、POS 索引字典、POS 向量
    """
    embedding_path = os.path.join(os.path.dirname(__file__), './Embedding/pos_emb_win5_size20.txt')
    posList = ['-pad-']
    posVectors = [np.zeros(20, dtype=np.float32)]
    try:
        with open(embedding_path, encoding="utf8") as pos_file:
            for line in pos_file:
                pos, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                posList.append(pos)
                posVectors.append(coefs)
    except FileNotFoundError:
        print(f"Error: File not found at {embedding_path}")
        return None, None, None

    pos_idx = {p: i for i, p in enumerate(posList)}
    return posList, pos_idx, posVectors

def Transform2PosEmbedding(Sentence, posList, pos_idx, maxlen=20):
    """
    將句子轉換為 POS 標籤嵌入索引。
    :param Sentence: 輸入句子
    :param posList: POS 標籤列表
    :param pos_idx: POS 索引字典
    :param maxlen: 最大長度
    :return: POS 標籤嵌入索引的序列
    """
    x_pos = []
    words = nltk.word_tokenize(Sentence)
    pos_tags = nltk.pos_tag(words)
    temp = [pos for word, pos in pos_tags]
    x_pos.append(temp)
    
    posembedding = []
    for line in x_pos:
        temp = [pos_idx.get(tag, 0) for tag in line]  # 使用 0 作為未知 POS 的預設值
        posembedding.append(temp)
    
    posembedding = pad_sequences(posembedding, padding='post', maxlen=maxlen)
    return posembedding
