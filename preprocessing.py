from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
_TOKENIZER = None
_NLTK = None


def _get_nltk():
    global _NLTK
    if _NLTK is None:
        try:
            import nltk
        except ImportError as exc:
            raise ImportError("Install nltk to tokenize sentences or transform POS tags.") from exc
        _NLTK = nltk
    return _NLTK


def _ensure_nltk_resource(resource, download_name):
    nltk = _get_nltk()
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(download_name, quiet=True)


def _ensure_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")
    return _TOKENIZER


def _pad_sequences(sequences, maxlen, padding_value=0):
    padded = np.full((len(sequences), maxlen), padding_value, dtype=np.int32)
    for row, sequence in enumerate(sequences):
        truncated = list(sequence)[:maxlen]
        padded[row, : len(truncated)] = truncated
    return padded


def _read_embedding_file(embedding_path, vector_size, initial_tokens):
    tokens = [token for token, _ in initial_tokens]
    vectors = [vector for _, vector in initial_tokens]

    with embedding_path.open(encoding="utf8") as embedding_file:
        for line in embedding_file:
            token, coefs = line.split(maxsplit=1)
            vector = np.fromstring(coefs, dtype=np.float32, sep=" ")
            if vector.size == vector_size:
                tokens.append(token)
                vectors.append(vector)

    token_idx = {token: i for i, token in enumerate(tokens)}
    return tokens, token_idx, vectors


def WordEmbedding(emb_size, embedding_dir=None):
    embedding_dir = Path(embedding_dir) if embedding_dir is not None else BASE_DIR.parent / "Embedding"
    embedding_path = embedding_dir / f"glove.6B.{emb_size}d.txt"

    initial_tokens = [
        ("-pad-", np.zeros(emb_size, dtype=np.float32)),
        ("-oov-", np.random.randn(emb_size).astype(np.float32)),
    ]

    try:
        return _read_embedding_file(embedding_path, emb_size, initial_tokens)
    except FileNotFoundError:
        print(f"Error: File not found at {embedding_path}")
        return None, None, None


def Transform2WordEmbedding(Sentence, wordsList, word_idx, maxlen=20):
    if word_idx is None:
        raise ValueError("word_idx must be provided. Did WordEmbedding fail to load the embedding file?")
    _ensure_nltk_resource("tokenizers/punkt", "punkt")
    nltk = _get_nltk()

    token_ids = []
    for word in nltk.word_tokenize(Sentence):
        token_ids.append(word_idx.get(word.lower(), 1))

    return _pad_sequences([token_ids], maxlen=maxlen)


def tokenize(sentence, maxlen=20):
    tokenizer = _ensure_tokenizer()
    tokens = tokenizer.encode_plus(
        sentence,
        max_length=maxlen,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="tf",
    )
    return tokens["input_ids"], tokens["attention_mask"]


def PosEmbedding(embedding_path=None):
    embedding_path = Path(embedding_path) if embedding_path is not None else BASE_DIR / "POS embedding" / "pos_emb_win5_size20.txt"
    initial_tokens = [("-pad-", np.zeros(20, dtype=np.float32))]

    try:
        return _read_embedding_file(embedding_path, 20, initial_tokens)
    except FileNotFoundError:
        print(f"Error: File not found at {embedding_path}")
        return None, None, None


def Transform2PosEmbedding(Sentence, posList, pos_idx, maxlen=20):
    if pos_idx is None:
        raise ValueError("pos_idx must be provided. Did PosEmbedding fail to load the embedding file?")
    _ensure_nltk_resource("tokenizers/punkt", "punkt")
    _ensure_nltk_resource("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")
    nltk = _get_nltk()

    words = nltk.word_tokenize(Sentence)
    token_ids = [pos_idx.get(pos, 0) for _, pos in nltk.pos_tag(words)]
    return _pad_sequences([token_ids], maxlen=maxlen)
