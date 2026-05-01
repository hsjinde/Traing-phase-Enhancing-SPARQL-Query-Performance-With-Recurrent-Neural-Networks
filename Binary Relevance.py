import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    LayerNormalization,
)
from transformers import TFAutoModel


def create_model(
    wordsList,
    wordVectors,
    posList,
    posVectors,
    W,
    U,
    layers=1,
    bidirectional=False,
    use_pos=False,
    maxlen=20,
):
    K.clear_session()

    word_input = Input(shape=(maxlen,), name="word_input")
    word_embed = Embedding(len(wordsList), W, weights=[np.array(wordVectors)], trainable=False)(word_input)

    inputs = [word_input]

    if use_pos:
        pos_input = Input(shape=(maxlen,), name="pos_input")
        pos_embed = Embedding(len(posList), 20, weights=[np.array(posVectors)], trainable=False)(pos_input)
        x = Concatenate()([word_embed, pos_embed])
        inputs.append(pos_input)
    else:
        x = word_embed

    for _ in range(max(layers - 1, 0)):
        residual = Dense(U * 2 if bidirectional else U)(x)
        if bidirectional:
            x = Bidirectional(LSTM(U, return_sequences=True))(residual)
        else:
            x = LSTM(U, return_sequences=True)(residual)
        x = Add()([residual, x])
        x = LayerNormalization()(x)

    if bidirectional:
        x = Bidirectional(LSTM(U, return_sequences=False))(x)
    else:
        x = LSTM(U, return_sequences=False)(x)

    output = Dense(1, activation="sigmoid", name="outputs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


class BertBinaryRelevanceModel:
    @staticmethod
    def create_layer(U, maxlen=20, num_layers=1):
        K.clear_session()
        bert = TFAutoModel.from_pretrained("bert-base-cased")
        input_ids = Input(shape=(maxlen,), name="input_ids", dtype="int32")
        mask = Input(shape=(maxlen,), name="attention_mask", dtype="int32")
        x = bert(input_ids, attention_mask=mask)[0]
        x = BatchNormalization()(x)

        for _ in range(num_layers):
            residual = Dense(U * 2)(x)
            x = Bidirectional(LSTM(U, return_sequences=True))(residual)
            x = Add()([residual, x])
            x = LayerNormalization()(x)

        x = GlobalMaxPooling1D()(x)
        output = Dense(1, activation="sigmoid", name="outputs")(x)
        model = tf.keras.Model(inputs=[input_ids, mask], outputs=output)
        bert.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        return model


CreateModel = BertBinaryRelevanceModel
