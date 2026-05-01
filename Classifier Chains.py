import numpy as np
import tensorflow as tf
from tensorflow import keras
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


class GloveClassifierChainModel:
    @staticmethod
    def shared_layers(word_input, pos_input, wordsList, wordVectors, posList, posVectors, W, U=None, with_pos=True):
        word_embed = Embedding(len(wordsList), W, weights=[np.array(wordVectors)], trainable=False)(word_input)
        if with_pos:
            pos_embed = Embedding(len(posList), 20, weights=[np.array(posVectors)], trainable=False)(pos_input)
            x = Concatenate()([word_embed, pos_embed])
        else:
            x = word_embed
        return BatchNormalization()(x)

    @staticmethod
    def lstm_stack(input_layer, U, layers=1):
        x = input_layer
        for _ in range(max(layers, 1)):
            residual = Dense(U * 2)(x)
            x = Bidirectional(LSTM(U, return_sequences=True))(residual)
            x = Add()([residual, x])
            x = LayerNormalization()(x)
        return GlobalMaxPooling1D()(x)

    @staticmethod
    def build_model(wordsList, wordVectors, posList, posVectors, W, U, Number_label, layers=1, with_pos=True, maxlen=20):
        previous = Input(shape=(Number_label,), name="previous_labels")
        word_input = Input(shape=(maxlen,), name="word_input")
        pos_input = Input(shape=(maxlen,), name="pos_input") if with_pos else None

        x = GloveClassifierChainModel.shared_layers(
            word_input, pos_input, wordsList, wordVectors, posList, posVectors, W, U, with_pos
        )
        x = GloveClassifierChainModel.lstm_stack(x, U, layers)
        x = Concatenate()([x, previous])
        output = Dense(1, activation="sigmoid", name="outputs")(x)

        inputs = [previous, word_input] if not with_pos else [previous, word_input, pos_input]
        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        return model


class BertClassifierChainModel:
    @staticmethod
    def build_model(U, n_layers=1, maxlen=20):
        K.clear_session()
        bert = TFAutoModel.from_pretrained("bert-base-cased")

        input_ids = Input(shape=(maxlen,), name="input_ids", dtype="int32")
        mask = Input(shape=(maxlen,), name="attention_mask", dtype="int32")

        x = bert.bert(input_ids, attention_mask=mask)[0]
        x = BatchNormalization()(x)

        for _ in range(max(n_layers, 1)):
            residual = Dense(U * 2)(x)
            x = Bidirectional(LSTM(U, return_sequences=True))(residual)
            x = Add()([residual, x])
            x = LayerNormalization()(x)

        x = GlobalMaxPooling1D()(x)
        output = Dense(1, activation="sigmoid", name="outputs")(x)

        model = tf.keras.Model(inputs=[input_ids, mask], outputs=output)
        bert.trainable = False

        acc = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=[acc])

        return model


CreateGloveModel = GloveClassifierChainModel
CreateBertModel = BertClassifierChainModel
CreateModel = BertClassifierChainModel
