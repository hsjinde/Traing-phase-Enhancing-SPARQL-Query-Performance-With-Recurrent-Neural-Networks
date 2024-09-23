
#### Using Glove
class CreateModel:

    def shared_layers(word_input, pos_input, wordsList, wordVectors, posList, posVectors, W, U, with_pos=True):
        # 嵌入層
        word_embed = Embedding(len(wordsList), W, weights=[np.array(wordVectors)], trainable=False)(word_input)
        if with_pos:
            pos_embed = Embedding(len(posList), 20, weights=[np.array(posVectors)], trainable=False)(pos_input)
            data_concatenate = Concatenate()([word_embed, pos_embed])
        else:
            data_concatenate = word_embed
        return BatchNormalization()(data_concatenate)

    def lstm_stack(input_layer, U, layers=1):
        x = input_layer
        for _ in range(layers):
            dense = Dense(U * 2)(x)
            lstm = Bidirectional(LSTM(U, return_sequences=False))(dense)
            x = Add()([dense, lstm])
            x = LayerNormalization()(x)
        return x

    def build_model(wordsList, wordVectors, posList, posVectors, W, U, Number_label, layers=1, with_pos=True):
        Previous = Input(shape=(Number_label,))
        word_input = Input(shape=(maxlen,))
        pos_input = Input(shape=(maxlen,)) if with_pos else None

        Bn = CreateModel.shared_layers(word_input, pos_input, wordsList, wordVectors, posList, posVectors, W, U, with_pos)
        lstm_output = CreateModel.lstm_stack(Bn, U, layers)
        final_concat = Concatenate()([lstm_output, Previous])
        output = Dense(1, activation='sigmoid')(final_concat)

        inputs = [Previous, word_input] if not with_pos else [Previous, word_input, pos_input]
        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
#### Using BERT
from keras import backend as K

class CreateModel:

    def build_model(U, n_layers):
        K.clear_session()
        bert = TFAutoModel.from_pretrained('bert-base-cased')
        
        input_ids = Input(shape=(maxlen,), name='input_ids', dtype='int32')
        mask = Input(shape=(maxlen,), name='attention_mask', dtype='int32')
        
        # BERT嵌入
        embeddings = bert.bert(input_ids, attention_mask=mask)[0]
        Bn = BatchNormalization()(embeddings)
        
        # LSTM 層堆疊
        lstm_input = Bn
        for _ in range(n_layers):
            dense = Dense(U * 2)(lstm_input)
            lstm_layer = Bidirectional(LSTM(U, return_sequences=False))(dense)
            add = Add()([dense, lstm_layer])
            lstm_input = LayerNormalization()(add)
        
        output = Dense(1, activation='sigmoid', name='outputs')(lstm_input)
        
        model = tf.keras.Model(inputs=[input_ids, mask], outputs=output)
        
        # 凍結 BERT 層
        model.layers[2].trainable = False
        
        # 編譯模型
        acc = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=[acc])
        
        return model