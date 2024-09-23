
#### Using Glove
def create_model(wordsList, wordVectors, posList, posVectors, W, U, layers=1, bidirectional=False, use_pos=False):
    K.clear_session()
    
    word_input = Input(shape=(maxlen,))
    word_embed = Embedding(len(wordsList), W, weights=[np.array(wordVectors)], trainable=False)(word_input)
    
    inputs = [word_input]
    
    if use_pos:
        pos_input = Input(shape=(maxlen,))
        pos_embed = Embedding(len(posList), 20, weights=[np.array(posVectors)], trainable=False)(pos_input)
        data_concatenate = Concatenate()([word_embed, pos_embed])
        inputs.append(pos_input)
    else:
        data_concatenate = word_embed
    
    if bidirectional:
        lstm_layer = Bidirectional(LSTM(U, return_sequences=(layers > 1)))(data_concatenate)
    else:
        lstm_layer = LSTM(U, return_sequences=(layers > 1))(data_concatenate)
    
    for _ in range(layers - 1):
        dense_1 = Dense(U * 2)(lstm_layer)
        add = Add()([dense_1, lstm_layer])
        norm = LayerNormalization()(add)
        if bidirectional:
            lstm_layer = Bidirectional(LSTM(U, return_sequences=False))(norm)
        else:
            lstm_layer = LSTM(U, return_sequences=False)(norm)


#### Using BERT
from keras import backend as K

class CreateModel:  
    def create_layer(U, maxlen, num_layers):
        K.clear_session()
        bert = TFAutoModel.from_pretrained('bert-base-cased')
        input_ids = Input(shape=(maxlen,), dtype='int32')
        mask = Input(shape=(maxlen,), dtype='int32')
        embeddings = bert(input_ids, attention_mask=mask)[0]
        embeddings = BatchNormalization()(embeddings)
        
        x = Dense(U * 2)(embeddings)
        for _ in range(num_layers):
            lstm_layer = Bidirectional(LSTM(U, return_sequences=False))(x)
            x = Add()([x, lstm_layer])
            x = LayerNormalization()(x)
            x = Dense(U * 2)(x)
        
        output = Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=[input_ids, mask], outputs=output)
        model.layers[2].trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
