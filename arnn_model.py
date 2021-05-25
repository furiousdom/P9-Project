from keras.models import Model, Sequential
from keras.layers import GRU, LSTM, Bidirectional
from keras.layers import Input, Flatten, Reshape, Dense, Dropout
from performance_meter import measure_and_print_performance

def molecule_model_RNN_RNN(model_name):
    # Encoder
    encoder_input = Input(shape=(300, 1))
    encoded = LSTM(64, return_sequences=True)(encoder_input)
    encoded = LSTM(32)(encoded)
    encoded = Dense(128, activation='sigmoid')(encoded)
    encoded = Dense(50, activation='relu')(encoded)

    # Decoder
    decoded = Reshape((50, 1))(encoded)
    decoded = Dense(128, activation='sigmoid')(decoded)
    decoded = LSTM(32, return_sequences=True)(decoded)
    decoded = LSTM(64)(decoded)
    decoded = Dense(300, activation='relu')(decoded)

    autoencoder = Model(inputs=encoder_input, outputs=decoded, name=model_name)

    encoder = Model(inputs=encoder_input, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def molecule_model_RNN_DNN(model_name):
    # Encoder
    encoder_input = Input(shape=(300, 1))
    encoded = LSTM(64, return_sequences=True)(encoder_input)
    encoded = LSTM(32)(encoded)
    encoded = Dense(128, activation='sigmoid')(encoded)
    encoded = Dense(50, activation='relu')(encoded)

    # Decoder
    decoded = Dense(100, activation='sigmoid')(encoded)
    decoded = Dense(200, activation='relu')(decoded)
    decoded = Dense(300, activation='relu')(decoded)

    autoencoder = Model(inputs=encoder_input, outputs=decoded, name=model_name)

    encoder = Model(inputs=encoder_input, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def protein_model_RNN_RNN(model_name):
    # Encoder
    encoder_input = Input(shape=(100, 1))
    encoded = LSTM(64, return_sequences=True)(encoder_input)
    encoded = LSTM(32)(encoded)
    encoded = Dense(128, activation='sigmoid')(encoded)
    encoded = Dense(30, activation='relu')(encoded)

    # Decoder
    decoded = Reshape((30, 1))(encoded)
    decoded = Dense(128, activation='sigmoid')(decoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='relu')(decoded)
    decoded = Dense(100, activation='relu')(decoded)

    autoencoder = Model(inputs=encoder_input, outputs=decoded, name=model_name)

    encoder = Model(inputs=encoder_input, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def protein_model_RNN_DNN(model_name):
    # Encoder
    encoder_input = Input(shape=(100, 1))
    encoded = LSTM(64, return_sequences=True)(encoder_input)
    encoded = LSTM(32)(encoded)
    encoded = Dense(128, activation='sigmoid')(encoded)
    encoded = Dense(30, activation='relu')(encoded)

    # Decoder
    decoded = Dense(50, activation='sigmoid')(encoded)
    decoded = Dense(75, activation='relu')(decoded)
    decoded = Dense(100, activation='relu')(decoded)

    autoencoder = Model(inputs=encoder_input, outputs=decoded, name=model_name)

    encoder = Model(inputs=encoder_input, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def interaction_model(model_name):
    model = Sequential(name=model_name)

    model.add(Input(shape=(80,)))
    model.add(Dense(700, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))

    metrics=['accuracy', 'mean_squared_error']
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(model.summary())
    return model

def train_molecule_model(model_name, x_train, x_test, batch_size, epochs, callbacks=None):
    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    mol_autoencoder, mol_encoder = None, None
    if model_name == 'arnn_molecule_RNN_RNN':
        mol_autoencoder, mol_encoder = molecule_model_RNN_RNN(model_name)
        mol_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=callbacks)
    elif model_name == 'arnn_molecule_RNN_DNN':
        mol_autoencoder, mol_encoder = molecule_model_RNN_DNN(model_name)
        mol_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=callbacks)

    encoded_x_test = mol_encoder.predict(x_test_reshaped)
    encoded_x_train = mol_encoder.predict(x_train_reshaped)

    return encoded_x_train, encoded_x_test

def train_protein_model(model_name, x_train, x_test, batch_size, epochs, callbacks=None):
    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    prot_autoencoder, prot_encoder = None, None
    if model_name == 'arnn_protein_RNN_RNN':
        prot_autoencoder, prot_encoder = protein_model_RNN_RNN(model_name)
        prot_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=callbacks)
    elif model_name == 'arnn_protein_RNN_DNN':
        prot_autoencoder, prot_encoder = protein_model_RNN_DNN(model_name)
        prot_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=callbacks)

    encoded_x_test = prot_encoder.predict(x_test_reshaped)
    encoded_x_train = prot_encoder.predict(x_train_reshaped)

    return encoded_x_train, encoded_x_test

def train_interaction_model(model_name, dataset, batch_size, epochs, callbacks=None):
    model = interaction_model(model_name)
    model.fit(dataset['x_train'], dataset['y_train'], batch_size, epochs, callbacks=callbacks)
    predictions = model.predict(dataset['x_test'])
    print(measure_and_print_performance(model_name, dataset['name'], dataset['y_test'], predictions.flatten()))
