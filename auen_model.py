from keras.models import Model, Sequential
from keras.layers import Conv1D, UpSampling1D, MaxPooling1D
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D # TODO
from keras.layers import Input, Flatten, Reshape, Dense, Dropout
from keras.activations import softmax
from performance_meter import measure_and_print_performance

NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2 = 32, 8, 4 # [8, 12], [4, 8]

def molecule_model_CNN_CNN(model_name, NUM_FILTERS, FILTER_LENGTH):
    # Encoder
    drug = Input(shape=(100, 64))

    encoded = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH, activation='relu', input_shape=(None, ))(drug)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)

    encoded = Flatten()(encoded)
    encoded = Dense(50, activation='relu')(encoded)

    # Decoder
    decoded = Dense(2976, activation='relu')(encoded)
    decoded = Reshape((31, 96))(decoded)

    decoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = UpSampling1D()(decoded)
    decoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = UpSampling1D()(decoded)
    decoded = Conv1D(filters=1, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = UpSampling1D(4)(decoded)
    decoded = Flatten()(decoded)
    decoded = Dense(6400, activation='relu')(decoded)
    decoded = Reshape((100, 64))(decoded)
    decoded = softmax(decoded)

    autoencoder = Model(inputs=drug, outputs=decoded, name=model_name)

    encoder = Model(inputs=drug, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def molecule_model_CNN_DNN(model_name, NUM_FILTERS, FILTER_LENGTH):
    # Encoder
    drug = Input(shape=(100, 64))
    encoded = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH, activation='relu', input_shape=(None, ))(drug)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='sigmoid')(encoded)
    encoded = MaxPooling1D()(encoded)

    encoded = Flatten()(encoded)
    encoded = Dense(50, activation='relu')(encoded)

    # Decoder
    decoded = Dense(1920, activation='sigmoid')(encoded)
    decoded = Dense(2560, activation='relu')(decoded)
    decoded = Dense(6400, activation='relu')(decoded)
    decoded = Reshape((100, 64))(decoded)
    decoded = softmax(decoded)

    autoencoder = Model(inputs=drug, outputs=decoded, name=model_name)

    encoder = Model(inputs=drug, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def protein_model_CNN_CNN(model_name, NUM_FILTERS, FILTER_LENGTH):
    # Encoder
    target = Input(shape=(1000, 25))
    encoded = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH, activation='relu', input_shape=(None, ))(target)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)

    encoded = Flatten()(encoded)
    encoded = Dense(250, activation='relu')(encoded)

    # Decoder
    decoded = Dense(864, activation='relu')(encoded)
    decoded = Reshape((9, -1))(decoded)

    decoded = UpSampling1D(4)(decoded)
    decoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = UpSampling1D()(decoded)
    decoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = UpSampling1D()(decoded)
    decoded = Conv1D(filters=1, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = Flatten()(decoded)
    decoded = Dense(25000, activation='relu')(decoded)
    decoded = Reshape((1000, 25))(decoded)
    decoded = softmax(decoded)

    autoencoder = Model(inputs=target, outputs=decoded, name=model_name)

    encoder = Model(inputs=target, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def protein_model_CNN_DNN(model_name, NUM_FILTERS, FILTER_LENGTH):
    # Encoder
    target = Input(shape=(1000, 25))
    encoded = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH, activation='relu', input_shape=(None, ))(target)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)

    encoded = Flatten()(encoded)
    encoded = Dense(250, activation='relu')(encoded)

    # Decoder
    decoded = Dense(1000, activation='sigmoid')(encoded)
    decoded = Dense(10000, activation='relu')(decoded)
    decoded = Dense(25000, activation='relu')(decoded)
    decoded = Reshape((1000, 25))(decoded)
    decoded = softmax(decoded)

    autoencoder = Model(inputs=target, outputs=decoded, name=model_name)

    encoder = Model(inputs=target, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def interaction_model(model_name):
    model = Sequential(name=model_name)

    model.add(Input(shape=(300,)))
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
    mol_autoencoder, mol_encoder = None, None
    if model_name == 'auen_molecule_CNN_CNN':
        mol_autoencoder, mol_encoder = molecule_model_CNN_CNN(model_name, NUM_FILTERS, FILTER_LENGTH1)
        mol_autoencoder.fit(x_train, x_train, batch_size, epochs, callbacks=callbacks)
    elif model_name == 'auen_molecule_CNN_DNN':
        mol_autoencoder, mol_encoder = molecule_model_CNN_DNN(model_name, NUM_FILTERS, FILTER_LENGTH1)
        mol_autoencoder.fit(x_train, x_train, batch_size, epochs, callbacks=callbacks)

    # x = mol_autoencoder.predict(x_test[:2])
    # with open('./data/x.txt', 'w') as file:
    #     file.write(str(x))
    # print(x)
    encoded_x_test = mol_encoder.predict(x_test)
    encoded_x_train = mol_encoder.predict(x_train)

    return encoded_x_train, encoded_x_test

def train_protein_model(model_name, x_train, x_test, batch_size, epochs, callbacks=None):
    prot_autoencoder, prot_encoder = None, None
    if model_name == 'auen_protein_CNN_CNN':
        prot_autoencoder, prot_encoder = protein_model_CNN_CNN(model_name, NUM_FILTERS, FILTER_LENGTH2)
        prot_autoencoder.fit(x_train, x_train, batch_size, epochs, callbacks=callbacks)
    elif model_name == 'auen_protein_CNN_DNN':
        prot_autoencoder, prot_encoder = protein_model_CNN_DNN(model_name, NUM_FILTERS, FILTER_LENGTH2)
        prot_autoencoder.fit(x_train, x_train, batch_size, epochs, callbacks=callbacks)

    encoded_x_test = prot_encoder.predict(x_test)
    encoded_x_train = prot_encoder.predict(x_train)

    return encoded_x_train, encoded_x_test

def train_interaction_model(model_name, dataset, batch_size, epochs, callbacks=None):
    model = interaction_model(model_name)
    model.fit(dataset['x_train'], dataset['y_train'], batch_size, epochs, callbacks=callbacks)
    predictions = model.predict(dataset['x_test'])
    print(measure_and_print_performance(model_name, dataset['name'], dataset['y_test'], predictions.flatten()))
