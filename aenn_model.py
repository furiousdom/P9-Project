import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D
from performance_meter import calc_mean_squared_error
from performance_meter import measure_and_print_performance

import pandas as pd
from data_loader import load_dataset
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2 = 32, 8, 4 # [8, 12], [4, 8]

def molecule_model(model_name, NUM_FILTERS, FILTER_LENGTH):
    # Encoder
    XDinput = Input(shape=(300, 1))
    encoded = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(XDinput)
    encoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encoded)
    encoded = GlobalMaxPooling1D()(encoded)
    encoded = Dense(50, activation='relu')(encoded)

    # Decoder
    # encoded_input = Input(shape=(encoded.shape[1],))
    decoded = Dense(100, activation='relu')(encoded)
    decoded = Dense(200, activation='relu')(decoded)
    decoded = Dense(300, activation='relu')(decoded)

    # Decoder (it's not)
    # FC1 = Dense(1024, activation='relu')(encoded)
    # FC2 = Dropout(0.1)(FC1)
    # FC2 = Dense(1024, activation='relu')(FC2)
    # FC2 = Dropout(0.1)(FC2)
    # FC2 = Dense(512, activation='relu')(FC2)

    autoencoder = Model(inputs=XDinput, outputs=decoded, name=model_name)

    encoder = Model(inputs=XDinput, outputs=encoded)

    # decoder = Model(encoded, decoded) BREAKS ON THIS LINE => ValueError: Graph disconnected: cannot obtain value for tensor Tensor("input_1:0", shape=(None, 300, 1), dtype=float32) at layer "conv1d". The following previous layers were accessed without issue: []

    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded, decoder_layer(encoded)) BREAKS ON THIS LINE => ValueError: Input 0 of layer dense_3 is incompatible with the layer: expected axis -1 of input shape to have value 200 but received input with shape [None, 50]

    # Independent Decoder
    encoded_input = Input(shape=(encoded.shape[1],))
    decoded_output = Dense(100, activation='relu')(encoded_input)
    decoded_output = Dense(200, activation='relu')(decoded_output)
    decoded_output = Dense(300, activation='relu')(decoded_output)

    # # decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoded_output)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder, decoder

def protein_model(model_name, NUM_FILTERS, FILTER_LENGTH):
    # Encoder
    XTinput = Input(shape=(100, 1))
    encoded = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(XTinput)
    encoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(encoded)
    encoded = GlobalMaxPooling1D()(encoded)

    # Decoder
    FC1 = Dense(1024, activation='relu')(encoded)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    decoded = Dense(100, activation='relu')(FC2)

    autoencoder = Model(inputs=XTinput, outputs=decoded)
    # encoder = Model(inputs=XTinput, outputs=encoded)
    # decoder = Model(inputs=encoded, outputs=decoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder#, encoder, decoder

def interaction_model():
    model = keras.models.Sequential()

    model.add(Input(shape=(400,)))
    model.add(Dense(700, activation='relu'))
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='relu'))

    metrics=['accuracy', 'mean_squared_error']
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(model.summary())
    return model

def reshape_network_input(x_input):
    x_input = np.hsplit(x_input, [300])
    x_input[0] = x_input[0].reshape(x_input[0].shape[0], 300, 1).astype('float32')
    x_input[1] = x_input[1].reshape(x_input[1].shape[0], 100, 1).astype('float32')
    return x_input

def train_mols(dataset, batch_size, epochs, callbacks=None):
    x_train = dataset['x_train']
    x_test = dataset['x_test']
    model_name = 'molecule_autoencoder'

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    print(f'x_train shape after reshape: {x_train.shape}')

    # mol_autoencoder, mol_encoder, mol_decoder = molecule_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    # mol_autoencoder.fit(x_train, dataset['y_train'], batch_size, epochs)

    # decoded_mols = mol_autoencoder.predict(x_test)

    # calc_mean_squared_error(dataset['y_test'], decoded_mols)


    mol_autoencoder, mol_encoder, mol_decoder = molecule_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    mol_autoencoder.fit(x_train, dataset['y_train'], batch_size, epochs)

    encoded_mols = mol_encoder.predict(x_test)
    decoded_mols = mol_decoder.predict(encoded_mols)

    print(calc_mean_squared_error(dataset['y_test'].flatten(), decoded_mols.flatten()))

    # model = get_model(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2)
    # model.fit(x_train, dataset['y_train'], batch_size, epochs, callbacks=callbacks)
    # predictions = model.predict(x_test)
    # measure_and_print_performance(dataset['name'], dataset['y_test'], predictions.flatten())

def train_prots(dataset, batch_size, epochs, callbacks=None):
    x_train = dataset['x_train']
    x_test = dataset['x_test']
    model_name = 'protein_autoencoder'

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    print(f'x_train shape after reshape: {x_train.shape}')

    # mol_autoencoder, mol_encoder, mol_decoder = molecule_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    # mol_autoencoder.fit(x_train, dataset['y_train'], batch_size, epochs)

    # decoded_mols = mol_autoencoder.predict(x_test)

    # calc_mean_squared_error(dataset['y_test'], decoded_mols)


    prot_autoencoder = protein_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    prot_autoencoder.fit(x_train, dataset['y_train'], batch_size, epochs)

    # encoded_mols = prot_encoder.predict(x_test)
    # decoded_mols = prot_decoder.predict(encoded_mols)

    print(calc_mean_squared_error(dataset['y_test'].flatten(), decoded_mols.flatten()))

    # model = get_model(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2)
    # model.fit(x_train, dataset['y_train'], batch_size, epochs, callbacks=callbacks)
    # predictions = model.predict(x_test)
    # measure_and_print_performance(dataset['name'], dataset['y_test'], predictions.flatten())

def get_dataset_split(dataset_name, X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
    return {
        'name': dataset_name,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

# dataset_name = 'kiba'
# molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
# molecules.drop(molecules.filter(regex="Unname"),axis=1, inplace=True)
# molecules = np.array(molecules)
# print(f'molecules shape: {molecules.shape}')
# dataset = get_dataset_split(dataset_name, molecules, molecules)
# print(f'x_train shape: {dataset["x_train"].shape}')
# print(f'y_train shape: {dataset["y_train"].shape}')
# train_mols(dataset, 1, 10)

dataset_name = 'kiba'
proteins = np.array(pd.read_csv(f'./data/datasets/{dataset_name}/proteins.csv'))
dataset = get_dataset_split(dataset_name, proteins, proteins)
print(f'x_train shape: {dataset["x_train"].shape}')
print(f'y_train shape: {dataset["y_train"].shape}')
train_prots(dataset, 256, 10)
