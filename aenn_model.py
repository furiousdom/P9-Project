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
    encoded_input = Input(shape=(encoded.shape[1],))
    decoded = Dense(100, activation='relu')(encoded_input)
    decoded = Dense(200, activation='relu')(decoded)
    decoded = Dense(300, activation='relu')(decoded)

    # Decoder
    # FC1 = Dense(1024, activation='relu')(encoded)
    # FC2 = Dropout(0.1)(FC1)
    # FC2 = Dense(1024, activation='relu')(FC2)
    # FC2 = Dropout(0.1)(FC2)
    # FC2 = Dense(512, activation='relu')(FC2)

    

    autoencoder = Model(inputs=XDinput, outputs=decoded, name=model_name)

    encoder = Model(inputs=XDinput, outputs=encoded)

    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder, decoder

def protein_model(NUM_FILTERS, FILTER_LENGTH):
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

    decoded = Dense(300, activation='relu')(FC2)

    autoencoder = Model(inputs=XTinput, outputs=decoded)
    encoder = Model(inputs=XTinput, outputs=encoded)
    decoder = Model(inputs=encoded, outputs=decoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder, decoder

def interaction_model():
    model = tf.keras.models.Sequential()

    model.add(layers.Input(shape=(400,)))
    model.add(layers.Dense(700, activation='relu'))
    model.add(layers.Dense(500, activation='sigmoid'))
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dense(100, activation='sigmoid'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))

    metrics=['accuracy', 'mean_squared_error']
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(model.summary())
    return model

def reshape_network_input(x_input):
    x_input = np.hsplit(x_input, [300])
    x_input[0] = x_input[0].reshape(x_input[0].shape[0], 300, 1).astype('float32')
    x_input[1] = x_input[1].reshape(x_input[1].shape[0], 100, 1).astype('float32')
    return x_input

def train(dataset, batch_size, epochs, callbacks=None):
    # x_train = reshape_network_input(dataset['x_train'])
    # x_test = reshape_network_input(dataset['x_test'])

    x_train = dataset['x_train']
    x_test = dataset['x_test']
    model_name = 'molecule_autoencoder'

    x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    mol_autoencoder = molecule_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    mol_autoencoder.fit(x_train, dataset['y_train'], batch_size, epochs)

    decoded_mols = mol_autoencoder.predict(x_test)

    calc_mean_squared_error(dataset['y_test'], decoded_mols)


    # mol_autoencoder, mol_encoder, mol_decoder = molecule_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    # mol_autoencoder.fit(x_train, dataset['y_train'], batch_size, epochs)

    # encoded_mols = mol_encoder.predict(x_test)
    # decoded_mols = mol_decoded.predict(encoded_mols)

    # calc_mean_squared_error(dataset['y_test'], decoded_mols)

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

dataset_name = 'kiba'
molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
molecules.drop(molecules.filter(regex="Unname"),axis=1, inplace=True)
molecules = np.array(molecules)
dataset = get_dataset_split(dataset_name, molecules, molecules)
train(dataset, 256, 10)
