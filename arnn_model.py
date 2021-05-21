import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding, GRU, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D
from performance_meter import measure_and_print_performance

import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from data_loader import load_Y, load_mols_prots_Y

import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import UpSampling1D, Flatten, MaxPooling1D, Reshape

NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2 = 32, 8, 4 # [8, 12], [4, 8]

# def molecule_model_test():
#     model = Sequential()
#     model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#     model.add(LSTM(50, return_sequences = False))
#     model.add(Dense(25))
#     model.add(Dense(1))

def molecule_model_RNN_RNN(model_name):

    # Encoder
    encoder_input = Input(shape=(300, 1))
    # encoded = Embedding(input_dim=300, output_dim=64)(encoder_input)
    encoded = LSTM(64)(encoder_input)
    encoded = Dense(50, activation='relu')(encoded)

    # Decoder
    decoded = Reshape((50, 1))(encoded)
    decoded = LSTM(64)(decoded)
    decoded = Dense(300, activation='relu')(decoded)

    autoencoder = Model(inputs=encoder_input, outputs=decoded, name=model_name)

    encoder = Model(inputs=encoder_input, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def molecule_model_RNN_FCNN(model_name):

    # Encoder
    encoder_input = Input(shape=(300, 1))
    # encoded = Embedding(input_dim=300, output_dim=64)(encoder_input)
    encoded = LSTM(64)(encoder_input)
    encoded = Dense(50, activation='relu')(encoded)

    # Decoder
    decoded = Dense(100, activation='relu')(encoded)
    decoded = Dense(200, activation='relu')(decoded)
    decoded = Dense(300, activation='relu')(decoded)

    autoencoder = Model(inputs=encoder_input, outputs=decoded, name=model_name)

    encoder = Model(inputs=encoder_input, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def protein_model_CONV_CONV(model_name, NUM_FILTERS, FILTER_LENGTH):
    # Encoder
    XTinput = Input(shape=(100, 1))
    encoded = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH, activation='relu', input_shape=(100, ))(XTinput)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)
    encoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='relu')(encoded)
    encoded = MaxPooling1D()(encoded)

    encoded = Flatten()(encoded)
    encoded = Dense(30, activation='relu')(encoded)

    decoded = Dense(864, activation='relu')(encoded)
    decoded = Reshape((9, -1))(decoded)
    decoded = UpSampling1D(4)(decoded)
    decoded = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = UpSampling1D()(decoded)
    decoded = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = UpSampling1D()(decoded)
    decoded = Conv1D(filters=1, kernel_size=FILTER_LENGTH, activation='relu')(decoded)
    decoded = Flatten()(decoded)
    decoded = Dense(100, activation='relu')(decoded)
    # decoded = Reshape((100, 1))(decoded)

    autoencoder = Model(inputs=XTinput, outputs=decoded, name=model_name)

    encoder = Model(inputs=XTinput, outputs=encoded)

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder

def interaction_model(model_name):
    model = tf.keras.models.Sequential(name=model_name)

    model.add(layers.Input(shape=(80,)))
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

def train_molecule_model(model_name, model_version, x_train, x_test, batch_size, epochs, callbacks=None):
    checkpoint_callback = checkpoint(checkpoint_path(model_name, model_version))

    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    mol_autoencoder, mol_encoder = molecule_model_RNN_RNN(model_name)
    mol_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=[checkpoint_callback])

    encoded_x_test = mol_encoder.predict(x_test_reshaped)
    encoded_x_train = mol_encoder.predict(x_train_reshaped)

    print(f'encoded_x_test shape after reshape: {encoded_x_test.shape}')

    return encoded_x_train, encoded_x_test

def train_protein_model(model_name, model_version, x_train, x_test, batch_size, epochs, callbacks=None):
    checkpoint_callback = checkpoint(checkpoint_path(model_name, model_version))

    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    # print(f'x_train shape after reshape: {x_train.shape}')

    prot_autoencoder, prot_encoder = protein_model_CONV_CONV(model_name, NUM_FILTERS, FILTER_LENGTH2)
    prot_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=[checkpoint_callback])

    encoded_x_test = prot_encoder.predict(x_test_reshaped)
    encoded_x_train = prot_encoder.predict(x_train_reshaped)

    return encoded_x_train, encoded_x_test

def train_interaction_model(model_name, model_version, dataset, batch_size, epochs, callbacks=None):
    #protein latent vector shape: (14000, 30)
    #molecule latent vector shape: (14000, 50)
    checkpoint_callback = checkpoint(checkpoint_path(model_name, model_version))
    model = interaction_model(model_name)
    model.fit(dataset['x_train'], dataset['y_train'], batch_size, epochs, callbacks=[checkpoint_callback])
    predictions = model.predict(dataset['x_test'])
    print(measure_and_print_performance(dataset['name'], dataset['y_test'], predictions.flatten()))

def reshape_network_input(x_input):
    x_input = np.hsplit(x_input, [300])
    x_input[0] = x_input[0].reshape(x_input[0].shape[0], 300, 1).astype('float32')
    x_input[1] = x_input[1].reshape(x_input[1].shape[0], 100, 1).astype('float32')
    return x_input

def combined_dataset(dataset_name, mol_train_latent_vec, mol_test_latent_vec, prot_train_latent_vec, prot_test_latent_vec, y_train, y_test):
    x_train = np.concatenate([mol_train_latent_vec, prot_train_latent_vec], axis=1)
    x_test = np.concatenate([mol_test_latent_vec, prot_test_latent_vec], axis=1)
    print(f'x_train.shape: {x_train.shape}')
    print(f'x_test.shape: {x_test.shape}')
    return {
        'name': dataset_name,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

def checkpoint_path(model_name, model_version=1):
    return f'./data/models/{model_name}/model_{model_version}.ckpt'

def checkpoint(checkpoint_path):
    return ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

def run_train_session_ba(dataset_name):
    mols, prots, Y = load_mols_prots_Y(dataset_name)
    mol_train, mol_test = train_test_split(mols, train_size=0.8, random_state=0)
    prot_train, prot_test = train_test_split(prots, train_size=0.8, random_state=0)
    y_train, y_test = train_test_split(Y, train_size=0.8, random_state=0)
    mol_train_latent_vec, mol_test_latent_vec = train_molecule_model('mol_auto_RNN_FCNN', 1, mol_train, mol_test, 256, 50)
    prot_train_latent_vec, prot_test_latent_vec = train_protein_model('protein_autoencoder_3', 2, prot_train, prot_test, 256, 1)
    dataset = combined_dataset(dataset_name, mol_train_latent_vec, mol_test_latent_vec, prot_train_latent_vec, prot_test_latent_vec, y_train, y_test)
    train_interaction_model('interaction_model_3', 3, dataset, 256, 1)

run_train_session_ba('kiba')
