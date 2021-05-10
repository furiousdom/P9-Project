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
from data_loader import load_Y, load_mols_prots_Y

import tensorflow as tf
from tensorflow.keras import layers

import keras.backend as K

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
    decoded = Dense(300, activation='relu')(encoded)
    # decoded = Dense(200, activation='relu')(decoded)
    # decoded = Dense(300, activation='relu')(decoded)
    
    autoencoder = Model(inputs=XDinput, outputs=decoded, name=model_name)

    encoder = Model(inputs=XDinput, outputs=encoded)

    # Independent Decoder
    # encoded_input = Input(shape=(encoded.shape[1],))
    encoded_input = Input(shape=K.int_shape(encoded)[1:])
    decoded_output = Dense(300, activation='relu')(encoded_input)
    # decoded_output = Dense(200, activation='relu')(decoded_output)
    # decoded_output = Dense(300, activation='relu')(decoded_output)

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
    encoded = Dense(30, activation='relu')(encoded)

    # Decoder
    decoded = Dense(100, activation='relu')(encoded)
    # decoded = Dense(70, activation='relu')(decoded)
    # decoded = Dense(100, activation='relu')(decoded)
    
    autoencoder = Model(inputs=XTinput, outputs=decoded, name=model_name)

    encoder = Model(inputs=XTinput, outputs=encoded)

    # Independent Decoder
    # encoded_input = Input(shape=(encoded.shape[1],))
    encoded_input = Input(shape=K.int_shape(encoded)[1:])
    decoded_output = Dense(100, activation='relu')(encoded_input)
    # decoded_output = Dense(70, activation='relu')(decoded_output)
    # decoded_output = Dense(100, activation='relu')(decoded_output)

    decoder = Model(encoded_input, decoded_output) 

    metrics=['accuracy', 'mean_squared_error']
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(autoencoder.summary())
    return autoencoder, encoder, decoder

def interaction_model(model_name):
    model = tf.keras.models.Sequential(name=model_name)

    model.add(layers.Input(shape=(80,)))
    model.add(layers.Dense(700, activation='relu'))
    model.add(layers.Dense(500, activation='sigmoid'))
    # model.add(Dense(500, activation='relu'))
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

def train_molecule_model(model_name, x_train, x_test, batch_size, epochs, callbacks=None):
    checkpoint_callback = checkpoint(checkpoint_path(model_name))

    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    # print(f'x_train shape after reshape: {x_train.shape}')

    mol_autoencoder, mol_encoder, mol_decoder = molecule_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    mol_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=[checkpoint_callback])

    encoded_x_test = mol_encoder.predict(x_test_reshaped)
    encoded_x_train = mol_encoder.predict(x_train_reshaped)
    
    decoded_mols = mol_decoder.predict(encoded_x_test)

    print(calc_mean_squared_error(x_test.flatten(), decoded_mols.flatten()))
    return encoded_x_train, encoded_x_test

def train_protein_model(model_name, x_train, x_test, batch_size, epochs, callbacks=None):
    checkpoint_callback = checkpoint(checkpoint_path(model_name))

    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')

    # print(f'x_train shape after reshape: {x_train.shape}')

    prot_autoencoder, prot_encoder, prot_decoder = protein_model(model_name, NUM_FILTERS, FILTER_LENGTH1)
    prot_autoencoder.fit(x_train_reshaped, x_train, batch_size, epochs, callbacks=[checkpoint_callback])

    encoded_x_test = prot_encoder.predict(x_test_reshaped)
    encoded_x_train = prot_encoder.predict(x_train_reshaped)

    decoded_mols = prot_decoder.predict(encoded_x_test)

    print(calc_mean_squared_error(x_test.flatten(), decoded_mols.flatten()))
    return encoded_x_train, encoded_x_test

def train_interaction_model(model_name, dataset, batch_size, epochs, callbacks=None):
    #protein latent vector shape: (14000, 30)
    #molecule latent vector shape: (14000, 50) 
    checkpoint_callback = checkpoint(checkpoint_path(model_name))
    model = interaction_model(model_name)
    model.fit(dataset['x_train'], dataset['y_train'], batch_size, epochs, callbacks=[checkpoint_callback])
    predictions = model.predict(dataset['x_test'])
    print(measure_and_print_performance(dataset['name'], dataset['y_test'], predictions.flatten()))

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

def checkpoint_path(model_name, model_version=1): # TODO: Change model_version to be dynamic
    return f'./data/models/{model_name}/model_{model_version}.ckpt'

def checkpoint(checkpoint_path):
    return ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

def run_train_session_mols():
    molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
    molecules.drop(molecules.filter(regex="Unname"),axis=1, inplace=True)
    molecules = np.array(molecules)
    dataset = get_dataset_split(dataset_name, molecules, molecules)
    # print(f'x_train shape: {dataset["x_train"].shape}')
    # print(f'y_train shape: {dataset["y_train"].shape}')
    return train_molecule_model(dataset, 256, 1)

def run_train_session_prots():
    proteins = pd.read_csv(f'./data/datasets/{dataset_name}/proteins.csv')
    proteins.drop(proteins.filter(regex="Unname"),axis=1, inplace=True)
    proteins = np.array(proteins)
    dataset = get_dataset_split(dataset_name, proteins, proteins)
    # print(f'x_train shape: {dataset["x_train"].shape}')
    # print(f'y_train shape: {dataset["y_train"].shape}')
    return train_protein_model(dataset, 256, 1)

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

def run_train_session_ba():
    mols, prots, Y = load_mols_prots_Y(dataset_name)
    mol_train, mol_test = train_test_split(mols, train_size=0.8, random_state=0)
    prot_train, prot_test = train_test_split(prots, train_size=0.8, random_state=0)
    y_train, y_test = train_test_split(Y, train_size=0.8, random_state=0)
    mol_train_latent_vec, mol_test_latent_vec = train_molecule_model('molecule_autoencoder_2', mol_train, mol_test, 256, 1)
    prot_train_latent_vec, prot_test_latent_vec = train_protein_model('protein_autoencoder_2', prot_train, prot_test, 256, 1)
    dataset = combined_dataset(dataset_name, mol_train_latent_vec, mol_test_latent_vec, prot_train_latent_vec, prot_test_latent_vec, y_train, y_test)
    train_interaction_model('interaction_model_2', dataset, 256, 1)

def run_test_session(i_model_name, m_model_name, p_model_name):
    mols, prots, Y = load_mols_prots_Y(dataset_name)
    mol_train, mol_test = train_test_split(mols, train_size=0.8, random_state=0)
    prot_train, prot_test = train_test_split(prots, train_size=0.8, random_state=0)
    y_train, y_test = train_test_split(Y, train_size=0.8, random_state=0)
    mol_test_reshaped = mol_test.reshape(mol_test.shape[0], mol_test.shape[1], 1).astype('float32')
    prot_test_reshaped = prot_test.reshape(prot_test.shape[0], prot_test.shape[1], 1).astype('float32')

    mol_autoencoder, mol_encoder, mol_decoder = molecule_model(m_model_name, NUM_FILTERS, FILTER_LENGTH1)
    mol_autoencoder.load_weights(checkpoint_path(m_model_name))

    prot_autoencoder, prot_encoder, prot_decoder = protein_model(p_model_name, NUM_FILTERS, FILTER_LENGTH1)
    prot_autoencoder.load_weights(checkpoint_path(p_model_name))
    
    mol_test_latent_vec = mol_encoder.predict(mol_test_reshaped)
    prot_test_latent_vec = prot_encoder.predict(prot_test_reshaped)

    x_test = np.concatenate([mol_test_latent_vec, prot_test_latent_vec], axis=1)

    model = interaction_model(i_model_name)
    model.load_weights(checkpoint_path(i_model_name))
    predictions = model.predict(x_test)
    measure_and_print_performance(dataset_name, y_test, predictions.flatten()) 

run_train_session_ba()
# run_test_session('interaction_model', 'molecule_autoencoder', 'protein_autoencoder')