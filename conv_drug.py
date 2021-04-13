import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_handler import load_binary_scores

import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers

NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2 = 32, 8, 4
# [8, 12], [4, 8]

def load_dataset(dataset_name):
    molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
    proteins = pd.read_csv(f'./data/datasets/{dataset_name}/proteins.csv')
    return molecules, proteins

def concat_mol_prot(molecules_data_frame, protein_data_frame):
    return pd.concat([molecules_data_frame, protein_data_frame], axis=1)

def create_aau_output(df):
    Y = np.zeros((df.shape[0], 2), dtype=int)
    Y[:1000] = [0, 1]
    Y[1000:] = [1, 0]
    return Y

def load_train_dataset():
    aau_molecules, aau_proteins = load_dataset('aau')
    aau_X = pd.concat([aau_molecules, aau_proteins], axis=1)
    aau_X = np.array(aau_X)
    aau_Y = create_aau_output(aau_X)
    return aau_X, aau_Y

def load_test_datasets():
    kiba_molecules, kiba_proteins = load_dataset('kiba')
    davis_molecules, davis_proteins = load_dataset('davis')

    kiba = pd.concat([kiba_molecules, kiba_proteins], axis=1)
    davis = pd.concat([davis_molecules, davis_proteins], axis=1)

    kiba_X = np.array(kiba)
    davis_X = np.array(davis)

    kiba_Y = np.array(load_binary_scores('./data/datasets/kiba/scores.txt', 12.1))
    davis_Y = np.array(load_binary_scores('./data/datasets/davis/scores.txt', 7.0, True))
    return kiba_X, kiba_Y, davis_X, davis_Y

def check_and_print_accuracy(dataset_name, y_test, predictions):
    temp_y_test = []
    temp_predictions = []
    for i in range(y_test.shape[0]):
        temp_predictions.append(np.argmax(predictions[i]))
        temp_y_test.append(np.argmax(y_test[i]))

    counter = 0
    for i in range(y_test.shape[0]):
        if temp_y_test[i] == temp_predictions[i]:
            counter += 1

    f = open(f'./data/{dataset_name}-result.txt', 'w')
    f.write('actual state\t predicted state\n')
    for i in range(y_test.shape[0]):
        f.write(f'{temp_y_test[i]}\t{temp_predictions[i]}\n')
    f.close()

    acc = counter * 100 / y_test.shape[0]

    print(f'Accuracy on test set {dataset_name} is {acc}, predicted {counter} out of {y_test.shape[0]}')

def get_model(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(300, 1))
    XTinput = Input(shape=(100, 1))

    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]


    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    predictions = Dense(2, activation='softmax')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    print(interactionModel.summary())
    # plot_model(interactionModel, to_file='data/figures/model.png')

    return interactionModel

def split_reshape(data_X, data_Y, test_size=0.15):
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=test_size, random_state=0)

    x_train_split = np.hsplit(x_train, [300])
    x_test_split = np.hsplit(x_test, [300])

    x_train_split[0] = x_train_split[0].reshape(x_train_split[0].shape[0], 300, 1).astype('float32')
    x_train_split[1] = x_train_split[1].reshape(x_train_split[1].shape[0], 100, 1).astype('float32')

    x_test_split[0] = x_test_split[0].reshape(x_test_split[0].shape[0], 300, 1).astype('float32')
    x_test_split[1] = x_test_split[1].reshape(x_test_split[1].shape[0], 100, 1).astype('float32')
    return x_train_split, x_test_split, y_train, y_test

aau_X, aau_Y = load_train_dataset()
kiba_X, kiba_Y, davis_X, davis_Y = load_test_datasets()

x_train_split, x_test_split, y_train, y_test = split_reshape(aau_X, aau_Y, test_size=0.15)

model = get_model(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2) 

model.fit(x_train_split, y_train, batch_size = 8, epochs = 45)

predictions = model.predict(x_test_split)

check_and_print_accuracy('aau', y_test, predictions)

x_train_split, x_test_split, y_train, y_test = split_reshape(kiba_X, kiba_Y, test_size=0.95)

predictions = model.predict(x_test_split)

check_and_print_accuracy('kiba', y_test, predictions)

x_train_split, x_test_split, y_train, y_test = split_reshape(davis_X, davis_Y, test_size=0.95)

predictions = model.predict(x_test_split)

check_and_print_accuracy('davis', y_test, predictions)
