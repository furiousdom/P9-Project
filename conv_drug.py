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
# NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2 = [32], [8], [4]
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

def get_model(FEATURE_LENGTH, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    # XDinput = Input(shape=(FLAGS.max_smi_len, FLAGS.charsmiset_size))
    # XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))

    XDinput = Input(shape=(1054, 300))
    XTinput = Input(shape=(1054, 100))

    # encode_smiles = Conv1D(1, kernel_size = FILTER_LENGTH1, input_shape = (1054 , 300))(XDinput)
    # encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encode_smiles)
    # encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encode_smiles)
    # encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]

    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1, input_shape=(300, ))(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]


    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1, input_shape=(100, ))(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])
    # encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=1)
    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # predictions = Dense(1, kernel_initializer='normal')(FC2) 

    # interactionModel = Model(inputs=[XDNUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2 = [32], [8], [4]input, XTinput], outputs=[predictions])
    # interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']

    predictions = Dense(2, activation='softmax')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    print(interactionModel.summary())
    # plot_model(interactionModel, to_file='data/figures/model.png')

    return interactionModel

aau_molecules, aau_proteins = load_dataset('aau')
aau_X = pd.concat([aau_molecules, aau_proteins], axis=1)
aau_Y = create_aau_output(aau_X)

kiba_X, kiba_Y, davis_X, davis_Y = load_test_datasets()

x_train, x_test, y_train, y_test = train_test_split(aau_X, aau_Y, test_size=0.15, random_state=0)

x_train_split = np.hsplit(x_train, [300])
# x_train_drugs = x_train_split[0]
# x_train_targets = x_train_split[1]
x_test_split = np.hsplit(x_test, [300])

# np.expand_dims(x_train_split, axis=())
# x_train_split[:, :,]

model = get_model(x_train.shape[1], NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2) 

model.fit(x_train_split, y_train, batch_size = 8, epochs = 63)

predictions = model.predict(x_test_split)

check_and_print_accuracy('kiba', y_test, predictions)

# x_train, x_test, y_train, y_test = train_test_split(davis_X, davis_Y, test_size=0.8, random_state=0)

# predictions = model.predict(x_test)

# check_and_print_accuracy('davis', y_test, predictions)
