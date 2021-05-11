import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D
from performance_meter import  measure_and_print_performance

NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2 = 32, 8, 4 # [8, 12], [4, 8]

def get_model(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(300, 1))
    XTinput = Input(shape=(100, 1))

    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', input_shape=(300, ))(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1, activation='relu')(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1, activation='relu')(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]

    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', input_shape=(100, ))(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2, activation='relu')(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2, activation='relu')(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    predictions = Dense(1, activation='relu')(FC2)

    model = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    metrics=['accuracy', 'mean_squared_error']
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(model.summary())
    # plot_model(model, to_file='data/figures/model.png')
    return model

def reshape_network_input(x_input):
    x_input = np.hsplit(x_input, [300])
    x_input[0] = x_input[0].reshape(x_input[0].shape[0], 300, 1).astype('float32')
    x_input[1] = x_input[1].reshape(x_input[1].shape[0], 100, 1).astype('float32')
    return x_input

def train(dataset, batch_size, epochs, callbacks=None):
    x_train = reshape_network_input(dataset['x_train'])
    x_test = reshape_network_input(dataset['x_test'])
    model = get_model(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2)
    model.fit(x_train, dataset['y_train'], batch_size, epochs, callbacks=callbacks)
    predictions = model.predict(x_test)
    measure_and_print_performance(dataset['name'], dataset['y_test'], predictions.flatten())

def test(datasets, checkpoint_path):
    model = get_model(NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2)
    model.load_weights(checkpoint_path)
    for dataset in datasets:
        x_test = reshape_network_input(dataset['x_test'])
        predictions = model.predict(x_test)
        measure_and_print_performance(dataset['name'], dataset['y_test'], predictions.flatten())
