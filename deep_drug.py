import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from data_handler import load_binary_scores

class_names = ['no-interaction', 'ineraction']
checkpoint_path = './data/models/cpTraining1.ckpt'

def checkpoint():
    # checkpoint_dir = path.dirname(checkpoint_path)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

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

cp_callback = checkpoint()

kiba_molecules, kiba_proteins = load_dataset('kiba')
davis_molecules, davis_proteins = load_dataset('davis')

del kiba_molecules['Unnamed: 0']
del davis_molecules['Unnamed: 0']

kiba = pd.concat([kiba_molecules, kiba_proteins], axis=1)
davis = pd.concat([davis_molecules, davis_proteins], axis=1)

kiba_X = np.array(kiba)
davis_X = np.array(davis)

kiba_Y = np.array(load_binary_scores('./data/datasets/kiba/scores.txt', 12.1))
davis_Y = np.array(load_binary_scores('./data/datasets/davis/scores.txt', 7.0, True))

model = tf.keras.models.Sequential()

model.add(layers.Input(shape=(kiba_X.shape[1],)))
model.add(layers.Dense(700, activation='relu'))
model.add(layers.Dense(500, activation='sigmoid'))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(100, activation='sigmoid'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# model.fit(x_train, y_train, batch_size = 8, epochs = 63, callbacks=[cp_callback])
model.load_weights(checkpoint_path)

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


x_train, x_test, y_train, y_test = train_test_split(kiba_X, kiba_Y, test_size=0.8, random_state=0)

predictions = model.predict(x_test)

check_and_print_accuracy('kiba', y_test, predictions)

x_train, x_test, y_train, y_test = train_test_split(davis_X, davis_Y, test_size=0.8, random_state=0)

predictions = model.predict(x_test)

check_and_print_accuracy('davis', y_test, predictions)
