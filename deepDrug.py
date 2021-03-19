import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import math

class_names = ['no-interaction', 'ineraction']
checkpoint_path = './data/models/cpTraining1.ckpt'

def checkpoint():
    # checkpoint_dir = path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

# from dataHandler import readFASTAsFromFile
# from protvec import sequences2protvecsCSV
# sequences = readFASTAsFromFile('./data/proteins_FASTA.txt')
# sequences2protvecsCSV('./data/standard/proteinDataset.csv', sequences)

# moleculesDataFrame = pd.read_csv('./data/standard/moleculeDataset.csv')
# proteinDataFrame = pd.read_csv('./data/standard/proteinDataset.csv')

# df = pd.concat([moleculesDataFrame, proteinDataFrame], axis=1)

# X = np.array(df)
# Y = np.zeros((df.shape[0], 2), dtype=int)
# Y[:1000] = [0, 1]
# Y[1000:] = [1, 0]

cp_callback = checkpoint()

kibaMolecules = pd.read_csv('./data/kiba/kibaMolecules.csv')
kibaProtein = pd.read_csv('./data/kiba/kibaProteins.csv')
davisMolecules = pd.read_csv('./data/davis/davisMolecules.csv')
davisProtein = pd.read_csv('./data/davis/davisProteins.csv')

del kibaMolecules['Unnamed: 0']
del davisMolecules['Unnamed: 0']

kiba = pd.concat([kibaMolecules, kibaProtein], axis=1)
davis = pd.concat([davisMolecules, davisProtein], axis=1)

kiba_X = np.array(kiba)
davis_X = np.array(davis)

def turnToBinary(filename, threshold, preprocess = False):
    scoresList = []
    f = open(filename, 'r')
    for line in f:
        score = float(line)
        if preprocess:
            score = -1 * math.log10(score/pow(10, 9))
        if score >= threshold:
            scoresList.append([0, 1])
        else:
            scoresList.append([1, 0])
    f.close()
    return scoresList

kiba_Y = np.array(turnToBinary('./data/kiba/kibaScores.txt', 12.1))
davis_Y = np.array(turnToBinary('./data/davis/davisScores.txt', 7.0, True))

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = tf.keras.models.Sequential()

x_train, x_test, y_train, y_test = train_test_split(kiba_X, kiba_Y, test_size=0.8, random_state=0)

model.add(layers.Input(shape=(x_train.shape[1],)))
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

def checkAndPrintAccuracy(datasetName, y_test, predictions):
    temp_y_test = []
    temp_predictions = []
    for i in range(y_test.shape[0]):
        temp_predictions.append(np.argmax(predictions[i]))
        temp_y_test.append(np.argmax(y_test[i]))

    counter = 0
    for i in range(y_test.shape[0]):
        if temp_y_test[i] == temp_predictions[i]:
            counter += 1

    # print(f'{datasetName} {temp_predictions}')

    f = open(f'./data/{datasetName}-result.txt', 'w')
    f.write('actual state\t predicted state\n')
    for i in range(y_test.shape[0]):
        f.write(f'{temp_y_test[i]}\t{temp_predictions[i]}\n')
    f.close()

    acc = counter * 100 / y_test.shape[0]

    print(f'Accuracy on test set {datasetName} is {acc}, predicted {counter} out of {y_test.shape[0]}')


predictions = model.predict(x_test)

checkAndPrintAccuracy('kiba', y_test, predictions)

x_train, x_test, y_train, y_test = train_test_split(davis_X, davis_Y, test_size=0.8, random_state=0)

predictions = model.predict(x_test)

checkAndPrintAccuracy('davis', y_test, predictions)

# model.load_weights(checkpoint_path)

# loss, acc = model.evaluate(x_test, y_test, verbose=2)
# print(f'Restored model, accuracy: {100 * acc}')
