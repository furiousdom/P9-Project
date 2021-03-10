import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dataHandler import printDfShapeHeadTail

moleculesDataFrame = pd.read_csv('./data/moleculeDatasetWithHeaders.csv')
proteinDataFrame = pd.read_csv('./data/proteinOptimumDatasetWithHeaders.csv')

df = pd.concat([moleculesDataFrame, proteinDataFrame], axis=1)

# scaler = MinMaxScaler(feature_range = (0, 1))
# scaled_data = scaler.fit_transform(df)

class_names = ['no-interaction', 'ineraction']

X = np.array(df)
Y = np.zeros((df.shape[0],), dtype=int)
Y[:999] = 1

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = tf.keras.models.Sequential()

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
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size = 1, epochs = 6)

predictions = model.predict(x_test)

temp = []
for prediction in predictions:
    temp.append(np.argmax(prediction))

counter = 0
for i in range(y_test.size):
    if y_test[i] == temp[i]:
        counter += 1

acc = counter * 100 / y_test.size

print(f'Accuracy on test set is {acc}, predicted {counter} out of {y_test.size}')
