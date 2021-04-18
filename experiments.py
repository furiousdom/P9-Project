import base_model
import dcnn_model

from data_loader import load_dataset
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def checkpoint_path(model_name, model_version=1): # TODO: Change model_version to be dynamic
    return f'./data/models/{model_name}/model_{model_version}.ckpt'

def checkpoint(checkpoint_path):
    return ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

def run_train_session(model_name, batch_size, epochs):
    aau_X, aau_Y = load_dataset('aau1000', 1)
    x_train, x_test, y_train, y_test = train_test_split(aau_X, aau_Y, train_size=0.85, random_state=0)
    checkpoint_callback = checkpoint(checkpoint_path(model_name))
    if model_name is 'base_model':
        base_model.train('aau1000', x_train, x_test, y_train, y_test, batch_size, epochs, checkpoint_callback)
    elif model_name is 'dcnn_model':
        dcnn_model.train('aau1000', x_train, x_test, y_train, y_test, batch_size, epochs, checkpoint_callback)

def run_test_session():
    kiba_X, kiba_Y = load_dataset('kiba', 12.1)
    davis_X, davis_Y = load_dataset('davis', 7.0)
    datasets = [{
        'name': 'kiba',
        'x_test': kiba_X,
        'y_test': kiba_Y
    }, {
        'name': 'davis',
        'x_test': davis_X,
        'y_test': davis_Y
    }]
    base_model.test(datasets, checkpoint_path('base_model'))
    dcnn_model.test(datasets, checkpoint_path('dcnn_model'))
