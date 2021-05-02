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

def run_train_session(model_name, dataset_name, threshold, batch_size, epochs):
    X, Y = load_dataset(dataset_name, threshold)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.84, random_state=0)
    checkpoint_callback = checkpoint(checkpoint_path(model_name))
    base_model.train(dataset_name, x_train, x_test, y_train, y_test, batch_size, 63, checkpoint_callback)
    dcnn_model.train(dataset_name, x_train, x_test, y_train, y_test, batch_size, 30, checkpoint_callback)

def run_test_session():
    kiba_X, kiba_Y = load_dataset('kiba2', 12.1)
    davis_X, davis_Y = load_dataset('davis2', 7.0)
    datasets = [{
        'name': 'kiba2',
        'x_test': kiba_X,
        'y_test': kiba_Y
    }, {
        'name': 'davis2',
        'x_test': davis_X,
        'y_test': davis_Y
    }]
    base_model.test(datasets, checkpoint_path('base_model'))
    dcnn_model.test(datasets, checkpoint_path('dcnn_model'))

run_train_session('kiba_model_ba', 'kiba2', 12.1, 8, 50)
run_train_session('davis_model_ba', 'davis2', 7.0, 8, 50)

# run_test_session()