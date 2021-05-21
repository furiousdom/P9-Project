import base_model
import dcnn_model
import auen_model
import arnn_model

import numpy as np
from copy import Error
from data_loader import load_dataset
from data_loader import load_mols_prots_Y
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def compatible_dataset(version_of_models, dataset_name):
    if dataset_name == 'kiba' and version_of_models % 2 != 0: return True
    elif dataset_name == 'davis' and version_of_models % 2 == 0: return True
    else: return False

def checkpoint_path(model_name, model_version=1):
    return f'./data/models/{model_name}/model_{model_version}.ckpt'

def checkpoint(checkpoint_path):
    return ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

def get_dataset_split(dataset_name, X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.84, random_state=0)
    return {
        'name': dataset_name,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

def combine_latent_vecs(dataset_name, mol_train_latent_vecs, mol_test_latent_vecs, prot_train_latent_vecs, prot_test_latent_vecs, y_train, y_test):
    x_train = np.concatenate([mol_train_latent_vecs, prot_train_latent_vecs], axis=1)
    x_test = np.concatenate([mol_test_latent_vecs, prot_test_latent_vecs], axis=1)
    print(f'x_train.shape: {x_train.shape}')
    print(f'x_test.shape: {x_test.shape}')
    return {
        'name': dataset_name,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

def run_network_train_session(model_name, model_version, dataset_name, threshold=None, batch_size=256):
    X, Y = load_dataset(dataset_name)
    dataset = get_dataset_split(dataset_name, X, Y)
    checkpoint_callback = checkpoint(checkpoint_path('base_' + model_name))
    base_model.train(dataset, batch_size, 128, [checkpoint_callback])
    checkpoint_callback = checkpoint(checkpoint_path('dcnn_' + model_name))
    dcnn_model.train(dataset, batch_size, 100, [checkpoint_callback])

def run_network_test_session(model_name, model_version):
    kiba_X, kiba_Y = load_dataset('kiba')
    davis_X, davis_Y = load_dataset('davis')
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

def run_autoencoder_train_session(model_names, version_of_models, dataset_name):
    if not compatible_dataset(version_of_models, dataset_name):
        raise Error('Version of models not compatible with dataset.')
    mols, prots, Y = load_mols_prots_Y(dataset_name)
    mol_train, mol_test = train_test_split(mols, train_size=0.84, random_state=0)
    prot_train, prot_test = train_test_split(prots, train_size=0.84, random_state=0)
    y_train, y_test = train_test_split(Y, train_size=0.84, random_state=0)
    if model_names[0] == 'auen':
        mol_train_latent_vec, mol_test_latent_vec = auen_model.train_molecule_model(model_names[1], version_of_models, mol_train, mol_test, 256, 1)
        prot_train_latent_vec, prot_test_latent_vec = auen_model.train_protein_model(model_names[2], version_of_models, prot_train, prot_test, 256, 1)
        dataset = combine_latent_vecs(dataset_name, mol_train_latent_vec, mol_test_latent_vec, prot_train_latent_vec, prot_test_latent_vec, y_train, y_test)
        auen_model.train_interaction_model(model_names[3], version_of_models, dataset, 256, 1)
    elif model_names[0] == 'arnn':
        mol_train_latent_vec, mol_test_latent_vec = arnn_model.train_molecule_model(model_names[1], version_of_models, mol_train, mol_test, 256, 1)
        prot_train_latent_vec, prot_test_latent_vec = arnn_model.train_protein_model(model_names[2], version_of_models, prot_train, prot_test, 256, 1)
        dataset = combine_latent_vecs(dataset_name, mol_train_latent_vec, mol_test_latent_vec, prot_train_latent_vec, prot_test_latent_vec, y_train, y_test)
        arnn_model.train_interaction_model(model_names[3], version_of_models, dataset, 256, 1)

def run_autoencoder_test_session():
    pass

models = ['auen', 'auen_molecule', 'auen_protein', 'auen_interaction']
version = 1
dataset = 'kiba'

run_autoencoder_train_session(models, version, dataset)
