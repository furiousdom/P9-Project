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

from preprocess import DataSet, AeDataSet

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

def get_simple_dataset_split(dataset_name, X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.33, random_state=0)
    return {
        'name': dataset_name,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

def get_dataset_split(dataset_name, molecules, proteins, Y):
    mol_train, mol_test = train_test_split(molecules, train_size=0.84, random_state=0)
    prot_train, prot_test = train_test_split(proteins, train_size=0.84, random_state=0)
    y_train, y_test = train_test_split(Y, train_size=0.84, random_state=0)
    return {
        'name': dataset_name,
        'mol_train': mol_train,
        'mol_test': mol_test,
        'prot_train': prot_train,
        'prot_test': prot_test,
        'y_train': y_train,
        'y_test': y_test
    }

def combine_latent_vecs(dataset_name, mol_train_latent_vecs, mol_test_latent_vecs, prot_train_latent_vecs, prot_test_latent_vecs, y_train, y_test):
    # print(f'mol_train_latent_vecs shape: {mol_test_latent_vecs.shape}')
    # print(f'y_train shape: {y_train.shape}')
    # print(f'y_0: {y_train[0]}')
    # print(f'y_1: {y_train[1]}')
    x_train = np.concatenate([mol_train_latent_vecs, prot_train_latent_vecs], axis=1)
    x_test = np.concatenate([mol_test_latent_vecs, prot_test_latent_vecs], axis=1)
    # print(f'x_train.shape: {x_train.shape}')
    # print(f'x_test.shape: {x_test.shape}')
    return {
        'name': dataset_name,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

def run_network_train_session(model_name, model_version, dataset_name, threshold=None, epochs=100, batch_size=256):
    if not compatible_dataset(model_version, dataset_name):
        raise Error('Version of models not compatible with dataset.')
    # X, Y = load_dataset(dataset_name, threshold)
    # dataset = get_dataset_split(dataset_name, X, Y)
    mols, prots, Y = DataSet(dataset_name).parse_data()
    mols, prots, Y = np.asarray(mols), np.asarray(prots), np.asarray(Y)
    dataset = get_dataset_split(dataset_name, mols, prots, Y)
    print(f'mols.shape: {mols.shape} prots.shape: {prots.shape} Y.shape: {Y.shape}')
    print('Loaded dataset')
    checkpoint_callback = checkpoint(checkpoint_path(model_name, model_version))
    if model_name == 'base_model':
        base_model.train(model_name, dataset, batch_size, epochs, [checkpoint_callback])
    elif model_name == 'dcnn_model':
        dcnn_model.train(model_name, dataset, batch_size, epochs, [checkpoint_callback])

def train_autoencoders(model_names, version_of_models, epochs, batch_size):
    molecules, proteins = AeDataSet().load_external_data()
    molecule_encoder, protein_encoder = None, None
    if model_names[0] == 'auen':
        checkpoint_callback = checkpoint(checkpoint_path(model_names[1], version_of_models))
        molecule_encoder = auen_model.train_molecule_model(model_names[1], molecules, batch_size, epochs, [checkpoint_callback])
        checkpoint_callback = checkpoint(checkpoint_path(model_names[2], version_of_models))
        protein_encoder = auen_model.train_protein_model(model_names[2], proteins, batch_size, epochs, [checkpoint_callback])
    elif model_names[0] == 'arnn':
        checkpoint_callback = checkpoint(checkpoint_path(model_names[1], version_of_models))
        molecule_encoder = arnn_model.train_molecule_model(model_names[1], molecules, batch_size, epochs, [checkpoint_callback])
        checkpoint_callback = checkpoint(checkpoint_path(model_names[2], version_of_models))
        protein_encoder = arnn_model.train_protein_model(model_names[2], proteins, batch_size, epochs, [checkpoint_callback])
    return molecule_encoder, protein_encoder

def load_autoencoders(model_names, version_of_models):
    molecule_encoder, protein_encoder = None, None
    if model_names[0] == 'auen':
        checkpoint = checkpoint_path(model_names[1], version_of_models)
        print(f'Loading autoencoder: {checkpoint}')
        molecule_encoder = auen_model.load_molecule_model(model_names[1], checkpoint)
        checkpoint = checkpoint_path(model_names[2], version_of_models)
        print(f'Loading autoencoder: {checkpoint}')
        protein_encoder = auen_model.load_protein_model(model_names[2], checkpoint)
    elif model_names[0] == 'arnn':
        checkpoint = checkpoint_path(model_names[1], version_of_models)
        print(f'Loading autoencoder: {checkpoint}')
        molecule_encoder = arnn_model.load_molecule_model(model_names[1], checkpoint)
        checkpoint = checkpoint_path(model_names[2], version_of_models)
        print(f'Loading autoencoder: {checkpoint}')
        protein_encoder = arnn_model.load_protein_model(model_names[2], checkpoint)
    return molecule_encoder, protein_encoder

def prepare_interaction_dataset(dataset_name, molecule_encoder, protein_encoder):
    print(f'Loading {dataset_name} dataset..')
    mols, prots, Y = DataSet(dataset_name).parse_data()
    print('Feature extraction...')
    mols = molecule_encoder.predict(mols)
    prots = protein_encoder.predict(prots)
    print(f'mols[0]: {mols[0]}')
    print(f'mols[1]: {mols[1]}')
    print(f'mols[136]: {mols[136]}')
    print(f'prots[0]: {prots[0]}')
    print(f'prots[1]: {prots[1]}') 
    print(f'prots[136]: {prots[136]}')
    pairs = []
    for i in range(len(mols)):
        pairs.append(np.concatenate((mols[i], prots[i])))
    pairs = np.asarray(pairs)
    return get_simple_dataset_split(dataset_name, pairs, Y)

def train_interaction_network(model_names, version_of_models, dataset, epochs, batch_size):
    checkpoint_callback = checkpoint(checkpoint_path(model_names[3], version_of_models))
    if model_names[0] == 'auen':
        auen_model.train_interaction_model(model_names[3], dataset, batch_size, epochs, [checkpoint_callback])
    elif model_names[0] == 'arnn':   
        arnn_model.train_interaction_model(model_names[3], dataset, batch_size, epochs, [checkpoint_callback])

def test_interaction_network(model_names, version_of_models, dataset):
    checkpoint = checkpoint_path(model_names[3], version_of_models)
    print(f'testing interaction model: {checkpoint}')
    if model_names[0] == 'auen':
        auen_model.test_interaction_model(model_names[3], dataset, checkpoint)
    elif model_names[0] == 'arnn':
        arnn_model.test_interaction_model(model_names[3], dataset, checkpoint)

def run_train_session(model_names, version_of_models, dataset_name, epochs, batch_size, load_weights):
    if not compatible_dataset(version_of_models, dataset_name):
        raise Error('Version of models not compatible with dataset.')
    molecule_encoder, protein_encoder = None, None
    if load_weights:
        print('Loading autoencoders...')
        molecule_encoder, protein_encoder = load_autoencoders(model_names, version_of_models)
        dataset = prepare_interaction_dataset(dataset_name, molecule_encoder, protein_encoder)
        print('Prepared dataset.')
        # test_interaction_network(model_names, version_of_models, dataset)
    else:
        print('Training autoencoders...')
        molecule_encoder, protein_encoder = None, None
        if dataset_name == 'davis':
            molecule_encoder, protein_encoder = load_autoencoders(model_names, version_of_models - 1)
        else:
            molecule_encoder, protein_encoder = train_autoencoders(model_names, version_of_models, epochs, batch_size)
        dataset = prepare_interaction_dataset(dataset_name, molecule_encoder, protein_encoder)
        print('Prepared dataset.') 
        train_interaction_network(model_names, version_of_models, dataset, epochs, batch_size)
