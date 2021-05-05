import math
import numpy as np
import pandas as pd

DATASETS_TO_PREPROCESS = ['davis', 'davis2']

def converter(y, convert):
    return -1 * math.log10(y/pow(10, 9)) if convert else y

def binarize_score(y, threshold):
    return [0, 1] if y >= threshold else [1, 0]

def process_score(y, threshold=None, convert=False):
    y = converter(y, convert)
    return binarize_score(y, threshold) if threshold else y

def load_interactions(file_name, threshold=None, convert=False):
    Y = []
    with open(file_name, 'r') as scores_file:
        for line in scores_file:
            Y.append(process_score(float(line), threshold, convert))
    return Y

def load_Y(dataset_name, binding_affinity_threshold=None):
    preprocess_binding_affinity = dataset_name in DATASETS_TO_PREPROCESS
    return np.array(load_interactions(
        f'./data/datasets/{dataset_name}/binding_affinities.txt',
        binding_affinity_threshold,
        preprocess_binding_affinity
    ))

def load_X(dataset_name):
    molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
    proteins = pd.read_csv(f'./data/datasets/{dataset_name}/proteins.csv')
    molecules.drop(molecules.filter(regex="Unname"),axis=1, inplace=True)
    proteins.drop(proteins.filter(regex="Unname"),axis=1, inplace=True)
    return np.array(pd.concat([molecules, proteins], axis=1))

def load_dataset(dataset_name, binding_affinity_threshold=None):
    X = load_X(dataset_name)
    Y = load_Y(dataset_name, binding_affinity_threshold)
    return X, Y
