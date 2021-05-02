import math
import numpy as np
import pandas as pd

DATASETS_TO_PREPROCESS = ['davis', 'davis2']

def load_binary_interactions(file_name, threshold, preprocess = False):
    scores_list = []
    scores_file = open(file_name, 'r')
    for line in scores_file:
        score = float(line)
        if preprocess:
            score = -1 * math.log10(score/pow(10, 9))
        if score >= threshold:
            scores_list.append([0, 1])
        else:
            scores_list.append([1, 0])
    scores_file.close()
    return scores_list

def load_binding_affinities(file_name, preprocess = False):
    scores_list = []
    scores_file = open(file_name, 'r')
    for line in scores_file:
        score = float(line)
        if preprocess:
            score = -1 * math.log10(score/pow(10, 9))
        scores_list.append(score)
    scores_file.close()
    return scores_list

def load_dataset(dataset_name, binding_affinity_threshold):
    molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
    proteins = pd.read_csv(f'./data/datasets/{dataset_name}/proteins.csv')
    molecules.drop(molecules.filter(regex="Unname"),axis=1, inplace=True)
    proteins.drop(proteins.filter(regex="Unname"),axis=1, inplace=True)
    X = np.array(pd.concat([molecules, proteins], axis=1))
    preprocess_binding_affinity = (dataset_name in DATASETS_TO_PREPROCESS) or False
    print(f'Preprocess_binding_affinity: {preprocess_binding_affinity}')
    Y = np.array(load_binding_affinities(
        f'./data/datasets/{dataset_name}/binding_affinities.txt',
        preprocess_binding_affinity
    ))
    return X, Y

def check_dataset(dataset_name):
    binding_affinity_threshold = 7.0
    preprocess_binding_affinity = True
    Y = np.array(load_binary_interactions(
        f'./data/datasets/{dataset_name}/binding_affinities.txt',
        binding_affinity_threshold,
        preprocess_binding_affinity
    ))
    counter0 = 0
    counter1 = 0
    for y in Y:
        print(y)
        x = y.tolist()
        if x == [1,0]:
            counter0 +=1
        elif x == [0,1]:
            counter1 += 1
    print(f'0: {counter0}, 1: {counter1}, all: {len(Y)}')
# check_dataset('davis2')