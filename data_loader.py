import math
import numpy as np
import pandas as pd

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

def load_dataset(dataset_name, binding_affinity_threshold):
    molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
    proteins = pd.read_csv(f'./data/datasets/{dataset_name}/proteins.csv')
    X = np.array(pd.concat([molecules, proteins], axis=1))
    preprocess_binding_affinity = dataset_name is 'davis' or False
    Y = np.array(load_binary_interactions(
        f'./data/datasets/{dataset_name}/binding_affinities.txt',
        binding_affinity_threshold,
        preprocess_binding_affinity
    ))
    return X, Y
