import math
import numpy as np
import pandas as pd

def load_dataset(dataset_name):
    molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules_first.csv')
    molecules_rest = pd.read_csv(f'./data/datasets/{dataset_name}/molecules_rest.csv')
    proteins = pd.read_csv(f'./data/datasets/{dataset_name}/proteins_first.csv')
    proteins_rest = pd.read_csv(f'./data/datasets/{dataset_name}/proteins_rest.csv')
    molecules_rest.drop(molecules_rest.filter(regex="Unname"),axis=1, inplace=True)
    molecules = pd.concat([molecules, molecules_rest])
    proteins = pd.concat([proteins, proteins_rest])
    molecules.to_csv(f'./data/datasets/{dataset_name}/molecules.csv')
    proteins.to_csv(f'./data/datasets/{dataset_name}/proteins.csv')

load_dataset('davis2')