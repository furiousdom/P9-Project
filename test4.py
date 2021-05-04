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

# load_dataset('davis2')

f = open(f'./data/kiba2-results.txt', 'r')
scores = []
counter = 0
for line in f.readlines():
    splittet_line = line.split(' ')
    ground_truth = 1 if float(splittet_line[0]) >= 12.1 else 0
    prediction = 1 if float(splittet_line[1]) >= 12.1 else 0
    if ground_truth == prediction:
        counter += 1
    scores.append([ground_truth, prediction])

f.close()
f = open(f'./data/kiba2-results-binarised.txt', 'w')
for lines in scores:
    f.write(f'{lines[0]} {lines[1]}\n')

f.close()

print(f'Accuraccy = {round(counter * 100/len(scores), 3)}')