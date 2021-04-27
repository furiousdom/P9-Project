import pandas as pd

molecules = pd.read_csv('./data/datasets/aau40000/molecules.csv')
proteins = pd.read_csv('./data/datasets/aau40000/proteins.csv')

print(molecules.head)
print(proteins.head)