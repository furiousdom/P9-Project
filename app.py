from featurizers import featurize_molecules
import time

print('To start featurization please,')

prompt1 = 'q'
while prompt1 != '':
    prompt1 = input('Specify the featurizer (rdkit, convmol, circprint, weave, molgraph, mol2vec, smile2image, onehot, coulombmatrix, KNNmol2vec, manmol2vec, simmol2vec, minmol2vec): ')
    if prompt1 == 'rdkit' or prompt1 == 'convmol' or prompt1 == 'circprint' or prompt1 == 'weave' or prompt1 == 'molgraph' or prompt1 == 'mol2vec' or prompt1 == 'smile2image' or prompt1 == 'onehot' or prompt1 == 'coulombmatrix' or prompt1 == 'KNNmol2vec' or prompt1 == 'manmol2vec' or prompt1 == 'simmol2vec' or prompt1 == 'minmol2vec':
        break
    prompt1 = ''
    print('Wrong input.')

prompt2 = 'q'
while prompt2 != '':
    prompt2 = input('Specify the dataset (sd [sample dataset], fd [full dataset]): ')
    if prompt2 == 'sd' or prompt2 == 'fd':
        break
    prompt2 = ''
    print('Wrong input.')

tic = time.perf_counter()
featurize_molecules(prompt1, prompt2)
toc = time.perf_counter()
print(f"Execution time: {toc - tic} seconds")

# rdkit: 215.2466 seconds   For the first 10 values.  Only featurization
# convmol: 56.6997295 seconds..  Only featurization
# circprint: 61.857619099999994 seconds.  Only featurization
# weave: 6077.2592242 seconds.  Only featurization
# molgraph: 196.1333791 seconds.  Only featurization

# KNNmol2vec: 73.5418805 seconds. Target molecule set size = 20. Performs featurization, finds distance, sorts a list.
# manmol2vec: 106.32661039999999 seconds. Target molecule set size = 20. Performs featurization, finds distance, sorts a list.
# simmol2vec: 568.5501368 seconds. Target molecule set size = 20. Performs featurization, finds distance, sorts a list.
# minmol2vec: 63.571768500000005 seconds. Target molecule set size = 20. Performs featurization, finds distance, sorts a list.