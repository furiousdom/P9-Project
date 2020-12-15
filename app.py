from featurizers import featurize_molecules

print('To start featurization please,')

prompt1 = 'q'
while prompt1 != '':
    prompt1 = input('Specify the featurizer (rdkit, convmol, circprint, weave, molgraph, mol2vec, smile2image, onehot, coulombmatrix, KNNmol2vec): ')
    if prompt1 == 'rdkit' or prompt1 == 'convmol' or prompt1 == 'circprint' or prompt1 == 'weave' or prompt1 == 'molgraph' or prompt1 == 'mol2vec' or prompt1 == 'smile2image' or prompt1 == 'onehot' or prompt1 == 'coulombmatrix' or prompt1 == 'KNNmol2vec':
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

featurize_molecules(prompt1, prompt2)
