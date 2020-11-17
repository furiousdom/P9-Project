import deepchem as dc

from xdparser import parseSmiles



def smilesList(smiles_dic_values):
    """
    This is a function that converts dictionary values to list of smiles.
    Use it if any problems arise while using the featurize_molecules function.
    """
    smiles = []
    for item in smiles_dic_values:
        smiles.append(item)

def featurize_molecules(featurization=None, dataset=None):
    smiles = parseSmiles(dataset).values()

    print("Starting featurization...")
    if featurization == 'rdkit':
        rdkitFeaturization(smiles)
    else:
        convMolFeaturization(smiles)

def rdkitFeaturization(smiles):
    rdkit_featurizer = dc.feat.RDKitDescriptors()
    mols = rdkit_featurizer(smiles)
    counter = 1
    for features in mols[:10]:
        for feature, descriptor in zip(features[:10], rdkit_featurizer.descriptors):
            print(f'{counter}: {descriptor}, {feature}')
        print()
        counter += 1

def convMolFeaturization(smiles):
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    features = featurizer(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(features)
