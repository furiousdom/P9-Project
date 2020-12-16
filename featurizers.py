import deepchem as dc
from rdkit import Chem# Might need to import this one in a different manor.
from printer import mols_to_pngs, display_images
from xdparser import parseSmiles
import numpy
from scipy.spatial import distance

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
    if featurization == 'convmol':
        convMolFeaturization(smiles)
    if featurization == 'circprint':
        circularFingerprintFeaturization(smiles)
    if featurization == 'weave':
        weaveFeaturization(smiles)
    if featurization == 'molgraph':
        molGraphConvFeaturization(smiles)
    if featurization == 'mol2vec':
        mol2VecFeaturization(smiles)
    if featurization == 'smile2image':
        smilesToImageFeaturization(smiles)
    if featurization == 'onehot':
        oneHotFeaturization(smiles)
    if featurization == 'coulombmatrix':
        coulombMatrixFeaturization(smiles)
    if featurization == 'KNNmol2vec':
        KNNmol2VecFeaturization(smiles)
    if featurization == 'manmol2vec':
        Manhattanmol2VecFeaturization(smiles)
    
    """
    switch(featurization){
        case 'rdkit': rdkitFeaturization(smiles);
            break;
        case 'convmol': convMolFeaturization(smiles);
            break;
        case 'circprint': convMolFeaturization(smiles);
            break;
        case 'weave': weaveFeaturization(smiles);
            break;
        case 'molgraph': molGraphConvFeaturization(smiles);
            break;
        case 'mol2vec': mol2VecFeaturization(smiles);
            break;
        case 'smile2image': smilesToImageFeaturization(smiles);
            break;
        case 'onehot': oneHotFeaturization(smiles);
            break;
        case 'coulombmatrix': coulombMatrixFeaturization(smiles);
            break;
    }
    """

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
    for feature in features:
        print(feature.atom_features)
        print(feature.canon_adj_list)

def circularFingerprintFeaturization(smiles):
    featurizer = dc.feat.CircularFingerprint()
    features = featurizer(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(features)

def weaveFeaturization(smiles):
    featurizer = dc.feat.WeaveFeaturizer()
    features = featurizer.featurize(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(features)#Cannot print the objects directly. Need to present them in some other format.

def molGraphConvFeaturization(smiles):
    featurizer = dc.feat.MolGraphConvFeaturizer()
    features = featurizer(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(features)#Cannot print the objects directly. Need to present them in some other format.

#This is the original mol2vec that works. Do not modify this further.
def mol2VecFeaturization(smiles):#Needs packages that are incompatible I think.
    featurizer = dc.feat.Mol2VecFingerprint()
    features = featurizer(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(f'Length of features: {len(features)}')
    i = 1
    k = 0
    while k<len(features):
        while i<len(features):
            if (len(features[k])!=0 and len(features[i])!=0):
                print(f'Distance from Molecule {k+1} to molecule {i+1}')
                dist = numpy.linalg.norm(features[k]-features[i])
                i=i+1
                print(dist)
            else:
                print('A molecule could not be featurized. Skipping ahead to next molecule.')
                i=i+1
        k=k+1
        i=k+1

#This is an attempt at creating a better mol2vec based KKN algorithm ##############################################################
class molecule: #This class is used to store objects in an array. Just holds the average distance for now.
    def __init__(self, average_distance, name):
        self.average_distance = average_distance
        self.name = name

def KNNmol2VecFeaturization(smiles):#Needs packages that are incompatible I think.
    featurizer = dc.feat.Mol2VecFingerprint()
    features = featurizer(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(f'Length of features: {len(features)}')
    i = 20 #Just taking the first 20 molecules as my initial set. Need to change this to be the initial molecule set instead. Create the set by cutting it from the complete list.
    k = 0
    
    combinedtempdistance = 0
    moleculelist = []
    while i<len(features):
        while k<20:       
            if (len(features[k])!=0 and len(features[i])!=0):
                combinedtempdistance = combinedtempdistance + numpy.linalg.norm(features[k]-features[i])
                k=k+1
            else:
                print('A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / 20 #20 should be replaced by len(list of initial molecules)
        Molecule = molecule(combinedtempdistance, i) #Might want to use the actual name instead of just its number in the index.
        moleculelist.append(Molecule) #Index of array is same as the molecule number.
        combinedtempdistance = 0
        i=i+1
        k=0
    moleculelist.sort(key=lambda x: x.average_distance)
    for mole in moleculelist:
        print(mole.average_distance)
    

###############################################################################################################################

def Manhattanmol2VecFeaturization(smiles):#Needs packages that are incompatible I think.
    featurizer = dc.feat.Mol2VecFingerprint()
    features = featurizer(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(f'Length of features: {len(features)}')
    i = 20 #Just taking the first 20 molecules as my initial set. Need to change this to be the initial molecule set instead. Create the set by cutting it from the complete list.
    k = 0
    combinedtempdistance = 0
    moleculelist = []
    while i<len(features):
        while k<20:       
            if (len(features[k])!=0 and len(features[i])!=0):            
                for x in range(0,300):
                    combinedtempdistance = combinedtempdistance + abs(features[k][x]-features[i][x])
                k=k+1
            else:
                #print('A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / 20 #20 should be replaced by len(list of initial molecules)
        Molecule = molecule(combinedtempdistance, i) #Might want to use the actual name instead of just its number in the index.
        moleculelist.append(Molecule) #Index of array is same as the molecule number.
        combinedtempdistance = 0
        i=i+1
        k=0
    moleculelist.sort(key=lambda x: x.average_distance)
    for mole in moleculelist:
        print(mole.average_distance)

###################################################################################################################################

def smilesToImageFeaturization(smiles):
    # featurizer = dc.feat.SmilesToImage()
    #features = featurizer.featurize(mols)
    #features = featurizer(smiles)
    #print(f'Number of featurized items: {len(features)} \nValues:')
    #print(features)#Images cannot just be printed, need some other method.
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    display_images(mols_to_pngs(mols))
    

def oneHotFeaturization(smiles):
    featurizer = dc.feat.OneHotFeaturizer()
    features = featurizer(smiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(features)#Prints an empty array.

def coulombMatrixFeaturization(smiles):
    generator = dc.utils.ConformerGenerator(max_conformers=5)#Dont know what default number is best. Can potentially make the user decide.
    mol = generator.generate_conformers(Chem.MolFromSmiles(smiles))#Chem doesn't contain MolFromSmiles.
    print("Number of available conformers for propane: ", len(mol.GetConformers()))
    featurizer = dc.feat.CoulombMatrix(max_atoms=20)#Also dont know the optimal seetings for max_atoms
    features = featurizer(mol)
    print(features)