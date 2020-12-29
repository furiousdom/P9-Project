import xmltodict
import numpy as np
import deepchem as dc
from scipy.spatial import distance
from .models import FeaturesTable


class Molecule:
    def __init__(self, id=None, smiles=None, average_distance=None):
        self.id = id
        self.smiles = smiles
        self.average_distance = average_distance

def xmlToSmiles(xml):
    temp_list = []
    smiles_list = []
    for entry in xml:
        temp_list.append(xmltodict.parse(entry.properties))
    for entry in temp_list:
        val = ''
        if 'calculated-properties' in entry.keys():
            # if entry['calculated-properties'] is None:
                # print(entry['calculated-properties'])
            # else:
            try:
                if len(entry['calculated-properties'].keys()) >= 1:
                    # print(len(entry['calculated-properties'].keys()))
                    for prop in entry['calculated-properties']['property']:
                        if prop['kind'] == 'SMILES':
                            val = prop['value']
                            smiles_list.append(val)
            except Exception as error:
                print(error)
    return smiles_list


def xmlToProps(dbObjects):
    moleculeList = []
    for entry in dbObjects:
        props = xmltodict.parse(entry.properties)
        if 'calculated-properties' in props.keys():
            try:
                if len(props['calculated-properties'].keys()) >= 1:
                    # print(len(entry['calculated-properties'].keys()))
                    for prop in props['calculated-properties']['property']:
                        if prop['kind'] == 'SMILES':
                            val = prop['value']
                            moleculeList.append(Molecule(id=entry.drug_id, smiles=val))
            except Exception as error:
                print(error)
    return moleculeList

def knn(calcProps, allProps, noResults, logging):
    interactingMolecules = xmlToProps(calcProps)
    allMolecules = xmlToProps(allProps)

    print("Starting the featurization...")
    return knn_helper(interactingMolecules, allMolecules, noResults, logging)

def knn_helper(interactingMolecules, allMolecules, noResults, logging):
    interactingMolFeatures, allMolFeatures = featurize(interactingMolecules, allMolecules)
    for i in range(len(allMolFeatures)):
        combinedtempdistance = 0
        for interactingMol in interactingMolFeatures:
            if len(interactingMol) is not 0 and len(allMolFeatures[i]) is not 0:
                combinedtempdistance += euclideanDistance(interactingMol, allMolFeatures[i])
        combinedtempdistance /= len(interactingMolFeatures)
        allMolecules[i].average_distance = combinedtempdistance
    allMolecules.sort(key=lambda x: x.average_distance)
    if logging: logResults(allMolecules)
    return allMolecules[:noResults]

def featurize(interactingMolecules, allMolecules):
    featurizer = dc.feat.Mol2VecFingerprint()
    allMolecules = removeDuplicates(interactingMolecules, allMolecules)
    allMolecules = removeNonFeatures(allMolecules)
    interactingMolecules = removeNonFeatures(interactingMolecules)
    interactingMolFeatures = featurizer([im.smiles for im in interactingMolecules])
    allMolFeatures = featurizer([m.smiles for m in allMolecules])
    return interactingMolFeatures, allMolFeatures

def removeDuplicates(interactingMolecules, allMolecules):
    for interactingMol in interactingMolecules:
        for mol in allMolecules:
            if mol.smiles == interactingMol.smiles:
                allMolecules.remove(mol)
    return allMolecules

def removeNonFeatures(molecules):
    featureObjects = FeaturesTable.objects.all()
    featureIds = [featureObject.drug_id for featureObject in featureObjects]
    for mol in molecules:
        if mol.id not in featureIds:
            molecules.remove(mol)
    return molecules

def logResults(molecules):
    f = open("results.txt", "w")
    for mol in molecules:
        f.write(f'{mol.id}: {mol.average_distance}\n')
    f.close()

def euclideanDistance(interactingMoleculeFeature, moleculeFeature):
    return np.linalg.norm(interactingMoleculeFeature-moleculeFeature)

def manhattanDistance(interactingMoleculeFeature, moleculeFeature):
    sum = 0
    for x in range(0,300):
        sum = sum + abs(interactingMoleculeFeature[x]-moleculeFeature[x])
    return sum

def moleculeSimilarityFunction(interactingMoleculeFeature, moleculeFeature):
    sum = 0
    for x in range(0,300):
        sum = sum + ((interactingMoleculeFeature[x] * moleculeFeature[x]) / ((interactingMoleculeFeature[x] * moleculeFeature[x]) + (interactingMoleculeFeature[x] - moleculeFeature[x])**2))
    return sum

def minkowskiDistance(interactingMoleculeFeature, moleculeFeature):
    distance.minkowski(interactingMoleculeFeature, moleculeFeature, 2) # 2nd order. Needs to be changed to be set by the user.
