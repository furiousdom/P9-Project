import xmltodict
import numpy as np
from os import path, listdir
from django.conf import settings
from deepchem import feat
from scipy.spatial import distance
from .models import FeaturesTable
from .graphPrinter import printMolGraphs


class Molecule:
    def __init__(self, id=None, smiles=None, average_distance=None):
        self.id = id
        self.smiles = smiles
        self.average_distance = average_distance

def knn(calcProps, allProps, noResults, logging):
    allMolecules = querysetToMolecules(allProps)
    interactingMolecules = querysetToMolecules(calcProps)
    allMolecules = knn_helper(interactingMolecules, allMolecules)
    if logging: logResults(allMolecules)
    return allMolecules[:noResults]

def knn_helper(interactingMolecules, allMolecules):
    interactingMolFeatures, allMolFeatures = featurize(interactingMolecules, allMolecules)
    for i in range(len(allMolFeatures)):
        combinedtempdistance = 0
        for interactingMol in interactingMolFeatures:
            if len(interactingMol) is not 0 and len(allMolFeatures[i]) is not 0:
                combinedtempdistance += euclideanDistance(interactingMol, allMolFeatures[i])
        combinedtempdistance /= len(interactingMolFeatures)
        allMolecules[i].average_distance = combinedtempdistance
    allMolecules.sort(key=lambda x: x.average_distance)
    return allMolecules

def featurize(interactingMolecules, allMolecules):
    featurizer = feat.Mol2VecFingerprint()
    allMolecules = removeDuplicates(interactingMolecules, allMolecules)
    allMolecules = removeNonFeatures(allMolecules)
    allMolFeatures = featurizer([m.smiles for m in allMolecules])
    interactingMolecules = removeNonFeatures(interactingMolecules)
    interactingMolFeatures = featurizer([im.smiles for im in interactingMolecules])
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

def euclideanDistance(interactingMoleculeFeature, moleculeFeature):
    return np.linalg.norm(interactingMoleculeFeature-moleculeFeature)

def manhattanDistance(interactingMoleculeFeature, moleculeFeature):
    sum = 0
    for x in range(0,300):
        sum += abs(interactingMoleculeFeature[x]-moleculeFeature[x])
    return sum

def moleculeSimilarityFunction(interactingMoleculeFeature, moleculeFeature):
    sum = 0
    for x in range(0,300):
        sum += ((interactingMoleculeFeature[x] * moleculeFeature[x])
        / ((interactingMoleculeFeature[x] * moleculeFeature[x])
        + (interactingMoleculeFeature[x] - moleculeFeature[x])**2))
    return sum

def minkowskiDistance(interactingMoleculeFeature, moleculeFeature):
    return distance.minkowski(interactingMoleculeFeature, moleculeFeature, 2)

def querysetToMolecules(queryset):
    notNone = lambda item: item is not None
    molecules = [xmlToMol(entry) for entry in queryset]
    molecules = list(filter(notNone, molecules))
    print(molecules)
    return molecules

def xmlToMol(entry):
    props = xmltodict.parse(entry.properties)
    try:
        for prop in props['calculated-properties']['property']:
            if prop['kind'] == 'SMILES':
                return Molecule(id = entry.drug_id, smiles = prop['value'])
    except:
        return None

def logResults(molecules):
    f = open("results.txt", "w")
    for mol in molecules:
        f.write(f'{mol.id}: {mol.average_distance}\n')
    f.close()

def makeGraphs(queryset):
    graphsDirPath = path.join(settings.BASE_DIR, 'static\graphs\\')
    if len(listdir(graphsDirPath)) > 10000:
        return 208
    else:
        molecules = querysetToMolecules(queryset)
        return printMolGraphs(molecules)


# # SKLEARN implementation of KNN algorithm

# def knn(calcProps, allProps, noResults, logging):
#     interactingMolecules = querysetToMolecules(calcProps)
#     allMolecules = querysetToMolecules(allProps)

#     interactingMolFeatures, allMolFeatures, testSet = featurize(interactingMolecules, allMolecules)
#     alg = KNeighborsClassifier()
#     alg.fit(allMolFeatures, interactingMolFeatures)
#     alg.kneighbors(X=interactingMolecules, n_neighbors=noResults)
