import xmltodict
import numpy as np
from deepchem import feat
from scipy.spatial import distance
from .models import FeaturesTable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import random
from sklearn.decomposition import PCA


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
            try:
                if len(entry['calculated-properties'].keys()) >= 1:
                    for prop in entry['calculated-properties']['property']:
                        if prop['kind'] == 'SMILES':
                            val = prop['value']
                            smiles_list.append(val)
            except Exception as error:
                pass
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
                pass
    return moleculeList

def makeSmi(queryset):
    allSmiles = xmlToProps(queryset)
    f = open("mols.smi", "w")
    for mol in allSmiles:
        f.write(f'{mol.smiles}\n')
    f.close()
    return True

# def knn(calcProps, allProps, noResults, logging):
#     interactingMolecules = xmlToProps(calcProps)
#     allMolecules = xmlToProps(allProps)
#     interactingMolFeatures, allMolFeatures = featurize(interactingMolecules, allMolecules)
#     return cluster(interactingMolFeatures, allMolFeatures, allMolecules, noResults)

# def cluster(interactingMolFeatures, allMolFeatures, allMolecules, noResults):
#     X = np.array([[1, 2, 5], [1, 4, 6], [1, 0, 3], [10, 2, 7], [10, 4, 8], [10, 0, 9]])
#     kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
#     temp = kmeans.predict([[0, 0, 8], [12, 3, 10]])
#     print(f'Predict: {temp}')
#     temp2 = kmeans.cluster_centers_
#     print(f'Cluster center: {temp2}')

#     # for i in range(len(allMolecules)):
#     #     allMolecules[i].average_distance = temp[i]
#     return allMolecules[:noResults]

# def featurize(interactingMolecules, allMolecules):
#     featurizer = feat.Mol2VecFingerprint()
#     allMolecules = removeDuplicates(interactingMolecules, allMolecules)
#     allMolecules = removeNonFeatures(allMolecules)
#     interactingMolecules = removeNonFeatures(interactingMolecules)
#     interactingMolFeatures = featurizer([im.smiles for im in interactingMolecules])
#     allMolFeatures = featurizer([m.smiles for m in allMolecules])
#     return interactingMolFeatures, allMolFeatures

def knn(calcProps, allProps, noResults, logging):
    interactingMolecules = xmlToProps(calcProps)
    allMolecules = xmlToProps(allProps)

    print("Starting the featurization...")
    return knn_helper(interactingMolecules, allMolecules, noResults, logging)

def knn_helper(interactingMolecules, allMolecules, noResults, logging):
    interactingMolFeatures, allMolFeatures, testSet = featurize(interactingMolecules, allMolecules)
    for i in range(len(allMolFeatures)):
        combinedtempdistance = 0
        for interactingMol in interactingMolFeatures:
            if len(interactingMol) is not 0 and len(allMolFeatures[i]) is not 0:
                combinedtempdistance += euclideanDistance(interactingMol, allMolFeatures[i])
        combinedtempdistance /= len(interactingMolFeatures)
        allMolecules[i].average_distance = combinedtempdistance
    allMolecules.sort(key=lambda x: x.average_distance)
    testResults(allMolecules, testSet, noResults)
    if logging: logResults(allMolecules, testSet)
    return allMolecules[:noResults]

# def knn(calcProps, allProps, noResults, logging):
#     interactingMolecules = xmlToProps(calcProps)
#     allMolecules = xmlToProps(allProps)

#     interactingMolFeatures, allMolFeatures, testSet = featurize(interactingMolecules, allMolecules)
#     alg = KNeighborsClassifier()
#     alg.fit(allMolFeatures, interactingMolFeatures)
#     alg.kneighbors(X=interactingMolecules, n_neighbors=noResults)

def transformEmbeddings(embeddings, length):
    return [embedding[:length] for embedding in embeddings]

#Used for testing reduced dimensionality of mol2vec
def featurize(interactingMolecules, allMolecules):
    # embeddingLength = 10
    featurizer = feat.CircularFingerprint(size=20)
    interactingMolecules, testSet = makeTestSet(interactingMolecules)
    allMolecules = removeDuplicates(interactingMolecules, allMolecules)
    allMolecules = removeNonFeatures(allMolecules)
    interactingMolecules = removeNonFeatures(interactingMolecules)
    interactingMolFeatures = featurizer([im.smiles for im in interactingMolecules])
    allMolFeatures = featurizer([m.smiles for m in allMolecules])
    # interactingMolFeatures = transformEmbeddings(interactingMolFeatures, embeddingLength)
    # allMolFeatures = transformEmbeddings(allMolFeatures, embeddingLength)
    return interactingMolFeatures, allMolFeatures, testSet

#Tested the entire dataset and 3 molecules were missing. When we test normally, we only find 3 molecules. Coincidence? I think not!
#There also happened to be 3 molecules that we could not featurize in the dataset. Are we only finding the empty arrays? Since those have distance=0?
def testResults(candidates, testSet, noResults):
    if noResults == len(testSet):
        print('K is good')
        print(f'k is {noResults} while length of testset is {len(testSet)}')
    else:
        print(f'k is {noResults} while length of testset is {len(testSet)}')
    length = len(candidates) / 2 #Needed for testing entire set.
    results = candidates[:len(testSet)] #:noResults
    resultIds = [result.id for result in results] #Changed to candidates from results to check entire dataset.
    counter = 0
    for molecule in testSet:
        if molecule.id in resultIds:
            counter += 1
    accuracy = counter / len(testSet)
    print(f'Number of matches needed is {len(testSet)}')
    print(f'Number of actual matches is {counter}')
    print(f'Accuracy is {accuracy * 100}%')

def makeTestSet(molecules):
    middleIndex = len(molecules) // 2
    # np.random.shuffle(molecules)
    firstHalf = molecules[:middleIndex]
    secondHalf = molecules[middleIndex:]
    return firstHalf, secondHalf

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

def logResults(molecules, testSet):
    f = open("results.txt", "w")
    for mol in molecules:
        f.write(f'{mol.id}: {mol.average_distance}\n')
    f.close()
    f = open("resultids.txt", "w")
    for mol in molecules:
        f.write(f'{mol.id}\n')
    f.close()
    f = open("testset.txt", "w")
    for mol in testSet:
        f.write(f'{mol.id}\n')
    f.close()

def euclideanDistance(interactingMoleculeFeature, moleculeFeature):
    # return np.linalg.norm(interactingMoleculeFeature-moleculeFeature)
    return distance.hamming(interactingMoleculeFeature, moleculeFeature)

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
    return distance.minkowski(interactingMoleculeFeature, moleculeFeature, 1.5) # 2nd order. Needs to be changed to be set by the user.
