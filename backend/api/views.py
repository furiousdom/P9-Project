from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import CalcPropertiesTable, DrugInteractionsTable, MainTable
from .serializers import MainTableSerializer

import numpy
import xmltodict
import deepchem as dc
from scipy.spatial import distance

# Create your views here.

class DrugView(generics.ListAPIView):
    queryset = MainTable.objects.select_related('cprops', 'eprops')[:12]
    serializer_class = MainTableSerializer

class Drugs(APIView):

    # def post(self, request):
    #     if request.data:
    #         proteinName = request.data['proteinName']
    #         protein = self.findDrug(name = proteinName)
    #         queryset = self.findInteractionsById(protein.primary_id)
    #         interactingMolecules = [self.getSerializedDrug(q.drug_id_2).data for q in queryset]
    #         return Response({ "data": interactingMolecules })
    #     return Response({ "error": "No data in the request!" })

    def post(self, request):
        if request.data:
            proteinName = request.data['proteinName']
            noCandidates = request.data['noCandidates']
            protein = self.findDrug(name = proteinName)
            queryset = self.findInteractionsById(protein.primary_id)
            calcProps = [CalcPropertiesTable.objects.get(drug_id = q.drug_id_2) for q in queryset]
            allProps = CalcPropertiesTable.objects.all()
            candidateMolecules = knn(calcProps, allProps, noCandidates)
            candidates = [self.getSerializedDrug(c.id).data for c in candidateMolecules]
            return Response({ "data": candidates })
        return Response({ "error": "No data in the request!" })

    def getSerializedDrug(self, id=''):
        return MainTableSerializer(self.findDrug(id))

    def findDrug(self, id='', name=''):
        if id is not '':
            return MainTable.objects.get(primary_id = id)
        elif name is not '':
            return MainTable.objects.get(name = name)

    def findInteractionsById(self, id):
        return DrugInteractionsTable.objects.filter(drug_id_1 = id)

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

# def knn(calcProps, allProps, noCandidates):
#     interactingSmiles = xmlToSmiles(calcProps)
#     allSmiles = xmlToSmiles(allProps)

#     print("Starting the featurization...")
#     knn_helper(interactingSmiles, allSmiles, noCandidates)

def knn(calcProps, allProps, noCandidates):
    interactingMolecules = xmlToProps(calcProps)
    allMolecules = xmlToProps(allProps)

    print("Starting the featurization...")
    return knn_helper2(interactingMolecules, allMolecules, noCandidates)

class Molecule: #This class is used to store objects in an array. Just holds the average distance for now.
    def __init__(self, id=None, smiles=None, average_distance=None):
        self.id = id
        self.smiles = smiles
        self.average_distance = average_distance

def removeDuplicates(interactingMolecules, allMolecules):
    for mol in allMolecules:
        for interactingMol in interactingMolecules:
            if mol.smiles == interactingMol.smiles:
                allMolecules.remove(mol)
    return allMolecules

def knn_helper2(interactingMolecules, allMolecules, noCandidates):
    allMolecules = removeDuplicates(interactingMolecules, allMolecules)
    featurizer = dc.feat.Mol2VecFingerprint()
    interactingMolFeatures = featurizer([im.smiles for im in interactingMolecules])
    allMolFeatures = featurizer([m.smiles for m in allMolecules])
    i = 0
    k = 0
    combinedtempdistance = 0
    while i<len(allMolFeatures):
        while k<len(interactingMolFeatures):
            if (len(interactingMolFeatures[k])!=0 and len(allMolFeatures[i])!=0):
                combinedtempdistance = combinedtempdistance + euclideanDistance(interactingMolFeatures[k], allMolFeatures[i])
                k=k+1
            else:
                print(f'i:{i} K:{k} A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / len(interactingMolFeatures)
        allMolecules[i].average_distance = combinedtempdistance
        combinedtempdistance = 0
        i=i+1
        k=0
    allMolecules.sort(key=lambda x: x.average_distance)
    return allMolecules[:noCandidates]

def knn_helper(interactingSmiles, allSmiles, noCandidates):
    allSmiles = list(set(allSmiles) - set(interactingSmiles))
    featurizer = dc.feat.Mol2VecFingerprint()
    interactingMolFeatures = featurizer(interactingSmiles)
    allMolFeatures = featurizer(allSmiles)
    i = 0
    k = 0
    combinedtempdistance = 0
    moleculelist = []
    while i<len(allMolFeatures):
        while k<len(interactingMolFeatures):
            if (len(interactingMolFeatures[k])!=0 and len(allMolFeatures[i])!=0):
                combinedtempdistance = combinedtempdistance + euclideanDistance(interactingMolFeatures[k], allMolFeatures[i])
                k=k+1
            else:
                print(f'i:{i} K:{k} A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / len(interactingMolFeatures)
        molecule = Molecule(combinedtempdistance, allSmiles[i]) #Might want to use the actual name instead of just its number in the index.
        moleculelist.append(molecule) #Index of array is same as the molecule number.
        combinedtempdistance = 0
        i=i+1
        k=0
    moleculelist.sort(key=lambda x: x.average_distance)
    for mole in moleculelist:
        print(mole.average_distance)
    print(f'No Candidates: {noCandidates}')

def euclideanDistance(interactingMoleculeFeature, moleculeFeature):
    return numpy.linalg.norm(interactingMoleculeFeature-moleculeFeature)

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
