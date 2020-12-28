from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import CalcPropertiesTable, DrugInteractionsTable, FeaturesTable, MainTable
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
            # interactingFeatures = [FeaturesTable.objects.filter(drug_id = q.drug_id_2) for q in queryset]
            calcProps = [CalcPropertiesTable.objects.get(drug_id = q.drug_id_2) for q in queryset]
            allProps = CalcPropertiesTable.objects.all()
            turntoobject(calcProps, allProps, noCandidates)
            return Response({ "message": "FU" })
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

def turntoobject(calcProps, allProps, noCandidates):
    my_list = []
    smiles_list = []
    for entry in calcProps:
        my_list.append(xmltodict.parse(entry.properties))
    for entry in my_list:
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

    all_my_list = []
    all_smiles_list = []
    for entry in allProps:
        all_my_list.append(xmltodict.parse(entry.properties))
    for entry in all_my_list:
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
                            all_smiles_list.append(val)
            except Exception as error:
                print(error)

    
    print("Starting the featurization...")
    # Make it possible for the user to select what distance algorithm they want to use.
    # KNNmol2VecFeaturization(smiles_list, all_smiles_list, noCandidates)
    # Manhattanmol2VecFeaturization(smiles_list, all_smiles_list, noCandidates)
    # Simmol2VecFeaturization(smiles_list, all_smiles_list, noCandidates)
    Minkowskimol2VecFeaturization(smiles_list, all_smiles_list, noCandidates)

def helper():
    pass

class molecule: #This class is used to store objects in an array. Just holds the average distance for now.
    def __init__(self, average_distance, name):
        self.average_distance = average_distance
        self.name = name

def KNNmol2VecFeaturization(smiles, allsmiles, noCandidates):
    featurizer = dc.feat.Mol2VecFingerprint()
    allsmiles = list(set(allsmiles) - set(smiles))
    features = featurizer(smiles)
    all_features = featurizer(allsmiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(f'Length of features: {len(features)}')   
    i = 0 
    k = 0
    combinedtempdistance = 0
    moleculelist = []
    while i<len(all_features):
        while k<len(features):
            if (len(features[k])!=0 and len(all_features[i])!=0):
                combinedtempdistance = combinedtempdistance + numpy.linalg.norm(features[k]-all_features[i])
                k=k+1
            else:
                print(f'i:{i} K:{k} A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / len(features)
        Molecule = molecule(combinedtempdistance, i) #Might want to use the actual name instead of just its number in the index.
        moleculelist.append(Molecule) #Index of array is same as the molecule number.
        combinedtempdistance = 0
        i=i+1
        k=0
    moleculelist.sort(key=lambda x: x.average_distance)
    for mole in moleculelist:
        print(mole.average_distance)

def Manhattanmol2VecFeaturization(smiles, allsmiles, noCandidates):#Needs packages that are incompatible I think.
    featurizer = dc.feat.Mol2VecFingerprint()
    allsmiles = list(set(allsmiles) - set(smiles))
    features = featurizer(smiles)
    all_features = featurizer(allsmiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(f'Length of features: {len(features)}')
    i = 0
    k = 0
    combinedtempdistance = 0
    moleculelist = []
    while i<len(all_features):
        while k<len(features):       
            if (len(features[k])!=0 and len(all_features[i])!=0):            
                for x in range(0,300):
                    combinedtempdistance = combinedtempdistance + abs(features[k][x]-all_features[i][x])
                k=k+1
            else:
                print(f'i:{i} K:{k} A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / len(features)
        Molecule = molecule(combinedtempdistance, i) #Might want to use the actual name instead of just its number in the index.
        moleculelist.append(Molecule) #Index of array is same as the molecule number.
        combinedtempdistance = 0
        i=i+1
        k=0
    moleculelist.sort(key=lambda x: x.average_distance)
    for mole in moleculelist:
        print(mole.average_distance)

def Simmol2VecFeaturization(smiles, allsmiles, noCandidates):#Needs packages that are incompatible I think.
    featurizer = dc.feat.Mol2VecFingerprint()
    allsmiles = list(set(allsmiles) - set(smiles))
    features = featurizer(smiles)
    all_features = featurizer(allsmiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(f'Length of features: {len(features)}')
    i = 0
    k = 0
    combinedtempdistance = 0
    moleculelist = []
    while i<len(all_features):
        while k<len(features):       
            if (len(features[k])!=0 and len(all_features[i])!=0):            
                for x in range(0,300):
                    combinedtempdistance = combinedtempdistance + ((features[k][x] * all_features[i][x]) / ((features[k][x] * all_features[i][x]) + (features[k][x] - all_features[i][x])**2))
                k=k+1
            else:
                print(f'i:{i} K:{k} A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / len(features)
        Molecule = molecule(combinedtempdistance, i) #Might want to use the actual name instead of just its number in the index.
        moleculelist.append(Molecule) #Index of array is same as the molecule number.
        combinedtempdistance = 0
        i=i+1
        k=0
    moleculelist.sort(key=lambda x: x.average_distance, reverse=True)#Higher is better for this method. 
    for mole in moleculelist:
        print(mole.average_distance)

def Minkowskimol2VecFeaturization(smiles, allsmiles, noCandidates):#Needs packages that are incompatible I think.
    featurizer = dc.feat.Mol2VecFingerprint()
    allsmiles = list(set(allsmiles) - set(smiles))
    features = featurizer(smiles)
    all_features = featurizer(allsmiles)
    print(f'Number of featurized items: {len(features)} \nValues:')
    print(f'Length of features: {len(features)}')
    i = 0
    k = 0
    combinedtempdistance = 0
    moleculelist = []
    while i<len(all_features):
        while k<len(features):       
            if (len(features[k])!=0 and len(all_features[i])!=0):            
                combinedtempdistance = combinedtempdistance + distance.minkowski(features[k], all_features[i],2) # 2nd order. Needs to be changed to be set by the user.
                k=k+1
            else:
                print(f'i:{i} K:{k} A molecule could not be featurized. Skipping ahead to next molecule.')
                k=k+1
        combinedtempdistance = combinedtempdistance / len(features)
        Molecule = molecule(combinedtempdistance, i) #Might want to use the actual name instead of just its number in the index.
        moleculelist.append(Molecule) #Index of array is same as the molecule number.
        combinedtempdistance = 0
        i=i+1
        k=0
    moleculelist.sort(key=lambda x: x.average_distance) 
    for mole in moleculelist:
        print(mole.average_distance)
