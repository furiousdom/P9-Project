from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import CalcPropertiesTable, DrugInteractionsTable, FeaturesTable, MainTable
from .serializers import MainTableSerializer

import numpy
import xmltodict
import deepchem as dc

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
            protein = self.findDrug(name = proteinName)
            queryset = self.findInteractionsById(protein.primary_id)[:10]
            # interactingFeatures = [FeaturesTable.objects.filter(drug_id = q.drug_id_2) for q in queryset]
            calcProps = [CalcPropertiesTable.objects.get(drug_id = q.drug_id_2) for q in queryset]
            turntoobject(calcProps)
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

def turntoobject(calcProps):
    my_list = []
    smiles_list = []
    for entry in calcProps:
        my_list.append(xmltodict.parse(entry.properties))
    for entry in my_list:
        val = ''
        for prop in entry['calculated-properties']['property']:
            if prop['kind'] == 'SMILES':
                val = prop['value']
        smiles_list.append(val)
    KNNmol2VecFeaturization(smiles_list)

# def helper(interactingFeatures):
#     features = []
#     for entry in interactingFeatures:
#         features.append(numpy.fromstring(entry.features, sep=' '))
#     KNNmol2VecFeaturization(features)

def helper():
    pass

class molecule: #This class is used to store objects in an array. Just holds the average distance for now.
    def __init__(self, average_distance, name):
        self.average_distance = average_distance
        self.name = name

def KNNmol2VecFeaturization(smiles):
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
