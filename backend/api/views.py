from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import MainTable
from .models import CalcPropertiesTable as CalcProps
from .models import DrugInteractionsTable as DrugInteractions
from .models import FeaturesTable as Features
from .serializers import MainTableSerializer

from os import path, listdir
from django.conf import settings
from .algorithm import knn, makeGraphs

# Create your views here.

class DrugView(generics.ListAPIView):
    featureIds = Features.objects.values_list('drug_id', flat=True)[:12]
    queryset = MainTable.objects.filter(primary_id__in=featureIds)
    serializer_class = MainTableSerializer

class Drugs(APIView):

    def post(self, request):
        try:
            if len(request.data.values()) == 3:
                proteinName, noResults, logging = request.data.values()
                candidates = self.getCandidates(proteinName, noResults, logging)
                return Response(candidates, status.HTTP_200_OK)
            else:
                error = { "error": "Parameter missing!" }
                return Response(error, status.HTTP_400_BAD_REQUEST)
        except Exception as identifier:
            return Response(identifier, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def getCandidates(self, proteinName, noResults, logging):
        protein = findDrug(name = proteinName)
        interactions = findInteractionsById(protein.primary_id)
        interProps = [findCalcPropsById(i.drug_id_2) for i in interactions]
        allProps = CalcProps.objects.all()
        candidateMolecules = knn(interProps, allProps, noResults, logging)
        return [getSerializedDrug(c.id).data for c in candidateMolecules]

class MakeMolGraphs(APIView):

    def get(self, request):
        try:
            graphsDirPath = path.join(settings.BASE_DIR, 'static\graphs\\')
            if len(listdir(graphsDirPath)) > 10000:
                detailMsg = { "message": "This request was already fulfilled." }
                return Response(detailMsg, status.HTTP_208_ALREADY_REPORTED)
            featureSet = Features.objects.all()
            ids = [item.drug_id for item in featureSet]
            calcPropSet = getCalcPropsByIds(ids)
            completionCode = makeGraphs(calcPropSet)
            if completionCode == 201: return Response({}, status.HTTP_201_CREATED)
        except Exception as identifier:
            return Response(identifier, status.HTTP_500_INTERNAL_SERVER_ERROR)


def getSerializedDrug(id=''):
    return MainTableSerializer(findDrug(id))

def findDrug(id='', name=''):
    if id is not '':
        return MainTable.objects.get(primary_id = id)
    elif name is not '':
        return MainTable.objects.get(name = name)

def findInteractionsById(id):
    return DrugInteractions.objects.filter(drug_id_1 = id)

def findCalcPropsById(id):
    return CalcProps.objects.get(drug_id = id)

def getCalcPropsByIds(ids):
    return CalcProps.objects.filter(drug_id__in = ids)
