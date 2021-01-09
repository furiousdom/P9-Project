from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import CalcPropertiesTable, DrugInteractionsTable, FeaturesTable, MainTable
from .serializers import MainTableSerializer

from .algorithm import knn, makeGraphs

# Create your views here.

class DrugView(generics.ListAPIView):
    featureObjects = FeaturesTable.objects.all()[:12]
    featureIds = [featureObject.drug_id for featureObject in featureObjects]
    queryset = MainTable.objects.filter(primary_id__in=featureIds)
    serializer_class = MainTableSerializer

class Drugs(APIView):

    def post(self, request):
        try:
            if request.data:
                proteinName = request.data['proteinName']
                noResults = request.data['noResults']
                logging = request.data['logging']
                candidates = self.getCandidates(proteinName, noResults, logging)
                return Response(candidates, status.HTTP_200_OK)
            else:
                error = { "error": "No data in the request!" }
                return Response(error, status.HTTP_400_BAD_REQUEST)
        except Exception as identifier:
            return Response(identifier, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def getCandidates(self, proteinName, noResults, logging):
        protein = self.findDrug(name = proteinName)
        interactions = self.findInteractionsById(protein.primary_id)
        interProps = [self.findCalcPropsById(i.drug_id_2) for i in interactions]
        allProps = CalcPropertiesTable.objects.all()
        candidateMolecules = knn(interProps, allProps, noResults, logging)
        return [self.getSerializedDrug(c.id).data for c in candidateMolecules]

    def getSerializedDrug(self, id=''):
        return MainTableSerializer(self.findDrug(id))

    def findDrug(self, id='', name=''):
        if id is not '':
            return MainTable.objects.get(primary_id = id)
        elif name is not '':
            return MainTable.objects.get(name = name)

    def findInteractionsById(self, id):
        return DrugInteractionsTable.objects.filter(drug_id_1 = id)

    def findCalcPropsById(self, id):
        return CalcPropertiesTable.objects.get(drug_id = id)

class MakeMolGraphs(APIView):

    def get(self, request):
        try:
            featuresTableQueryset = FeaturesTable.objects.all()
            ids = [item.drug_id for item in featuresTableQueryset]
            calcPropTableQueryset = CalcPropertiesTable.objects.filter(drug_id__in = ids)
            completionCode = makeGraphs(calcPropTableQueryset)
            if completionCode == 201: return Response({}, status.HTTP_201_CREATED)
            elif completionCode == 208:
                detailMsg = { "message": "This request was already fulfilled." }
                return Response(detailMsg, status.HTTP_208_ALREADY_REPORTED)
        except Exception as identifier:
            return Response(identifier, status.HTTP_500_INTERNAL_SERVER_ERROR)
