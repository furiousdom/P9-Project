from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import CalcPropertiesTable, DrugInteractionsTable, MainTable
from .serializers import MainTableSerializer

from .algorithm import knn

# Create your views here.

class DrugView(generics.ListAPIView):
    queryset = MainTable.objects.select_related('cprops', 'eprops')[:12]
    serializer_class = MainTableSerializer

class Drugs(APIView):

    def post(self, request):
        if request.data:
            proteinName = request.data['proteinName']
            noResults = request.data['noResults']
            logging = request.data['logging']
            protein = self.findDrug(name = proteinName)
            queryset = self.findInteractionsById(protein.primary_id)
            interProps = [self.findCalcPropsById(q.drug_id_2) for q in queryset]
            allProps = CalcPropertiesTable.objects.all()
            candidateMolecules = knn(interProps, allProps, noResults, logging)
            candidates = [self.getSerializedDrug(c.id).data for c in candidateMolecules]
            return Response(candidates, status.HTTP_200_OK)
        return Response({ "error": "No data in the request!" }, status.HTTP_400_BAD_REQUEST)

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
