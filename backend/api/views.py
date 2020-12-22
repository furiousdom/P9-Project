from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import DrugInteractionsTable, MainTable
from .serializers import MainTableSerializer

# Create your views here.

class DrugView(generics.ListAPIView):
    queryset = MainTable.objects.select_related('cprops', 'eprops')[:12]
    serializer_class = MainTableSerializer

class Drugs(APIView):

    def post(self, request):
        if request.data:
            proteinName = request.data['proteinName']
            protein = self.findDrug(name = proteinName)
            queryset = self.findInteractionsById(protein.primary_id)
            interactingMolecules = [self.getSerializedDrug(q.drug_id_2).data for q in queryset]
            return Response({ "data": interactingMolecules })
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
