from django.shortcuts import render
from rest_framework import generics
from .models import MainTable
from .serializers import MainTableSerializer

# Create your views here.

class DrugView(generics.ListAPIView):
    queryset = MainTable.objects.all()
    serializer_class = MainTableSerializer
