from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import MainTable
from .serializers import MainTableSerializer

# Create your views here.

class DrugView(generics.ListAPIView):
    queryset = MainTable.objects.select_related('props')[:10]
    serializer_class = MainTableSerializer
