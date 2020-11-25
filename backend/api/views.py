from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from .serializers import DrugXmlSerializer
from .models import DrugXml

# Create your views here.
class DrugXmlView(generics.ListAPIView):
    queryset = DrugXml.objects.all()
    serializer_class = DrugXmlSerializer
