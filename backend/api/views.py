from django.shortcuts import render
from rest_framework import generics
from .models import Drug
from .serializers import DrugSerializer

# Create your views here.

class DrugView(generics.ListAPIView):
    queryset = Drug.objects.all()
    serializer_class = DrugSerializer
