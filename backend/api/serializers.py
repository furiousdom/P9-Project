from rest_framework import serializers
from .models import MainTable

class MainTableSerializer(serializers.ModelSerializer):
    class Meta:
        model = MainTable
        fields = ('primary_id', 'name', 'description')
