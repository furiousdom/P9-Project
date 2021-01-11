from rest_framework import serializers
from .models import MainTable

class MainTableSerializer(serializers.ModelSerializer):
    cprops = serializers.StringRelatedField()

    class Meta:
        model = MainTable
        fields = ('primary_id', 'name', 'cprops')
