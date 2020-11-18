from rest_framework import serializers
from .models import DrugXml

class DrugXmlSerializer(serializers.ModelSerializer):
    class Meta:
        model = DrugXml
        fields = ('id', 'name', 'content')
