from rest_framework import serializers
from .models import Datasets

class DatasetsSerializer(serializers.ModelSerializer):
  class Meta:
    model = Datasets
    fields = '__all__'
    read_only_fields = ('created_at',)