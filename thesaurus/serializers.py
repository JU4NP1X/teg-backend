from rest_framework import serializers
from .models import Thesaurus

class ThesaurusSerializer(serializers.ModelSerializer):
  class Meta:
    model = Thesaurus
    fields = '__all__'
    read_only_fields = ('created_at',)