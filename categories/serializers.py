from rest_framework import serializers
from .models import Categories

class CategoriesSerializer(serializers.ModelSerializer):
  class Meta:
    model = Categories
    fields = '__all__'
    read_only_fields = ('created_at',)