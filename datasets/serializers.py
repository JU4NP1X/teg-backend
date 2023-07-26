from rest_framework import serializers
from .models import Datasets, Datasets_English_Translations


class Datasets_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Datasets
        fields = "__all__"
        read_only_fields = ("created_at",)


class Datasets_English_Translations_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Datasets_English_Translations
        fields = "__all__"
        read_only_fields = ("created_at",)
