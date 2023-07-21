from rest_framework import serializers
from .models import Datasets, Datasets_English_Translations


class DatasetsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Datasets
        fields = "__all__"
        read_only_fields = ("created_at",)


class Datasets_English_TranslationsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Datasets_English_Translations
        fields = "__all__"
        read_only_fields = ("created_at",)
