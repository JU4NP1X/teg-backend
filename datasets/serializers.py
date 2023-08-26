from rest_framework import serializers
from categories.models import Authorities
from .models import Datasets, DatasetsEnglishTranslations


class DatasetsSerializer(serializers.ModelSerializer):
    """
    Serializer for the Datasets class.

    Converts instances of the Datasets class to and from JSON representations.

    Attributes:
        model (class): The model class to be serialized.
        fields (list): List of fields to include in the serialization.
        read_only_fields (tuple): Tuple of read-only fields in the serialization.
    """

    class Meta:
        model = Datasets
        fields = "__all__"
        read_only_fields = ("created_at",)


class DatasetsEnglishTranslationsSerializer(serializers.ModelSerializer):
    """
    Serializer for the DatasetsEnglishTranslations class.

    Converts instances of the DatasetsEnglishTranslations class to and from JSON representations.

    Attributes:
        model (class): The model class to be serialized.
        fields (list): List of fields to include in the serialization.
        read_only_fields (tuple): Tuple of read-only fields in the serialization.
    """

    class Meta:
        model = DatasetsEnglishTranslations
        fields = "__all__"
        read_only_fields = ("created_at",)


class DatasetSyncSerializer(serializers.Serializer):
    authorities = serializers.PrimaryKeyRelatedField(
        queryset=Authorities.objects.all(), many=True
    )
