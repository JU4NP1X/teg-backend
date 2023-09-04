from rest_framework import serializers
from categories.models import Authorities, Categories
from .models import Datasets, DatasetsEnglishTranslations


class CategoriesSerializer(serializers.ModelSerializer):
    """
    Serializer for the Datasets class.

    Converts instances of the Datasets class to and from JSON representations.

    Attributes:
        model (class): The model class to be serialized.
        fields (list): List of fields to include in the serialization.
        read_only_fields (tuple): Tuple of read-only fields in the serialization.
    """

    class Meta:
        model = Categories
        fields = "__all__"
        read_only_fields = ("created_at",)


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

    categories = serializers.SerializerMethodField()

    class Meta:
        model = DatasetsEnglishTranslations
        fields = "__all__"
        read_only_fields = ("created_at",)

    def get_categories(self, obj):
        serializer = CategoriesSerializer(
            Categories.objects.filter(
                parent=None,
                deprecated=False,
                tree_id__in=obj.dataset.categories.values("tree_id").distinct(),
            ).all(),
            many=True,
        )
        return serializer.data

    categories = serializers.SerializerMethodField()


class DatasetSyncSerializer(serializers.Serializer):
    authorities = serializers.PrimaryKeyRelatedField(
        queryset=Authorities.objects.all(), many=True
    )
