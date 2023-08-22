from rest_framework import serializers
from .models import Categories, Translations, Authorities


class CategoriesSerializer(serializers.ModelSerializer):
    """
    Serializer for the Categories model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    class Meta:
        model = Categories
        fields = "__all__"
        read_only_fields = ("created_at",)


class AuthoritySerializer(serializers.ModelSerializer):
    """
    Serializer for the Authorities model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    class Meta:
        model = Authorities
        fields = "__all__"
        read_only_fields = ("created_at",)


class TranslationsSerializer(serializers.ModelSerializer):
    """
    Serializer for the Translations model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    class Meta:
        model = Translations
        fields = "__all__"
        read_only_fields = ("created_at",)


class TextClassificationSerializer(serializers.Serializer):
    """
    Serializer for text classification.

    Attributes:
        title (CharField): The title field for text classification.
        summary (CharField): The summary field for text classification.
    """

    title = serializers.CharField(max_length=200)
    summary = serializers.CharField(max_length=1500)
