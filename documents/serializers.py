import base64
from rest_framework import serializers
from categories.models import Categories, Translations, Authorities
from categories.serializers import TranslationsSerializer
from users.models import User
from .models import Documents


def get_predicted_trees():
    try:
        return Categories.objects.filter(
            deprecated=False, parent=None, level=0
        ).values_list("tree_id", "name")
    except Exception as e:
        return []


class UsersSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["username"]


class CategoriesSerializer(serializers.ModelSerializer):
    translation = serializers.SerializerMethodField()
    authority = serializers.SerializerMethodField(method_name="get_authority")

    class Meta:
        model = Categories
        exclude = ["lft", "rght", "level"]

    def get_translation(self, obj):
        translation = Translations.objects.filter(category=obj, language="es").first()
        if translation:
            serializer = TranslationsSerializer(translation)
            return serializer.data
        return None


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["username"]


class DocumentsSerializer(serializers.ModelSerializer):
    pdf = serializers.CharField(
        max_length=None,
        style={"placeholder": "Enter the base64 of the pdf"},
    )
    img = serializers.CharField(
        max_length=None,
        style={"placeholder": "Enter the base64 of the img"},
        write_only=True,
    )

    category = serializers.SerializerMethodField()
    created_by = serializers.SerializerMethodField()
    updated_by = serializers.SerializerMethodField()
    predicted_trees = serializers.MultipleChoiceField(
        choices=get_predicted_trees(),
        write_only=True,
    )

    class Meta:
        model = Documents
        fields = "__all__"
        read_only_fields = (
            "created_at",
            "updated_at",
            "created_by",
            "updated_by",
            "num_of_access",
        )

    def get_category(self, obj):
        categories = obj.categories.filter(authority__disabled=False)
        if categories:
            serializer = CategoriesSerializer(categories, many=True)
            return serializer.data
        return None

    def get_created_by(self, obj):
        serializer = UserSerializer(obj.created_by)
        return serializer.data

    def get_updated_by(self, obj):
        serializer = UserSerializer(obj.updated_by)
        return serializer.data

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        representation["pdf"] = None
        if self.context.get("request") and self.context["request"].path.endswith(
            f"/{instance.id}/"
        ):
            Documents.objects.filter(id=instance.id).update(
                num_of_access=instance.num_of_access + 1
            )
            if instance.pdf:
                pdf_base64 = base64.b64encode(instance.pdf).decode("utf-8")
                representation["pdf"] = pdf_base64

        return representation

    def create(self, validated_data):
        # Obtener el array de números adicional
        predicted_trees = self.context.get("predicted_trees")

        # Validar que todos los elementos sean números
        for num in predicted_trees:
            if not isinstance(num, (int, float)):
                raise serializers.ValidationError(
                    "The array must contain only numbers."
                )

        validated_data["predicted_trees"] = predicted_trees

        instance = super().create(validated_data)

        return instance

    def update(self, validated_data):
        # Obtener el array de números adicional
        predicted_trees = self.context.get("predicted_trees")

        # Validar que todos los elementos sean números
        for num in predicted_trees:
            if not isinstance(num, (int, float)):
                raise serializers.ValidationError(
                    "The array must contain only numbers."
                )

        validated_data["predicted_trees"] = predicted_trees

        instance = super().create(validated_data)

        return instance

    def partial_update(self, validated_data):
        # Obtener el array de números adicional
        predicted_trees = self.context.get("predicted_trees")

        # Validar que todos los elementos sean números
        for num in predicted_trees:
            if not isinstance(num, (int, float)):
                raise serializers.ValidationError(
                    "The array must contain only numbers."
                )

        validated_data["predicted_trees"] = predicted_trees

        instance = super().create(validated_data)

        return instance


class DocumentsTextExtractorSerializer(serializers.Serializer):
    """
    Serializer for the DocumentsTextExtractor.

    Converts instances of the DocumentsTextExtractor to JSON and vice versa.

    Attributes:
        title (CharField): The base64 encoded title of the document.
        summary (CharField): The base64 encoded summary of the document.
    """

    title = serializers.CharField(
        max_length=None, style={"placeholder": "Enter the base64 for the title"}
    )
    summary = serializers.CharField(
        max_length=None, style={"placeholder": "Enter the base64 for the summary"}
    )
