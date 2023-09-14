import base64
from rest_framework import serializers
from categories.models import Categories, Translations, Authorities
from categories.serializers import TranslationsSerializer
from .models import Documents


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


class DocumentsSerializer(serializers.ModelSerializer):
    pdf = serializers.CharField(
        max_length=None, style={"placeholder": "Enter the base64 of the pdf"}
    )
    img = serializers.CharField(
        max_length=None, style={"placeholder": "Enter the base64 of the img"}
    )

    category = serializers.SerializerMethodField()

    class Meta:
        model = Documents
        fields = "__all__"
        read_only_fields = ("created_at", "updated_at", "created_by", "updated_by")

    def get_category(self, obj):
        categories = obj.categories.all()
        if categories:
            serializer = CategoriesSerializer(categories, many=True)
            return serializer.data
        return None

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        representation["pdf"] = None
        if self.context.get("request") and self.context["request"].path.endswith(
            f"/{instance.id}/"
        ):
            if instance.pdf:
                pdf_base64 = base64.b64encode(instance.pdf).decode("utf-8")
                representation["pdf"] = pdf_base64

        if instance.img:
            img_base64 = base64.b64encode(instance.img).decode("utf-8")
            representation["img"] = img_base64
        return representation


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
