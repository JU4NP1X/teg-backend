import base64
from rest_framework import serializers
from .models import Documents


class DocumentsSerializer(serializers.ModelSerializer):
    """
    Serializer for the Documents model.

    Converts instances of the Documents model to JSON and vice versa.

    Attributes:
        model (Documents): The Documents model to be serialized.
        fields (str): The fields to include in the serialization.
        read_only_fields (tuple): The fields that are read-only in the serialization.
    """

    class Meta:
        model = Documents
        fields = "__all__"
        read_only_fields = ("created_at", "updated_at", "created_by", "updated_by")

    def to_representation(self, instance):
        """
        Convert the instance to a representation that includes the base64 encoded PDF.

        Args:
            instance (Documents): The instance to be converted.

        Returns:
            dict: The representation of the instance.
        """
        representation = super().to_representation(instance)
        pdf_base64 = base64.b64encode(instance.pdf.read()).decode("utf-8")
        representation["pdf"] = pdf_base64

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
