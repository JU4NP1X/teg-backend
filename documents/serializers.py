from rest_framework import serializers
from .models import Documents
import base64


class Documents_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Documents
        fields = "__all__"
        read_only_fields = ("created_at", "updated_at", "created_by", "updated_by")

    def to_representation(self, instance):
        representation = super().to_representation(instance)

        # Get the binary content of the pdf field
        pdf_binary = instance.pdf

        # Convert the binary content to base64
        pdf_base64 = base64.b64encode(pdf_binary).decode("utf-8")

        # Update the representation with the base64 encoded pdf
        representation["pdf"] = pdf_base64

        return representation
