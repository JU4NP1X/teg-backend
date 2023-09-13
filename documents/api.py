import base64
import io
import os
import pytesseract
from rest_framework import viewsets, permissions
from rest_framework.response import Response
from PIL import Image
from .models import Documents
from .serializers import DocumentsSerializer, DocumentsTextExtractorSerializer
from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from utils.response_messages import RESPONSE_MESSAGES
from categories.models import Categories

# Language config
custom_config = f"--oem 3 --psm 6 -l {os.getenv('TESSERACT_ALPHA3', 'eng')}"


class DocumentsFilter(filters.FilterSet):
    """
    FilterSet for Categories model.

    Attributes:
        deprecated (filters.BooleanFilter): Filter for deprecated field.
        name (filters.CharFilter): Filter for name field.
        searched_for_datasets (filters.BooleanFilter): Filter for searched_for_datasets field.
        label_index (filters.NumberFilter): Filter for label_index field.
        authority (filters.ModelMultipleChoiceFilter): Filter for authority field.
    """

    categories = filters.CharFilter(method="filter_categories")

    def filter_categories(self, queryset, name, value):
        category_ids = value.split(
            ","
        )  # Dividir la cadena en una lista de IDs de categorías
        return queryset.filter(categories__id__in=category_ids).distinct()

    class Meta:
        model = Documents
        fields = [
            "title",
            "summary",
            "categories",
        ]


class DocumentsViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows documents to be viewed or edited.
    """

    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["title"]
    filterset_class = DocumentsFilter

    queryset = Documents.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = DocumentsSerializer

    def get_permissions(self):
        """
        Get the permissions required for the current action.

        Returns:
            list: List of permission classes.
        """
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [
                permissions.IsAdminUser()
            ]  # Only allow access to admin users to make changes
        return super().get_permissions()

    def perform_create(self, serializer):
        """
        Set the created_by field to the current user when creating a new document.

        Args:
            serializer (Serializer): The serializer instance.
        """
        serializer.save(created_by=self.request.user, updated_by=self.request.user)

    def perform_update(self, serializer):
        """
        Set the updated_by field to the current user when updating a document.

        Args:
            serializer (Serializer): The serializer instance.
        """
        serializer.save(updated_by=self.request.user)


class DocumentsTextExtractorViewSet(viewsets.ViewSet):
    """
    Extracts text from images using OCR.
    """

    permission_classes = [permissions.IsAuthenticated]  # Allow access to anyone to view

    def get_view_name(self):
        return "Document Images Text Extractor"

    serializer_class = DocumentsTextExtractorSerializer

    def create(self, request):
        """
        Extracts text from images using OCR.

        Args:
            request (Request): The HTTP request object containing the base64 encoded images.

        Returns:
            Response: A Response object containing the extracted text from the images.
        """
        title_base64 = request.data.get("title")
        summary_base64 = request.data.get("summary")

        try:
            # Decodificar las imágenes base64 a objetos de imagen
            title = Image.open(io.BytesIO(base64.b64decode(title_base64.split(",")[1])))
            summary = Image.open(
                io.BytesIO(base64.b64decode(summary_base64.split(",")[1]))
            )

            # Realizar el análisis OCR en las imágenes utilizando Tesseract OCR
            title_text = " ".join(
                pytesseract.image_to_string(title, config=custom_config).split()
            )
            summary_text = " ".join(
                pytesseract.image_to_string(summary, config=custom_config).split()
            )

            # Ejemplo de respuesta
            response_data = {
                "title": title_text,
                "summary": summary_text,
                "message": RESPONSE_MESSAGES["TEXT_EXTRACTION_SUCCESS"]["message"],
            }
            return Response(
                response_data,
                status=RESPONSE_MESSAGES["TEXT_EXTRACTION_SUCCESS"]["code"],
            )
        except Exception as e:
            print(e)
            return Response(
                {"message": RESPONSE_MESSAGES["INVALID_IMAGE_FORMAT"]["message"]},
                status=RESPONSE_MESSAGES["INVALID_IMAGE_FORMAT"]["code"],
            )
