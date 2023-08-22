import base64
import io
import os
import pytesseract
from rest_framework import viewsets, permissions
from rest_framework.response import Response
from PIL import Image
from .models import Documents
from .serializers import DocumentsSerializer, DocumentsTextExtractorSerializer

# Language config
custom_config = f"--oem 3 --psm 6 -l {os.getenv('TESSERACT_ALPHA3', 'eng')}"


class DocumentsViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows documents to be viewed or edited.

    Attributes:
        queryset (QuerySet): The queryset of Documents objects.
        permission_classes (list): The list of permission classes for the viewset.
        serializer_class (Serializer): The serializer class for Documents objects.
    """

    queryset = Documents.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = DocumentsSerializer


class DocumentsTextExtractorViewSet(viewsets.ViewSet):
    """
    Extracts text from images using OCR.
    """

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

        # Decodificar las imágenes base64 a objetos de imagen
        title = Image.open(io.BytesIO(base64.b64decode(title_base64.split(",")[1])))
        summary = Image.open(io.BytesIO(base64.b64decode(summary_base64.split(",")[1])))

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
        }

        return Response(response_data)
