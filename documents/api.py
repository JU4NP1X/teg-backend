from .models import Documents
from rest_framework import viewsets, permissions
from .serializers import Documents_Serializer, Documents_Text_Extractor_Serializer
import pytesseract
from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
import base64
import io
import os

# Configuración para el idioma español
custom_config = f"--oem 3 --psm 6 -l {os.getenv('TESSERACT_ALPHA3', 'eng')}"


class Documents_ViewSet(viewsets.ModelViewSet):
    queryset = Documents.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = Documents_Serializer


class Documents_Text_Extractor_ViewSet(viewsets.ViewSet):
    def get_view_name(self):
        return "Document Images Text Extractor"
    serializer_class = Documents_Text_Extractor_Serializer
    def create(self, request):
        title_base64 = request.data.get("title")
        summary_base64 = request.data.get("summary")
        # Decodificar las imágenes base64 a objetos de imagen
        title = Image.open(io.BytesIO(base64.b64decode(title_base64)))
        summary = Image.open(io.BytesIO(base64.b64decode(summary_base64)))

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
