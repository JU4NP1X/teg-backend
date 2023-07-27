from .models import Documents
from rest_framework import viewsets, permissions
from .serializers import Documents_Serializer
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
    def create(self, request):
        title_base64 = request.data.get("title")
        resume_base64 = request.data.get("resume")
        # Decodificar las imágenes base64 a objetos de imagen
        title = Image.open(io.BytesIO(base64.b64decode(title_base64)))
        resume = Image.open(io.BytesIO(base64.b64decode(resume_base64)))

        # Realizar el análisis OCR en las imágenes utilizando Tesseract OCR
        title_text = " ".join(
            pytesseract.image_to_string(title).split(), config=custom_config
        )
        resume_text = " ".join(
            pytesseract.image_to_string(resume).split(), config=custom_config
        )

        # Ejemplo de respuesta
        response_data = {
            "title": title_text,
            "resume": resume_text,
        }

        return Response(response_data)
