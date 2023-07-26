from .models import Datasets,Datasets_English_Translations
from rest_framework import viewsets, permissions
from .serializers import Datasets_Serializer, Datasets_English_Translations_Serializer


class Datasets_ViewSet(viewsets.ModelViewSet):
    queryset = Datasets.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = Datasets_Serializer


class Datasets_English_Translations_ViewSet(viewsets.ModelViewSet):
    queryset = Datasets_English_Translations.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = Datasets_English_Translations_Serializer
