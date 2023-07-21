from .models import Datasets,Datasets_English_Translations
from rest_framework import viewsets, permissions
from .serializers import DatasetsSerializer, Datasets_English_TranslationsSerializer


class DatasetsViewSet(viewsets.ModelViewSet):
    queryset = Datasets.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = DatasetsSerializer


class Datasets_English_TranslationsViewSet(viewsets.ModelViewSet):
    queryset = Datasets_English_Translations.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = Datasets_English_TranslationsSerializer
