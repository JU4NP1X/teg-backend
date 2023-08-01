from .models import Datasets, Datasets_English_Translations
from rest_framework import viewsets
from .serializers import Datasets_Serializer, Datasets_English_Translations_Serializer
from rest_framework.permissions import IsAdminUser
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter


class Datasets_ViewSet(viewsets.ModelViewSet):
    """
    Datasets
    """

    queryset = Datasets.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["paper_name"]
    permission_classes = [
        IsAdminUser
    ]  # Solo permitir acceso a usuarios administradores
    serializer_class = Datasets_Serializer


class Datasets_English_Translations_ViewSet(viewsets.ModelViewSet):
    queryset = Datasets_English_Translations.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["paper_name"]
    permission_classes = [
        IsAdminUser
    ]  # Solo permitir acceso a usuarios administradores
    serializer_class = Datasets_English_Translations_Serializer
