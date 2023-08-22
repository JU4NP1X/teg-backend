from .models import Datasets, DatasetsEnglishTranslations
from rest_framework import viewsets
from .serializers import DatasetsSerializer, DatasetsEnglishTranslationsSerializer
from rest_framework.permissions import IsAdminUser
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter


class DatasetsViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Datasets.

    This ViewSet provides CRUD operations for Datasets.

    Attributes:
        queryset (QuerySet): The queryset of Datasets objects.
        filter_backends (list): The list of filter backends to be used.
        search_fields (list): The list of fields to be searched.
        permission_classes (list): The list of permission classes.
        serializer_class (Serializer): The serializer class to be used.
    """

    queryset = Datasets.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["paper_name"]
    permission_classes = [
        IsAdminUser
    ]  # Only allow access to admin users
    serializer_class = DatasetsSerializer


class DatasetsEnglishTranslationsViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Datasets English Translations.

    This ViewSet provides CRUD operations for Datasets English Translations.

    Attributes:
        queryset (QuerySet): The queryset of Datasets English Translations objects.
        filter_backends (list): The list of filter backends to be used.
        search_fields (list): The list of fields to be searched.
        permission_classes (list): The list of permission classes.
        serializer_class (Serializer): The serializer class to be used.
    """

    queryset = DatasetsEnglishTranslations.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["paper_name"]
    permission_classes = [
        IsAdminUser
    ]  # Only allow access to admin users
    serializer_class = DatasetsEnglishTranslationsSerializer
