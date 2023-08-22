"""
This module contains viewsets and filters for the API endpoints related to categories, 
translations, authorities, and text classification.

Classes:
- CategoriesFilter: A filter class for filtering categories based on various criteria.
- Categories_ViewSet: A viewset for handling CRUD operations on categories.
- Translations_ViewSet: A viewset for handling CRUD operations on translations.
- Authorities_ViewSet: A viewset for handling CRUD operations on authorities.
- Text_Classification_ViewSet: A viewset for performing text classification.

Functions:
- create: A function for handling the creation of text classification predictions.

Note: This code assumes the existence of the necessary models, serializers, 
and neural network module.
"""

from rest_framework import viewsets
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from django_filters import rest_framework as filters
from rest_framework.permissions import IsAdminUser, AllowAny
from rest_framework.response import Response
from .neural_network.text_classifier import TextClassifier
from .models import Categories, Translations, Authorities
from .serializers import (
    CategoriesSerializer,
    TranslationsSerializer,
    AuthoritySerializer,
    TextClassificationSerializer,
)


class CategoriesFilter(filters.FilterSet):
    """
    FilterSet for Categories model.

    Attributes:
        deprecated (filters.BooleanFilter): Filter for deprecated field.
        name (filters.CharFilter): Filter for name field.
        searched_for_datasets (filters.BooleanFilter): Filter for searched_for_datasets field.
        label_index (filters.NumberFilter): Filter for label_index field.
        authority (filters.ModelMultipleChoiceFilter): Filter for authority field.
    """

    deprecated = filters.BooleanFilter()
    name = filters.CharFilter()
    searched_for_datasets = filters.BooleanFilter()
    label_index = filters.NumberFilter()
    authority = filters.ModelMultipleChoiceFilter(queryset=Authorities.objects.all())

    class Meta:
        model = Categories
        fields = [
            "deprecated",
            "name",
            "searched_for_datasets",
            "authority",
            "label_index",
        ]


class CategoriesViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Categories model.

    Attributes:
        queryset (QuerySet): QuerySet for Categories model.
        permission_classes (list): List of permission classes.
        serializer_class (Serializer): Serializer class for Categories model.
        filter_backends (list): List of filter backends.
        search_fields (list): List of fields to search on.
        filterset_class (CategoriesFilter): FilterSet class for Categories model.
    """

    queryset = Categories.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = CategoriesSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["name"]
    filterset_class = CategoriesFilter

    def get_permissions(self):
        """
        Get the permissions required for the current action.

        Returns:
            list: List of permission classes.
        """
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [
                IsAdminUser()
            ]  # Solo permitir acceso a usuarios administradores para hacer cambios
        return super().get_permissions()


class TranslationsViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Translations model.

    Attributes:
        queryset (QuerySet): QuerySet for Translations model.
        permission_classes (list): List of permission classes.
        serializer_class (Serializer): Serializer class for Translations model.
        filter_backends (list): List of filter backends.
        search_fields (list): List of fields to search on.
    """

    queryset = Translations.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = TranslationsSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["name"]

    def get_permissions(self):
        """
        Get the permissions required for the current action.

        Returns:
            list: List of permission classes.
        """
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [
                IsAdminUser()
            ]  # Solo permitir acceso a usuarios administradores para hacer cambios
        return super().get_permissions()


class AuthoritiesViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Authorities model.

    Attributes:
        queryset (QuerySet): QuerySet for Authorities model.
        permission_classes (list): List of permission classes.
        serializer_class (Serializer): Serializer class for Authorities model.
        filter_backends (list): List of filter backends.
        search_fields (list): List of fields to search on.
    """

    queryset = Authorities.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = AuthoritySerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["name"]

    def get_permissions(self):
        """
        Get the permissions required for the current action.

        Returns:
            list: List of permission classes.
        """
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [
                IsAdminUser()
            ]  # Solo permitir acceso a usuarios administradores para hacer cambios
        return super().get_permissions()


class TextClassificationViewSet(viewsets.ViewSet):
    """
    ViewSet for text classification.

    Attributes:
        serializer_class (Serializer): Serializer class for text classification.
    """

    serializer_class = TextClassificationSerializer

    def create(self, request):
        """
        Create a new text classification.

        Args:
            request (Request): The request object.

        Returns:
            Response: The response object containing the predicted labels.
        """
        title = request.data.get("title")
        summary = request.data.get("summary")
        text_classifier = TextClassifier()

        predicted_labels = text_classifier.classify_text(f"{title}: {summary}")

        return Response(predicted_labels)
