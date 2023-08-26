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
from rest_framework.permissions import IsAdminUser, AllowAny
from django.db.models import Count, Q
from rest_framework.response import Response
from django_filters import rest_framework as filters
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


class TranslationsFilter(filters.FilterSet):
    """
    FilterSet for Categories model.
    """

    name = filters.CharFilter()
    language = filters.CharFilter()
    authority = filters.BaseInFilter(field_name="categories__authority__id")
    exclude = filters.BaseInFilter(field_name="id", lookup_expr="inverted")

    class Meta:
        model = Translations
        fields = ["name", "language", "authority"]


class CategoriesViewSet(viewsets.ModelViewSet):
    """
    Categories of the documents and datasets classification
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
    Translations of the categories
    """

    queryset = Translations.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = TranslationsSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["name"]
    filterset_class = TranslationsFilter

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
    ViewSet for Authorities that are the owners of the categories.
    """

    queryset = Authorities.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = AuthoritySerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
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

    def get_queryset(self):
        queryset = super().get_queryset()

        # Obtener el valor del parámetro exclude_counts de la URL
        exclude_counts = self.request.query_params.get("exclude_counts", False)
        # Pasar el valor del parámetro al serializador
        self.serializer_class.exclude_counts = exclude_counts

        return queryset


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
