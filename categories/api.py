"""
This module contains viewsets and filters for the API endpoints related to categories, 
translations, authorities, and text classification.

Classes:
- CategoriesFilter: A filter class for filtering categories based on various criteria.
- CategoriesViewSet: A viewset for handling CRUD operations on categories.
- TranslationsViewSet: A viewset for handling CRUD operations on translations.
- AuthoritiesViewSet: A viewset for handling CRUD operations on authorities.
- TextClassificationViewSet: A viewset for performing text classification.

Functions:
- create: A function for handling the creation of text classification predictions.

Note: This code assumes the existence of the necessary models, serializers, 
and neural network module.
"""

import os
from datetime import datetime
from django.conf import settings
from threading import Lock
from datetime import datetime, timezone
from rest_framework import status
from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets, status
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.permissions import IsAdminUser, AllowAny, IsAuthenticated
from rest_framework.response import Response
from .neural_network.text_classifier import TextClassifier
from .models import Categories, Translations, Authorities
from .serializers import (
    CategoriesSerializer,
    TranslationsSerializer,
    AuthoritySerializer,
    TextClassificationSerializer,
)
from django.conf import settings

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


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
    exclude = filters.BaseInFilter(field_name="id", lookup_expr="in", exclude=True)

    class Meta:
        model = Translations
        fields = ["name", "language", "authority"]


class CategoriesViewSet(viewsets.ModelViewSet):
    """
    Categories of the documents and datasets classification
    """

    queryset = Categories.objects.filter(parent=None)
    permission_classes = [AllowAny]  # Allow access to anyone to view
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
            return [IsAdminUser()]  # Only allow access to admin users to make changes
        return super().get_permissions()


class TranslationsViewSet(viewsets.ModelViewSet):
    """
    Translations of the categories
    """

    queryset = Translations.objects.all()
    permission_classes = [AllowAny]  # Allow access to anyone to view
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
            return [IsAdminUser()]  # Only allow access to admin users to make changes
        return super().get_permissions()

    def get_queryset(self):
        """
        Get the filtered queryset based on the request parameters.

        Returns:
            queryset: Filtered queryset.
        """
        queryset = super().get_queryset()
        filter_params = self.request.GET.dict()
        exclude_ids = filter_params.pop("exclude", None)
        if exclude_ids:
            queryset = queryset.exclude(id__in=exclude_ids.split(","))
        return queryset


class AuthoritiesViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Authorities that are the owners of the categories.
    """

    queryset = Authorities.objects.all()
    permission_classes = [AllowAny]  # Allow access to anyone to view
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
            return [IsAdminUser()]  # Only allow access to admin users to make changes
        return super().get_permissions()

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.native:
            return Response(
                {"detail": "Cannot delete a native authority."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return super().destroy(request, *args, **kwargs)

    def get_queryset(self):
        queryset = super().get_queryset()

        # Get the value of the exclude_counts parameter from the URL
        exclude_counts = self.request.query_params.get("exclude_counts", False)
        # Pass the value of the parameter to the serializer
        self.serializer_class.exclude_counts = exclude_counts

        return queryset


class TextClassificationViewSet(viewsets.ViewSet):
    """
    ViewSet for text classification.

    Attributes:
        serializer_class (Serializer): Serializer class for text classification.
    """

    serializer_class = TextClassificationSerializer
    permission_classes = [IsAuthenticated]  # Allow access to anyone to view

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
        authority_id = request.data.get("authority_id")
        model_path = os.path.join(BASE_DIR, "trained_model")
        model_checkpoint = f"{model_path}/{authority_id}/model.ckpt"

        if not os.path.exists(model_path) or not os.path.exists(model_checkpoint):
            return Response("No trained classifier available for this authority")

        # Check if the lock exists
        if authority_id not in settings.CLASSIFIERS_LOCKS:
            settings.CLASSIFIERS_LOCKS[authority_id] = Lock()

        with settings.CLASSIFIERS_LOCKS[authority_id]:
            # Check if the text_classifier exists
            authority = Authorities.objects.filter(id=authority_id).first()

            if not authority.last_training_date:
                return Response("No trained classifier available for this authority")

            if authority_id not in settings.TEXT_CLASSIFIERS:
                text_classifier = settings.TEXT_CLASSIFIERS[
                    authority_id
                ] = TextClassifier(
                    authority_id=authority_id, loaded_at=datetime.now(timezone.utc)
                )
            else:
                text_classifier = settings.TEXT_CLASSIFIERS[authority_id]

            # Check if the classifier needs to be reloaded
            if text_classifier.loaded_at < authority.last_training_date:
                text_classifier = TextClassifier(
                    authority_id=authority_id, loaded_at=datetime.now(timezone.utc)
                )
                settings.TEXT_CLASSIFIERS[authority_id] = text_classifier

            predicted_labels = text_classifier.classify_text(f"{title}: {summary}")
            serializer = CategoriesSerializer(predicted_labels)
            serialized_data = serializer.data

            return Response(serialized_data)
