from .models import Categories, Translations, Authorities
from rest_framework import viewsets, permissions
from .serializers import (
    Categories_Serializer,
    Translations_Serializer,
    Authority_Serializer,
    Text_Classification_Serializer,
)
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from django_filters import rest_framework as filters
from rest_framework.permissions import IsAdminUser, AllowAny

from .neural_network.text_classifier import TextClassifier
from rest_framework.response import Response


class CategoriesFilter(filters.FilterSet):
    deprecated = filters.BooleanFilter()
    name = filters.CharFilter()
    searched_for_datasets = filters.BooleanFilter()
    label_index = filters.NumberFilter()
    authority = filters.ModelMultipleChoiceFilter(queryset=Authorities.objects.all())

    class Meta:
        model = Categories
        fields = ["deprecated", "name", "searched_for_datasets", "authority", "label_index"]


class Categories_ViewSet(viewsets.ModelViewSet):
    queryset = Categories.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = Categories_Serializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["name"]
    filterset_class = CategoriesFilter

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [
                IsAdminUser()
            ]  # Solo permitir acceso a usuarios administradores para hacer cambios
        return super().get_permissions()


class Translations_ViewSet(viewsets.ModelViewSet):
    queryset = Translations.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = Translations_Serializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["name"]

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [
                IsAdminUser()
            ]  # Solo permitir acceso a usuarios administradores para hacer cambios
        return super().get_permissions()


class Authorities_ViewSet(viewsets.ModelViewSet):
    queryset = Authorities.objects.all()
    permission_classes = [AllowAny]  # Permitir acceso a cualquiera para ver
    serializer_class = Authority_Serializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["name"]

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [
                IsAdminUser()
            ]  # Solo permitir acceso a usuarios administradores para hacer cambios
        return super().get_permissions()


class Text_Classification_ViewSet(viewsets.ViewSet):
    serializer_class = Text_Classification_Serializer

    def create(self, request):
        title = request.data.get("title")
        summary = request.data.get("summary")
        text_classifier = TextClassifier()

        predicted_labels = text_classifier.classify_text(f"{title}: {summary}")

        return Response({"predicted_labels": predicted_labels})
