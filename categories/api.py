from .models import Categories, Translations, Authorities
from rest_framework import viewsets, permissions
from .serializers import Categories_Serializer, Translations_Serializer, Authority_Serializer
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from django_filters import rest_framework as filters


class CategoriesFilter(filters.FilterSet):
    deprecated = filters.BooleanFilter()
    name = filters.CharFilter()
    searched_for_datasets = filters.BooleanFilter()

    class Meta:
        model = Categories
        fields = ["deprecated", "name", "searched_for_datasets"]


class Categories_ViewSet(viewsets.ModelViewSet):
    queryset = Categories.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = Categories_Serializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter,]
    search_fields = ["name"]
    filterset_class = CategoriesFilter


class Translations_ViewSet(viewsets.ModelViewSet):
    queryset = Translations.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = Translations_Serializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["name"]


class Authorities_ViewSet(viewsets.ModelViewSet):
    queryset = Authorities.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = Authority_Serializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["name"]