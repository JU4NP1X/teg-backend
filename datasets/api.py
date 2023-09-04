import subprocess
from rest_framework import viewsets
from rest_framework.permissions import IsAdminUser
from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import status
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.response import Response
from categories.models import Authorities
from .models import Datasets, DatasetsEnglishTranslations
from .serializers import (
    DatasetsSerializer,
    DatasetsEnglishTranslationsSerializer,
    DatasetSyncSerializer,
)


class DatasetTranslationFilter(filters.FilterSet):
    """
    FilterSet for Categories model.
    """

    paper_name = filters.CharFilter()

    authority = filters.CharFilter(method="filter_authority", field_name="authority")

    def filter_authority(self, queryset, name, value):
        return queryset.filter(dataset__categories__authority__id=value).distinct()

    class Meta:
        model = DatasetsEnglishTranslations
        fields = ["paper_name", "authority"]


class DatasetsViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Datasets.

    This ViewSet provides CRUD operations for Datasets.

    """

    queryset = Datasets.objects.all().distinct()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["paper_name"]
    permission_classes = [IsAdminUser]  # Only allow access to admin users
    serializer_class = DatasetsSerializer


class DatasetsEnglishTranslationsViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Datasets English Translations.
    """

    queryset = DatasetsEnglishTranslations.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["paper_name"]
    permission_classes = [IsAdminUser]  # Only allow access to admin users
    serializer_class = DatasetsEnglishTranslationsSerializer
    filterset_class = DatasetTranslationFilter

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context["authority"] = self.request.query_params.get("authority")
        return context


class DatasetSyncViewSet(viewsets.ViewSet):
    """
    Syncronize starter.
    """

    serializer_class = DatasetSyncSerializer

    def create(self, request):
        if request.user.is_superuser:
            serializer = DatasetSyncSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            authorities = serializer.validated_data["authorities"]
            authorities_ids = [authority.id for authority in authorities]
            authorities_without_pid = Authorities.objects.filter(
                id__in=authorities_ids, pid=0
            )
            authorities_without_pid.update(status="GETTING_DATA")

            authorities_ids_str = " ".join(str(id) for id in authorities_ids)
            subprocess.Popen(
                [
                    "python",
                    "./manage.py",
                    "datasets_sync",
                    "--authorities",
                    authorities_ids_str,
                ]
            )
            return Response(
                {"message": "Action initiated successfully"},
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"message": "Only administrators can execute this action"},
                status=status.HTTP_403_FORBIDDEN,
            )
