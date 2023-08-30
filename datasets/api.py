import subprocess
from rest_framework import viewsets
from rest_framework.permissions import IsAdminUser
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import status
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.response import Response
from .models import Datasets, DatasetsEnglishTranslations
from .serializers import (
    DatasetsSerializer,
    DatasetsEnglishTranslationsSerializer,
    DatasetSyncSerializer,
)


class DatasetsViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Datasets.

    This ViewSet provides CRUD operations for Datasets.

    """

    queryset = Datasets.objects.all()
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
                {"message": "Action initiated successfully"}, status=status.HTTP_200_OK
            )
        else:
            return Response(
                {"message": "Only administrators can execute this action"},
                status=status.HTTP_403_FORBIDDEN,
            )
