from .models import Datasets
from rest_framework import viewsets, permissions
from .serializers import DatasetsSerializer

class DatasetsViewSet(viewsets.ModelViewSet):
  queryset = Datasets.objects.all()
  permission_classes = [permissions.AllowAny]
  serializer_class = DatasetsSerializer