from .models import Categories
from rest_framework import viewsets, permissions
from .serializers import CategoriesSerializer

class CategoriesViewSet(viewsets.ModelViewSet):
  queryset = Categories.objects.all()
  permission_classes = [permissions.AllowAny]
  serializer_class = CategoriesSerializer