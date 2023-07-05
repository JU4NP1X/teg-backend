from .models import Thesaurus
from rest_framework import viewsets, permissions
from .serializers import ThesaurusSerializer

class ThesaurusViewSet(viewsets.ModelViewSet):
  queryset = Thesaurus.objects.all()
  permission_classes = [permissions.AllowAny]
  serializer_class = ThesaurusSerializer