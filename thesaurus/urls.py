from rest_framework import routers
from .api import ThesaurusViewSet

router = routers.DefaultRouter()

router.register('api/thesaurus', ThesaurusViewSet, basename='thesaurus')


urlpatterns = router.urls
