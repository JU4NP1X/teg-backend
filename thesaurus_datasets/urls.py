from rest_framework import routers
from .api import DatasetsViewSet

router = routers.DefaultRouter()

router.register('api/thesaurus-datasets', DatasetsViewSet, basename='thesaurus_datasets')

urlpatterns = router.urls
