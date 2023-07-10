from rest_framework import routers
from .api import DatasetsViewSet

router = routers.DefaultRouter()

router.register('api/datasets', DatasetsViewSet, basename='datasets')

urlpatterns = router.urls
