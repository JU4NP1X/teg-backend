from rest_framework import routers
from .api import DatasetsViewSet, Datasets_English_TranslationsViewSet

router = routers.DefaultRouter()

router.register('api/datasets', DatasetsViewSet, basename='datasets')
router.register('api/datasets-english-translations', Datasets_English_TranslationsViewSet, basename='datasets')

urlpatterns = router.urls
