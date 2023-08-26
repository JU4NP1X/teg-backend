from rest_framework import routers
from .api import DatasetsEnglishTranslationsViewSet, DatasetsViewSet, DatasetSyncViewSet

router = routers.DefaultRouter()

router.register("list", DatasetsViewSet)
router.register("translations", DatasetsEnglishTranslationsViewSet)
router.register("sync", DatasetSyncViewSet, basename="sync")

urlpatterns = router.urls
