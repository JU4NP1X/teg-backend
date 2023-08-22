from rest_framework import routers
from .api import DatasetsEnglishTranslationsViewSet, DatasetsViewSet

router = routers.DefaultRouter()

router.register("list", DatasetsViewSet)
router.register("translations", DatasetsEnglishTranslationsViewSet)

urlpatterns = router.urls
