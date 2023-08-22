from rest_framework import routers
from .api import DocumentsViewSet, DocumentsTextExtractorViewSet

router = routers.DefaultRouter()

router.register("list", DocumentsViewSet)
router.register(
    "text-extractor", DocumentsTextExtractorViewSet, basename="text-extractor"
)

urlpatterns = router.urls
