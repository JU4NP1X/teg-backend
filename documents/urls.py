from rest_framework import routers
from .api import (
    DocumentsViewSet,
    DocumentsTextExtractorViewSet,
    DocumentImageViewSet,
    DocumentPdfViewSet,
)

router = routers.DefaultRouter()

router.register("list", DocumentsViewSet)
router.register(
    "text-extractor", DocumentsTextExtractorViewSet, basename="text-extractor"
)

router.register(r"img", DocumentImageViewSet, basename="image")
router.register(r"pdf", DocumentPdfViewSet, basename="pdf")

urlpatterns = router.urls
