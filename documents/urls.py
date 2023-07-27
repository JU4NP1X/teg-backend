from rest_framework import routers
from .api import Documents_ViewSet, Documents_Text_Extractor_ViewSet

router = routers.DefaultRouter()

router.register("list", Documents_ViewSet)
router.register(
    "text-extractor", Documents_Text_Extractor_ViewSet, basename="text-extractor"
)

urlpatterns = router.urls
