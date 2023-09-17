from rest_framework.routers import DefaultRouter
from .api import SystemInfoViewSet


router = DefaultRouter()

router.register("system-info", SystemInfoViewSet, basename="system-info")

urlpatterns = router.urls
