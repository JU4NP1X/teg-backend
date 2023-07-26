from rest_framework import routers
from .api import Categories_ViewSet, Translations_ViewSet, Authorities_ViewSet

router = routers.DefaultRouter()

router.register("list", Categories_ViewSet)
router.register("translations", Translations_ViewSet)
router.register("authorities", Authorities_ViewSet)


urlpatterns = router.urls
