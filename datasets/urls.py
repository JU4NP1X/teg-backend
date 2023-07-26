from rest_framework import routers
from .api import Datasets_English_Translations_ViewSet, Datasets_ViewSet

router = routers.DefaultRouter()

router.register("list", Datasets_ViewSet)
router.register("translations", Datasets_English_Translations_ViewSet)

urlpatterns = router.urls
