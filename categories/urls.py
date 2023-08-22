from rest_framework import routers
from .api import (
    CategoriesViewSet,
    TranslationsViewSet,
    AuthoritiesViewSet,
    TextClassificationViewSet,
)

router = routers.DefaultRouter()

router.register("list", CategoriesViewSet)
router.register("translations", TranslationsViewSet)
router.register("authorities", AuthoritiesViewSet)
router.register("classify", TextClassificationViewSet, basename="classify")


urlpatterns = router.urls
