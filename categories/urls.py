from rest_framework import routers
from .api import (
    CategoriesViewSet,
    TranslationsViewSet,
    AuthoritiesViewSet,
    TextClassificationViewSet,
    TrainAuthorityViewSet,
    GetAuthorityCategoriesViewSet,
)

router = routers.DefaultRouter()

router.register("list", CategoriesViewSet)
router.register("translations", TranslationsViewSet)
router.register("authorities", AuthoritiesViewSet)
router.register("classify", TextClassificationViewSet, basename="classify")
router.register("train", TrainAuthorityViewSet, basename="train")
router.register("csv", GetAuthorityCategoriesViewSet, basename="csv")


urlpatterns = router.urls
