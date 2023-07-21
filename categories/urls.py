from rest_framework import routers
from .api import CategoriesViewSet

router = routers.DefaultRouter()

router.register('api/categories', CategoriesViewSet, basename='categories')


urlpatterns = router.urls
