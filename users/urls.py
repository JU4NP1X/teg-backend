from rest_framework.routers import DefaultRouter
from .api import UsersViewSet, LoginViewSet


router = DefaultRouter()

router.register("list", UsersViewSet)
router.register("login", LoginViewSet, basename="login")

urlpatterns = router.urls
