from rest_framework.routers import DefaultRouter
from .api import UsersViewSet, LoginViewSet, GoogleLoginViewSet


router = DefaultRouter()

router.register("list", UsersViewSet)
router.register("login", LoginViewSet, basename="login")
router.register("google-login", GoogleLoginViewSet, basename="google-login")

urlpatterns = router.urls
