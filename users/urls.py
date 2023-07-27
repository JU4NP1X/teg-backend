from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api import Users_ViewSet, Login_ViewSet


router = DefaultRouter()

router.register("list", Users_ViewSet)
router.register("login", Login_ViewSet, basename="login")

urlpatterns = router.urls
