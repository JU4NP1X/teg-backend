from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from rest_framework.response import Response
from rest_framework.views import APIView


class AppsView(APIView):
    """
    Base Apps Router
    """

    def get(self, request, *args, **kwargs):
        apidocs = {
            "admin": request.build_absolute_uri("admin/"),
            "categories": request.build_absolute_uri("categories/"),
            "datasets": request.build_absolute_uri("datasets/"),
            "documents": request.build_absolute_uri("documents/"),
            "users": request.build_absolute_uri("users/"),
            "utils": request.build_absolute_uri("utils/"),
        }
        return Response(apidocs)


urlpatterns = [
    path("users/", include("users.urls")),
    path("admin/", admin.site.urls),
    path("categories/", include("categories.urls")),
    path("datasets/", include("datasets.urls")),
    path("documents/", include("documents.urls")),
    path("utils/", include("utils.urls")),
    path("", AppsView.as_view(), name="list_apps"),
]
