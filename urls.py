from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("categories/", include("categories.urls")),
    path("datasets/", include("datasets.urls")),
    path("documents/", include("documents.urls")),
]
