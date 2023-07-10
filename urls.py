from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("datasets/", include("datasets.urls")),
    path("thesaurus/", include("thesaurus.urls")),
]
