from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("thesaurus-datasets/", include("thesaurus_datasets.urls")),
    path("thesaurus/", include("thesaurus.urls")),
]
