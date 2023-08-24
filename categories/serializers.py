from django.db.models import Count, Q, Subquery, OuterRef
from rest_framework import serializers
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.permissions import AllowAny, IsAdminUser
from .models import Categories, Translations, Authorities


class CategoriesSerializer(serializers.ModelSerializer):
    """
    Serializer for the Categories model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    class Meta:
        model = Categories
        fields = "__all__"
        read_only_fields = ("created_at", "")


class AuthoritySerializer(serializers.ModelSerializer):
    """
    Serializer for the Authorities model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    resume = serializers.SerializerMethodField()

    class Meta:
        model = Authorities
        fields = "__all__"
        read_only_fields = ("created_at", "native")

    def get_resume(self, obj):
        return Authorities.objects.annotate(
            category_not_trained_count=Count(
                "categories",
                filter=Q(
                    categories__label_index__isnull=True,
                    categories__parent_category__isnull=True,
                    categories__deprecated=False,
                ),
            ),
            category_trained_count=Count(
                "categories",
                filter=Q(
                    categories__label_index__isnull=False,
                    categories__parent_category__isnull=True,
                    categories__deprecated=False,
                ),
            ),
            deprecated_category_trained_count=Count(
                "categories",
                filter=Q(
                    categories__label_index__isnull=False,
                    categories__parent_category__isnull=True,
                    categories__deprecated=True,
                ),
            ),
            representated_category_count=Subquery(
                Categories.objects.filter(
                    deprecated=False,
                    parent_category__isnull=True,
                    datasets__categories__parent_category__isnull=True,
                    authority=OuterRef("pk"),  # Relacionar con la autoridad actual
                )
                .annotate(
                    total_datasets=Count("datasets")
                    + Count("datasets__categories__parent_category__datasets")
                    + Count(
                        "datasets__categories__parent_category__parent_category__datasets"
                    )
                )
                .filter(total_datasets__gte=50)
                .values("authority")
                .annotate(count=Count("authority"))
                .values("count")
            ),
            not_representated_category_count=Subquery(
                Categories.objects.filter(
                    deprecated=False,
                    parent_category__isnull=True,
                    datasets__categories__parent_category__isnull=True,
                    authority=OuterRef("pk"),  # Relacionar con la autoridad actual
                )
                .annotate(
                    total_datasets=Count("datasets")
                    + Count("datasets__categories__parent_category__datasets")
                    + Count(
                        "datasets__categories__parent_category__parent_category__datasets"
                    )
                )
                .filter(total_datasets__lt=50)
                .values("authority")
                .annotate(count=Count("authority"))
                .values("count")
            ),
        )


class TranslationsSerializer(serializers.ModelSerializer):
    """
    Serializer for the Translations model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    class Meta:
        model = Translations
        fields = "__all__"
        read_only_fields = ("created_at",)


class TextClassificationSerializer(serializers.Serializer):
    """
    Serializer for text classification.

    Attributes:
        title (CharField): The title field for text classification.
        summary (CharField): The summary field for text classification.
    """

    title = serializers.CharField(max_length=200)
    summary = serializers.CharField(max_length=1500)
