from django.db.models import Count, Q, Subquery, OuterRef, IntegerField
from django.db.models.functions import Coalesce
from rest_framework import serializers
from .models import Categories, Translations, Authorities
from datasets.models import Datasets


class CategoriesSerializer(serializers.ModelSerializer):
    children = serializers.SerializerMethodField()
    translation = serializers.SerializerMethodField()
    authority = serializers.SerializerMethodField()

    class Meta:
        model = Categories
        exclude = ["lft", "rght", "level"]

    def get_children(self, obj):
        children = obj.get_children()
        serializer = self.__class__(children, many=True)
        return serializer.data

    def get_translation(self, obj):
        translation = Translations.objects.filter(category=obj, language="es").first()
        if translation:
            serializer = TranslationsSerializer(translation)
            return serializer.data
        return None

    def get_authority(self, obj):
        serializer = AuthoritySerializerAlt(obj.authority)
        return serializer.data


class CategoriesSerializerAlt(serializers.ModelSerializer):
    class Meta:
        model = Categories
        exclude = ["lft", "rght", "level"]


class AuthoritySerializerAlt(serializers.ModelSerializer):
    """
    Serializer for the Authorities model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    class Meta:
        model = Authorities
        fields = "__all__"
        read_only_fields = ("created_at", "native")


class AuthoritySerializer(serializers.ModelSerializer):
    """
    Serializer for the Authorities model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    resume = serializers.SerializerMethodField()
    exclude_counts = False

    class Meta:
        model = Authorities
        fields = "__all__"
        read_only_fields = ("created_at", "native")

    def get_resume(self, obj):
        if not self.exclude_counts:
            return (
                Authorities.objects.filter(id=obj.id)
                .annotate(
                    datasets_count=Coalesce(
                        Datasets.objects.filter(
                            categories__deprecated=False,
                            categories__authority__id=obj.id,
                        )
                        .values("id")
                        .distinct()
                        .count(),
                        0,
                        output_field=IntegerField(),
                    ),
                    category_not_trained_count=Coalesce(
                        Count(
                            "categories",
                            filter=Q(
                                categories__label_index__isnull=True,
                                categories__parent=None,
                                categories__deprecated=False,
                            ),
                        ),
                        0,
                    ),
                    category_trained_count=Coalesce(
                        Count(
                            "categories",
                            filter=Q(
                                categories__label_index__isnull=False,
                                categories__parent__isnull=True,
                                categories__deprecated=False,
                            ),
                        ),
                        0,
                    ),
                    deprecated_category_trained_count=Coalesce(
                        Count(
                            "categories",
                            filter=Q(
                                categories__label_index__isnull=False,
                                categories__parent__isnull=True,
                                categories__deprecated=True,
                            ),
                        ),
                        0,
                    ),
                    representated_category_count=Coalesce(
                        Categories.objects.filter(
                            deprecated=False,
                            parent=None,
                            authority__id=obj.id,
                        )
                        .exclude(datasets=None)
                        .count(),
                        0,
                        output_field=IntegerField(),
                    ),
                )
                .values(
                    "datasets_count",
                    "category_trained_count",
                    "category_not_trained_count",
                    "deprecated_category_trained_count",
                    "representated_category_count",
                )
                .first()
            )
        return {}


class TranslationsSerializer(serializers.ModelSerializer):
    """
    Serializer for the Translations model.

    Attributes:
        Meta (class): The metadata class for the serializer.
    """

    authority = serializers.SerializerMethodField()

    category = serializers.SerializerMethodField()

    class Meta:
        model = Translations
        fields = "__all__"
        read_only_fields = ("created_at",)

    def get_authority(self, obj):
        serializer = AuthoritySerializerAlt(obj.category.authority)
        return serializer.data

    def get_category(self, obj):
        serializer = CategoriesSerializerAlt(obj.category)
        return serializer.data


class TextClassificationSerializer(serializers.Serializer):
    """
    Serializer for text classification.

    Attributes:
        title (CharField): The title field for text classification.
        summary (CharField): The summary field for text classification.
    """

    title = serializers.CharField(max_length=200)
    summary = serializers.CharField(max_length=1500)
    authority_id = serializers.PrimaryKeyRelatedField(
        queryset=Authorities.objects.all(), many=False
    )


class TrainAuthoritySerializer(serializers.Serializer):
    """
    Serializer of training authority
    """
