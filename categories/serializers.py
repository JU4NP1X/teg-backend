from rest_framework import serializers
from .models import Categories, Translations, Authorities


class Categories_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Categories
        fields = "__all__"
        read_only_fields = ("created_at",)


class Authority_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Authorities
        fields = "__all__"
        read_only_fields = ("created_at",)


class Translations_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Translations
        fields = "__all__"
        read_only_fields = ("created_at",)


class Text_Classification_Serializer(serializers.Serializer):
    title = serializers.CharField(max_length=200)
    summary = serializers.CharField(max_length=1500)
