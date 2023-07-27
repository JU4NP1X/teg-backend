from rest_framework import serializers
from django.contrib.auth.models import User


class Users_Serializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "password"]
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user


class User_Login_Serializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()
