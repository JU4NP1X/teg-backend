from rest_framework import serializers
from django.contrib.auth.models import User


class Users_Serializer(serializers.ModelSerializer):
    password = serializers.CharField(min_length=8, max_length=128, write_only=True, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = ["id", "username", "email", "password"]
        extra_kwargs = {"password": {"write_only": True}}

    def validate_username(self, value):
        # Verificar si el username solo contiene letras y números
        if not value.isalnum():
            raise serializers.ValidationError(
                "El nombre de usuario solo puede contener letras y números."
            )
        return value

    def validate_password(self, value):
        # Verificar si la contraseña cumple con los requisitos mínimos
        if len(value) < 8:
            raise serializers.ValidationError(
                "La contraseña debe tener al menos 8 caracteres."
            )
        # Verificar si la contraseña contiene caracteres especiales
        if not any(char in value for char in "!@#$%^&*()_+"):
            raise serializers.ValidationError(
                "La contraseña debe contener al menos un carácter especial."
            )
        return value

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user


class User_Login_Serializer(serializers.Serializer):
    username = serializers.RegexField(r"^[a-zA-Z0-9]{1,20}$", max_length=20)
    password = serializers.CharField(min_length=8, max_length=128, write_only=True, style={'input_type': 'password'})
