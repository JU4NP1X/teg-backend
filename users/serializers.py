from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from users.models import User

class Users_Serializer(serializers.ModelSerializer):
    password = serializers.CharField(min_length=8, max_length=128, write_only=True, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name", "password"]
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
        return value

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password'])
        user = User.objects.create_user(**validated_data)
        return user
        
    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        if password:
            validated_data['password'] = make_password(password)
        return super().update(instance, validated_data)


class User_Login_Serializer(serializers.Serializer):
    username = serializers.RegexField(r"^[a-zA-Z0-9]{1,20}$", max_length=20)
    password = serializers.CharField(min_length=8, max_length=128, write_only=True, style={'input_type': 'password'})
