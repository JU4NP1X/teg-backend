from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from users.models import User


class UsersSerializer(serializers.ModelSerializer):
    """
    Serializer for the User model.

    Attributes:
        password (CharField): The password field for the user.
    """

    password = serializers.CharField(
        min_length=8, max_length=128, write_only=True, style={"input_type": "password"}
    )

    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name", "password"]
        extra_kwargs = {"password": {"write_only": True}}

    def validate_username(self, value):
        """
        Validate the username field.

        Args:
            value (str): The username value.

        Returns:
            str: The validated username value.

        Raises:
            serializers.ValidationError: If the username contains invalid characters.
        """
        # Verify if the username only contains letters and numbers
        if not value.isalnum():
            raise serializers.ValidationError(
                "El nombre de usuario solo puede contener letras y números."
            )
        return value

    def validate_password(self, value):
        """
        Validate the password field.

        Args:
            value (str): The password value.

        Returns:
            str: The validated password value.

        Raises:
            serializers.ValidationError: If the password does not meet the minimum requirements.
        """
        # Verify if the password meets the minimum requirements
        if len(value) < 8:
            raise serializers.ValidationError(
                "La contraseña debe tener al menos 8 caracteres."
            )
        return value

    def create(self, validated_data):
        """
        Create a new user.

        Args:
            validated_data (dict): The validated data for creating the user.

        Returns:
            User: The created user object.
        """
        validated_data["password"] = make_password(validated_data["password"])
        user = User.objects.create_user(**validated_data)
        return user

    def update(self, instance, validated_data):
        """
        Update an existing user.

        Args:
            instance (User): The user instance to be updated.
            validated_data (dict): The validated data for updating the user.

        Returns:
            User: The updated user object.
        """
        password = validated_data.pop("password", None)
        if password:
            validated_data["password"] = make_password(password)
        return super().update(instance, validated_data)


class UserLoginSerializer(serializers.Serializer):
    """
    Serializer for user login.

    Attributes:
        username (RegexField): The username field for login.
        password (CharField): The password field for login.
    """

    username = serializers.RegexField(r"^[a-zA-Z0-9]{1,20}$", max_length=20)
    password = serializers.CharField(
        min_length=8, max_length=128, write_only=True, style={"input_type": "password"}
    )
