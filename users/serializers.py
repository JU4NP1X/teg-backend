from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from users.models import User


class UsersSerializer(serializers.ModelSerializer):
    """
    Serializer for the User model.

    Attributes:
        password (CharField): The password field for the user.
        is_admin (BooleanField): Indicates if the user is an administrator.
    """

    password = serializers.CharField(
        min_length=8, max_length=128, write_only=True, style={"input_type": "password"}
    )
    is_admin = serializers.BooleanField(required=False)

    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "password",
            "is_admin",
        ]
        extra_kwargs = {"password": {"write_only": True}}

    def get_is_admin(self, obj):
        """
        Get the value of the is_admin field.

        Args:
            obj (User): The User object.

        Returns:
            bool: True if the user is an administrator, False otherwise.
        """
        return obj.is_superuser

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

    def create(self, validated_data):
        """
        Create a new user.

        Args:
            validated_data (dict): The validated data for creating the user.

        Returns:
            User: The created user object.
        """
        password = validated_data.pop("password")
        is_admin = validated_data.pop("is_admin", False)
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.is_superuser = is_admin
        user.save()
        return user

    def to_representation(self, instance):
        """
        Convert the user instance to a representation.

        Args:
            instance (User): The User instance.

        Returns:
            dict: The serialized representation of the user.
        """
        representation = super().to_representation(instance)
        representation["is_admin"] = self.get_is_admin(instance)
        return representation

    def update(self, instance, validated_data):
        """
        Update an existing user.

        Args:
            instance (User): The user instance to update.
            validated_data (dict): The validated data for updating the user.

        Returns:
            User: The updated user object.
        """
        password = validated_data.pop("password", None)
        is_admin = validated_data.pop("is_admin", None)
        user = super().update(instance, validated_data)
        if password:
            user.set_password(password)
        if is_admin is not None:
            user.is_superuser = is_admin
        user.save()
        return user

    def partial_update(self, instance, validated_data):
        """
        Partially update an existing user.

        Args:
            instance (User): The user instance to update.
            validated_data (dict): The validated data for updating the user.

        Returns:
            User: The updated user object.
        """
        return self.update(instance, validated_data)


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


class UserGoogleLoginSerializer(serializers.Serializer):
    """
    Serializer for user login.

    Attributes:
        username (RegexField): The username field for login.
        password (CharField): The password field for login.
    """

    id_token = serializers.CharField(min_length=8, max_length=None)
