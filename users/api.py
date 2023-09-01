from rest_framework import viewsets
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from google.auth.transport import requests
from .models import User
from .serializers import UsersSerializer, UserLoginSerializer, UserGoogleLoginSerializer


class UsersViewSet(viewsets.ModelViewSet):
    """
    Managing User objects.
    """

    queryset = User.objects.all()
    serializer_class = UsersSerializer
    permission_classes = [IsAdminUser]

    def get_permissions(self):
        """
        Get the permissions for the viewset based on the action.

        Returns:
            list: The list of permission classes.
        """
        if self.action == "update" or self.action == "partial_update":
            self.permission_classes = [IsAuthenticated]
        return super().get_permissions()


class LoginViewSet(viewsets.ViewSet):
    """
    Viewset for user login.
    """

    serializer_class = UserLoginSerializer

    def create(self, request):
        """
        Authenticate a user and generate an authentication token.

        Args:
            request (Request): The request object.

        Returns:
            Response: The response object containing the authentication token.
        """
        username = request.data.get("username")
        password = request.data.get("password")
        try:
            user = User.objects.get(username=username)
            print(user.check_password(password))
            if user.check_password(password):
                token, _ = Token.objects.get_or_create(user=user)
                serializer = UsersSerializer(user)  # Serialize the user object
                data = serializer.data
                data["token"] = token.key  # Add the token to the response data
                return Response(data)
        except User.DoesNotExist:
            pass

        return Response({"error": "Invalid credentials"}, status=400)


class GoogleLoginViewSet(viewsets.ViewSet):
    """
    Viewset for user login.
    """

    serializer_class = UserGoogleLoginSerializer

    def create(self, request):
        """
        Authenticate a user and generate an authentication token.

        Args:
            request (Request): The request object.

        Returns:
            Response: The response object containing the authentication token.
        """
        id_token = request.data.get("id_token")
        try:
            # Verify the ID token with Google Auth
            id_info = id_token.verify_oauth2_token(id_token, requests.Request())
            if id_info["iss"] not in [
                "accounts.google.com",
                "https://accounts.google.com",
            ]:
                raise ValueError("Invalid token")

            # Get or create the user based on the email
            user, created = User.objects.get_or_create(email=id_info["email"])

            if created:
                # Set additional user information if needed
                user.username = id_info["email"]
                user.save()

            # Generate the authentication token
            token, _ = Token.objects.get_or_create(user=user)
            serializer = UserGoogleLoginSerializer(user)  # Serialize the user object
            data = serializer.data
            data["token"] = token.key  # Add the token to the response data
            return Response(data)
        except ValueError:
            return Response({"error": "Invalid token"}, status=400)
