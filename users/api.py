from rest_framework import viewsets
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from .models import User
from .serializers import UsersSerializer, UserLoginSerializer


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
        user = authenticate(
            request,
            username=username,
            password=password,
        )
        if user is not None:
            token, _ = Token.objects.get_or_create(user=user)
            serializer = UsersSerializer(user)  # Serialize the user object
            data = serializer.data
            data["token"] = token.key  # Add the token to the response data
            return Response(data)
        else:
            return Response({"error": "Invalid credentials"}, status=400)
