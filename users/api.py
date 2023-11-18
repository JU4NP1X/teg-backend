import os
from rest_framework import viewsets
from rest_framework.permissions import IsAdminUser, IsAuthenticated, BasePermission
from rest_framework.response import Response
from google.auth.transport import requests
from rest_framework_simplejwt.tokens import RefreshToken, OutstandingToken
from google.oauth2 import id_token
from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from google.auth.transport import requests
from utils.response_messages import RESPONSE_MESSAGES
from .models import User
from .serializers import UsersSerializer, UserLoginSerializer, UserGoogleLoginSerializer


class UserFilter(filters.FilterSet):
    """
    FilterSet for Categories model.

    Attributes:
    """

    is_admin = filters.BaseInFilter(
        field_name="is_staff",
        label="If the user is an admin",
    )

    class Meta:
        model = User
        fields = ["first_name", "last_name", "email", "is_staff"]


class IsSelf(BasePermission):
    """
    Custom permission to only allow users to modify themselves.
    """

    def has_object_permission(self, request, view, obj):
        """
        Check if the user is modifying themselves.

        Args:
            request (Request): The request object.
            view (View): The view object.
            obj (User): The user object.

        Returns:
            bool: True if the user is modifying themselves, False otherwise.
        """
        return obj.id == request.user.id


class UsersViewSet(viewsets.ModelViewSet):
    """
    Managing User objects.
    """

    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["first_name", "last_name", "email"]
    filterset_class = UserFilter
    queryset = User.objects.all()
    serializer_class = UsersSerializer
    permission_classes = [IsAuthenticated]

    def get_permissions(self):
        """
        Get the permissions for the viewset based on the action.

        Returns:
            list: The list of permission classes.
        """
        if self.action == "update" or self.action == "partial_update":
            self.permission_classes = [IsAdminUser | IsSelf]
        elif self.action == "destroy":
            self.permission_classes = [IsAdminUser]
        else:
            self.permission_classes = [IsAuthenticated]
        return super().get_permissions()

    def list(self, request, *args, **kwargs):
        """
        Get the list of users.
        Args:
            request (Request): The request object.

        Returns:
            Response: The response object containing the list of users.
        """
        if not request.user.is_superuser:
            self.queryset = self.queryset.filter(id=request.user.id)
        return super().list(request, *args, **kwargs)


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
            if user.check_password(password):
                refresh = RefreshToken.for_user(user)
                serializer = UsersSerializer(user)
                data = serializer.data
                data["token"] = str(refresh.access_token)
                return Response(data)
        except User.DoesNotExist:
            pass

        return Response(
            {"error": RESPONSE_MESSAGES["INVALID_CREDENTIALS"]["message"]},
            status=RESPONSE_MESSAGES["INVALID_CREDENTIALS"]["code"],
        )

    def delete(self, request):
        # Obtén el token de acceso del encabezado de autorización
        authorization_header = request.headers.get("Authorization")
        if authorization_header:
            access_token = authorization_header.split(" ")[1]
            # Invalida el token de acceso actual
            OutstandingToken.objects.filter(token=access_token).update(
                is_blacklisted=True
            )

        return Response(
            {"message": RESPONSE_MESSAGES["LOGOUT"]["message"]},
            status=RESPONSE_MESSAGES["LOGOUT"]["code"],
        )


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
        id_token_value = request.data.get("id_token")
        try:
            # Verify the ID token with Google Auth
            id_info = id_token.verify_oauth2_token(id_token_value, requests.Request())
            # Check if the email domain is allowed
            email = id_info["email"]
            allowed_domains = os.environ.get("ALLOWED_DOMAINS", "").split(",")
            if not any(email.endswith(domain.strip()) for domain in allowed_domains):
                raise ValueError("Invalid email domain")

            # Get or create the user based on the email
            user, created = User.objects.get_or_create(email=email)

            if created:
                # Set additional user information if needed
                user.username = email
                user.save()

            # Generate the authentication token
            refresh = RefreshToken.for_user(user)
            serializer = UsersSerializer(user)
            data = serializer.data
            data["token"] = str(refresh.access_token)
            return Response(data)

        except ValueError:
            return Response(
                {"error": RESPONSE_MESSAGES["INVALID_TOKEN"]["message"]},
                status=RESPONSE_MESSAGES["INVALID_TOKEN"]["code"],
            )
