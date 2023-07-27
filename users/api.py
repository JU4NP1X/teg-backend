from rest_framework import viewsets 
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .serializers import Users_Serializer, User_Login_Serializer
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action

class Users_ViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = Users_Serializer
    permission_classes = [IsAdminUser]

    def get_permissions(self):
        if self.action == "update" or self.action == "partial_update":
            self.permission_classes = [IsAuthenticated]
        return super().get_permissions()

class Login_ViewSet(viewsets.ViewSet):
    serializer_class = User_Login_Serializer
    def create(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        user = authenticate(
            request,
            username=username,
            password=password,
        )
        if user is not None:
            token, created = Token.objects.get_or_create(user=user)
            return Response({"token": token.key})
        else:
            return Response({"error": "Invalid credentials"}, status=400)
