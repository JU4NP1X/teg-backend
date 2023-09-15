"""
This module contains viewsets and filters for the API endpoints related to categories, 
translations, authorities, and text classification.

Classes:
- CategoriesFilter: A filter class for filtering categories based on various criteria.
- CategoriesViewSet: A viewset for handling CRUD operations on categories.
- TranslationsViewSet: A viewset for handling CRUD operations on translations.
- AuthoritiesViewSet: A viewset for handling CRUD operations on authorities.
- TextClassificationViewSet: A viewset for performing text classification.

Functions:
- create: A function for handling the creation of text classification predictions.

Note: This code assumes the existence of the necessary models, serializers, 
and neural network module.
"""

import io
import os
import base64
import csv
import subprocess
from datetime import datetime, timezone
from threading import Lock
from django.conf import settings
from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Subquery, OuterRef, CharField, Max
from django.db.models.functions import Coalesce
from googletrans import Translator as GoogleTranslator
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.permissions import IsAdminUser, AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import viewsets, status
from utils.response_messages import RESPONSE_MESSAGES
from .neural_network.text_classifier import TextClassifier
from .models import Categories, Translations, Authorities
from .serializers import (
    CategoriesSerializer,
    TranslationsSerializer,
    AuthoritySerializer,
    TextClassificationSerializer,
    TrainAuthoritySerializer,
)


from .sync import categories_tree_adjust


BASE_DIR = os.path.dirname(os.path.realpath(__name__))


def has_invalid_relation(data):
    """
    Crear un diccionario para almacenar los padres de cada elemento
    """
    parents = {}
    names = set()

    # Recorrer los datos y almacenar los padres de cada elemento
    for row in data:
        element_id = row["id"]
        name = row["name"]
        parent_id = row["parent_id"]

        # Verificar si el elemento ya tiene un padre asignado
        if element_id in parents:
            return RESPONSE_MESSAGES["CIRCULAR_RELATIONSHIP"]
        # Verificar si el nombre del elemento ya ha sido utilizado
        if name in names:
            return RESPONSE_MESSAGES["DUPLICATE_NAME"]

        # Almacenar el padre del elemento
        parents[element_id] = parent_id
        names.add(name)

        # Verificar si el padre del elemento es el propio elemento (relación circular)
        if parent_id == element_id:
            return RESPONSE_MESSAGES["CIRCULAR_RELATIONSHIP"]

        # Verificar si el padre del elemento existe en los datos
        if parent_id and parent_id not in [row["id"] for row in data]:
            return RESPONSE_MESSAGES["INVALID_RELATIONSHIP"]

        # Verificar si hay una cadena de padres que forma una relación circular
        current_parent = parent_id
        while current_parent != "":
            if current_parent == element_id:
                return RESPONSE_MESSAGES["CIRCULAR_RELATIONSHIP"]
            current_parent = parents.get(current_parent, "")

    return {"code": 200}


def create_categories(authority_name, data):
    """
    Crear un diccionario para almacenar los padres de cada elemento
    """

    authority = Authorities.objects.filter(name=authority_name).first()
    Categories.objects.filter(authority=authority).update(deprecated=True)
    max_tree_id = Categories.objects.aggregate(Max("tree_id"))["tree_id__max"]

    # Recorrer los datos y almacenar los padres de cada elemento
    for row in data:
        name = row["name"]
        translation = row["translation"]
        category, _ = Categories.objects.update_or_create(
            name=name,
            authority=authority,
            defaults={"deprecated": False},
        )
        Translations.objects.update_or_create(
            category=category, language="es", defaults={"name": translation}
        )

    for row in data:
        name = row["name"]
        parent_id = row["parent_id"]  # Obtén el ID del padre desde los datos
        category = Categories.objects.filter(name=name, authority=authority).first()
        parent_name = None
        for parent_row in data:
            if parent_row["id"] == parent_id:
                parent_name = parent_row["name"]
                break
        parent = Categories.objects.filter(
            name=parent_name, authority=authority
        ).first()  # Busca el padre por su nombre en la base de datos
        if category and parent:
            max_tree_id += 1
            category.tree_id = max_tree_id
            category.move_to(parent, "last-child")
            category.save()
        elif not parent and category.level != 0:
            max_tree_id += 1
            category.tree_id = max_tree_id
            category.move_to(None, "last-child")
            category.save()

    categories_tree_adjust()


class CategoriesFilter(filters.FilterSet):
    """
    FilterSet for Categories model.

    Attributes:
        deprecated (filters.BooleanFilter): Filter for deprecated field.
        name (filters.CharFilter): Filter for name field.
        searched_for_datasets (filters.BooleanFilter): Filter for searched_for_datasets field.
        label_index (filters.NumberFilter): Filter for label_index field.
        authority (filters.ModelMultipleChoiceFilter): Filter for authority field.
    """

    deprecated = filters.BooleanFilter()
    name = filters.CharFilter()
    searched_for_datasets = filters.BooleanFilter()
    label_index = filters.NumberFilter()
    tree_id = filters.CharFilter(method="filter_tree_id")

    def filter_tree_id(self, queryset, name, value):
        tree_ids = value.split(
            ","
        )  # Dividir la cadena en una lista de IDs de categorías
        return queryset.filter(tree_id__in=tree_ids).distinct()

    authority = filters.ModelMultipleChoiceFilter(queryset=Authorities.objects.all())

    class Meta:
        model = Categories
        fields = [
            "deprecated",
            "name",
            "searched_for_datasets",
            "authority",
            "label_index",
            "tree_id",
        ]


class TranslationsFilter(filters.FilterSet):
    """
    FilterSet for Categories model.
    """

    name = filters.CharFilter()
    language = filters.CharFilter()
    authority = filters.BaseInFilter(
        field_name="category__authority__id",
        label="Comma separated authority",
    )
    exclude = filters.BaseInFilter(field_name="id", lookup_expr="in", exclude=True)

    class Meta:
        model = Translations
        fields = ["name", "language", "authority"]


class AuthorityFilter(filters.FilterSet):
    """
    FilterSet for Categories model.
    """

    name = filters.CharFilter()
    active = filters.BooleanFilter()

    class Meta:
        model = Authorities
        fields = ["name", "active"]


class GetAuthorityCategoriesFilter(filters.FilterSet):
    """
    FilterSet for Categories model.
    """

    authorities = filters.BaseInFilter(
        field_name="authority__id",
        label="Comma separated authority",
    )

    class Meta:
        model = Categories
        fields = ["authorities"]


class CategoriesViewSet(viewsets.ModelViewSet):
    """
    Categories of the documents and datasets classification
    """

    queryset = Categories.objects.filter(parent=None)
    permission_classes = [AllowAny]  # Allow access to anyone to view
    serializer_class = CategoriesSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["name"]
    filterset_class = CategoriesFilter

    def get_permissions(self):
        """
        Get the permissions required for the current action.
        Returns:
            list: List of permission classes.
        """
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [IsAdminUser()]  # Only allow access to admin users to make changes
        return super().get_permissions()


class TranslationsViewSet(viewsets.ModelViewSet):
    """
    Translations of the categories
    """

    queryset = Translations.objects.filter(category__authority__active=True)
    permission_classes = [AllowAny]  # Allow access to anyone to view
    serializer_class = TranslationsSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["name"]
    filterset_class = TranslationsFilter

    def get_permissions(self):
        """
        Get the permissions required for the current action.

        Returns:
            list: List of permission classes.
        """
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [IsAdminUser()]  # Only allow access to admin users to make changes
        return super().get_permissions()

    def get_queryset(self):
        """
        Get the filtered queryset based on the request parameters.

        Returns:
            queryset: Filtered queryset.
        """
        queryset = super().get_queryset()
        filter_params = self.request.GET.dict()
        exclude_ids = filter_params.pop("exclude", None)
        if exclude_ids:
            queryset = queryset.exclude(id__in=exclude_ids.split(","))
        return queryset


class AuthoritiesViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Authorities that are the owners of the categories.
    """

    queryset = Authorities.objects.all()
    permission_classes = [AllowAny]  # Allow access to anyone to view
    serializer_class = AuthoritySerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["name"]
    filterset_class = AuthorityFilter

    def get_permissions(self):
        """
        Get the permissions required for the current action.

        Returns:
            list: List of permission classes.
        """
        if self.action in ["create", "destroy", "update"]:
            return [
                IsAdminUser()
            ]  # Only allow access to admin users to create or delete
        return super().get_permissions()

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.native:
            return Response(
                {
                    "message": RESPONSE_MESSAGES["CANNOT_DELETE_NATIVE_AUTHORITY"][
                        "message"
                    ]
                },
                status=RESPONSE_MESSAGES["CANNOT_DELETE_NATIVE_AUTHORITY"]["code"],
            )
        return super().destroy(request, *args, **kwargs)

    def create(self, request, *args, **kwargs):
        csv_base64 = request.data["csv_base64"]
        exist = Authorities.objects.filter(name=request.data["name"])
        if exist:
            return Response(
                {"message": RESPONSE_MESSAGES["AUTHORITY_ALREADY_EXISTS"]["message"]},
                status=RESPONSE_MESSAGES["AUTHORITY_ALREADY_EXISTS"]["code"],
            )

        try:
            csv_data_list = self.process_csv_data(csv_base64)

            response = has_invalid_relation(csv_data_list)

            if response["code"] != 200:
                return Response(
                    {"message": response["message"]},
                    status=response["code"],
                )

            response = super().create(request, *args, **kwargs)
            create_categories(request.data["name"], csv_data_list)
            if response.status_code > 299:
                raise Exception("Error creating category")

            return response

        except Exception as e:
            print(e)
            Authorities.objects.filter(name=request.data["name"]).delete()

            return Response(
                {"message": RESPONSE_MESSAGES["INVALID_CSV_FORMAT"]["message"]},
                status=RESPONSE_MESSAGES["INVALID_CSV_FORMAT"]["code"],
            )

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        csv_base64 = request.data.get("csv_base64")
        if instance.native and request.data.get("name") != instance.name:
            return Response(
                {
                    "message": RESPONSE_MESSAGES["CANNOT_MODIFY_NATIVE_AUTHORITY_NAME"][
                        "message"
                    ]
                },
                status=RESPONSE_MESSAGES["CANNOT_MODIFY_NATIVE_AUTHORITY_NAME"]["code"],
            )
        if csv_base64:
            try:
                name = request.data.get("name", instance.name)
                csv_data_list = self.process_csv_data(csv_base64)

                response = has_invalid_relation(csv_data_list)

                if response["code"] != 200:
                    return Response(
                        {"message": response["message"]},
                        status=response["code"],
                    )
                create_categories(name, csv_data_list)

            except (TypeError, ValueError, UnicodeDecodeError) as e:
                print(e)
                return Response(
                    {"message": RESPONSE_MESSAGES["INVALID_CSV_FORMAT"]["message"]},
                    status=RESPONSE_MESSAGES["INVALID_CSV_FORMAT"]["code"],
                )
        return super().update(request, *args, **kwargs)

    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        csv_base64 = request.data.get("csv_base64")
        if instance.native and request.data.get("name") != instance.name:
            return Response(
                {
                    "message": RESPONSE_MESSAGES["CANNOT_MODIFY_NATIVE_AUTHORITY_NAME"][
                        "message"
                    ]
                },
                status=RESPONSE_MESSAGES["CANNOT_MODIFY_NATIVE_AUTHORITY_NAME"]["code"],
            )

        if csv_base64:
            try:
                csv_data_list = self.process_csv_data(csv_base64)

                response = has_invalid_relation(csv_data_list)

                if response["code"] != 200:
                    return Response(
                        {"message": response["message"]},
                        status=response["code"],
                    )

                create_categories(request.data["name"], csv_data_list)
            except Exception as e:
                print(e)
                return Response(
                    {"message": RESPONSE_MESSAGES["INVALID_CSV_FORMAT"]["message"]},
                    status=RESPONSE_MESSAGES["INVALID_CSV_FORMAT"]["code"],
                )
        return super().partial_update(request, *args, **kwargs)

    def get_queryset(self):
        queryset = super().get_queryset()

        # Get the value of the exclude_counts parameter from the URL
        exclude_counts = self.request.query_params.get("exclude_counts", False)
        # Pass the value of the parameter to the serializer
        self.serializer_class.exclude_counts = exclude_counts

        return queryset

    def process_csv_data(self, csv_base64):
        csv_data = base64.b64decode(csv_base64).decode("utf-8")
        csv_reader = csv.DictReader(csv_data.splitlines())
        return list(csv_reader)


class TextClassificationViewSet(viewsets.ViewSet):
    """
    ViewSet for text classification.

    Attributes:
        serializer_class (Serializer): Serializer class for text classification.
    """

    serializer_class = TextClassificationSerializer
    permission_classes = [IsAuthenticated]  # Allow access to anyone to view

    def create(self, request):
        """
        Create a new text classification.

        Args:
            request (Request): The request object.

        Returns:
            Response: The response object containing the predicted labels.
        """
        title = request.data.get("title")
        summary = request.data.get("summary")
        authority_id = request.data.get("authority_id")
        model_path = os.path.join(BASE_DIR, "trained_model")
        model_checkpoint = os.path.join(model_path, f"{authority_id}", "model.ckpt")

        if not os.path.exists(model_path) or not os.path.exists(model_checkpoint):
            return Response(
                {"message": RESPONSE_MESSAGES["TRAINING_MODEL_NOT_EXIST"]["message"]},
                status=RESPONSE_MESSAGES["TRAINING_MODEL_NOT_EXIST"]["code"],
            )

        # Check if the lock exists
        if authority_id not in settings.CLASSIFIERS_LOCKS:
            settings.CLASSIFIERS_LOCKS[authority_id] = Lock()

        with settings.CLASSIFIERS_LOCKS[authority_id]:
            # Check if the text_classifier exists
            authority = Authorities.objects.filter(id=authority_id).first()

            if not authority.last_training_date:
                return Response(
                    {
                        "message": RESPONSE_MESSAGES["TRAINING_MODEL_NOT_EXIST"][
                            "message"
                        ]
                    },
                    status=RESPONSE_MESSAGES["TRAINING_MODEL_NOT_EXIST"]["code"],
                )

            if authority_id not in settings.TEXT_CLASSIFIERS:
                text_classifier = settings.TEXT_CLASSIFIERS[
                    authority_id
                ] = TextClassifier(
                    authority_id=authority_id, loaded_at=datetime.now(timezone.utc)
                )
            else:
                text_classifier = settings.TEXT_CLASSIFIERS[authority_id]

            # Check if the classifier needs to be reloaded
            if text_classifier.loaded_at < authority.last_training_date:
                text_classifier = TextClassifier(
                    authority_id=authority_id, loaded_at=datetime.now(timezone.utc)
                )
                settings.TEXT_CLASSIFIERS[authority_id] = text_classifier

            try:
                # Translate the title to English
                translator = GoogleTranslator()
                predicted_labels = text_classifier.classify_text(
                    translator.translate(f"{title}: {summary}", dest="en").text
                )
                serializer = CategoriesSerializer(predicted_labels, many=True)
                serialized_data = serializer.data

                return Response(serialized_data)
            except Exception as e:
                return Response(
                    {
                        "message": RESPONSE_MESSAGES["TRAINING_MODEL_NOT_EXIST"][
                            "message"
                        ]
                    },
                    status=RESPONSE_MESSAGES["TRAINING_MODEL_NOT_EXIST"]["code"],
                )


class TrainAuthorityViewSet(viewsets.ViewSet):
    """
    Syncronize starter.
    """

    serializer_class = TrainAuthoritySerializer
    permission_classes = [IsAdminUser]

    def create(self, request):
        serializer = TrainAuthoritySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        authorities = serializer.validated_data["authorities"]
        authorities_ids = [authority.id for authority in authorities]
        authorities_without_pid = Authorities.objects.filter(
            id__in=authorities_ids, pid=0
        )
        authorities_without_pid.update(status="TRAINING")

        authorities_ids_str = " ".join(str(id) for id in authorities_ids)

        subprocess.Popen(
            [
                "python",
                "./manage.py",
                "categories_model_train",
                "--authorities",
                authorities_ids_str,
            ]
        )
        return Response(
            {"message": RESPONSE_MESSAGES["ERROR_TRAINING"]["message"]},
            status=RESPONSE_MESSAGES["ERROR_TRAINING"]["code"],
        )


class GetAuthorityCategoriesViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Syncronize starter.
    """

    queryset = Categories.objects.filter(deprecated=False)

    permission_classes = [IsAdminUser]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_class = GetAuthorityCategoriesFilter
    serializer_class = TrainAuthoritySerializer

    http_method_names = ["get"]

    def list(self, request):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            # Check if the filter is provided in the request
            if "authorities" not in request.query_params:
                return Response(
                    {"message": "The 'authorities' parameter is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Apply the filter
            queryset = self.filter_queryset(self.get_queryset())

            # Annotate the queryset with translation using a subquery
            subquery = Translations.objects.filter(
                category_id=OuterRef("id"),
                language="es",
            ).values("name")[:1]
            queryset = queryset.annotate(
                translation=Subquery(subquery, output_field=CharField()),
            )

            # Create a CSV string
            csv_data = io.StringIO()
            csv_writer = csv.writer(csv_data)
            csv_writer.writerow(["id", "name", "translation", "parent_id"])
            for category in queryset:
                csv_writer.writerow(
                    [
                        category.id,
                        category.name,
                        category.translation or "",
                        category.parent_id or "",
                    ]
                )

            # Convert the CSV string to base64
            csv_bytes = csv_data.getvalue().encode("utf-8")
            base64_csv = base64.b64encode(csv_bytes).decode("utf-8")
            return Response(
                {"csv_data": base64_csv},
                status=RESPONSE_MESSAGES["OK"]["code"],
            )

        except Exception as e:
            print(e)
            return Response(
                {"message": RESPONSE_MESSAGES["CSV_CREATION_FAILED"]["message"]},
                status=RESPONSE_MESSAGES["CSV_CREATION_FAILED"]["code"],
            )
