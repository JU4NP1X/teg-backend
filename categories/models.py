import gc
import time
from django.db import models
from django.conf import settings
from mptt.models import MPTTModel, TreeForeignKey
from mptt.managers import TreeManager
from datetime import datetime, timezone
from db_mutex.db_mutex import db_mutex
from django.db import connection
from django.core.validators import MaxValueValidator
from db_mutex import DBMutexError
from utils.response_messages import RESPONSE_MESSAGES


def execute_query(query):
    with connection.cursor() as cursor:
        cursor.execute(query)


def categories_tree_adjust():
    query = """
        UPDATE categories AS ca
        SET "level" = 0
        WHERE ca.parent_id is null
    """

    # Tree correction (there is a bug, that the move_to not update the tree of it chindrens)
    execute_query(query)

    query = """
        UPDATE categories AS ca
        SET tree_id = ca2.tree_id, "level" = ca2.level + 1
        FROM categories AS ca2
        WHERE ca2.tree_id <> ca.tree_id AND ca2.id = ca.parent_id
    """

    # Tree correction (there is a bug, that the move_to not update the tree of it chindrens)
    execute_query(query)

    query = """
        UPDATE categories AS ca
        SET "level" = 0
        WHERE ca.deprecated = TRUE
    """

    execute_query(query)


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
        relation_list = []

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
        relation_list.append(str(element_id))
        while current_parent != "":
            relation_list.append(str(current_parent))
            if current_parent == element_id:
                error = {}
                error["code"] = RESPONSE_MESSAGES["CIRCULAR_RELATIONSHIP"]["code"]
                error["message"] = (
                    RESPONSE_MESSAGES["CIRCULAR_RELATIONSHIP"]["message"]
                    + ": "
                    + ",".join(relation_list)
                )
                return error
            current_parent = parents.get(current_parent, "")

    return {"code": 200}


def create_categories(authority_name, data):
    """
    Crear un diccionario para almacenar los padres de cada elemento
    """

    authority = Authorities.objects.filter(name=authority_name).first()
    Categories.objects.filter(authority=authority).update(deprecated=True)

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
        update_categories_tree(category, parent)


def update_categories_tree(category, parent=None):
    try_again = True
    while try_again:
        try_again = False
        try:
            with db_mutex("tree_adjust"):
                free_tree_id = 1
                while Categories.objects.filter(tree_id=free_tree_id).count():
                    free_tree_id += 1

                children = Categories.objects.get(pk=category.id)

                if parent:
                    parent = Categories.objects.get(pk=parent.id)
                    if parent.name == children.name:
                        return

                if parent and (not children.parent or children.parent.id != parent.id):
                    children.tree_id = free_tree_id
                    children.lft = 0
                    children.rght = 0
                    children.move_to(parent, "last-child")
                    children.save()
                elif not parent:
                    children.tree_id = free_tree_id
                    children.level = 0
                    children.lft = 0
                    children.rght = 0
                    children.move_to(None, "last-child")
                    children.save()
                    print("hola", children)
                category = children

                categories_tree_adjust()
        except DBMutexError:
            time.sleep(0.2)
            try_again = True


class Authorities(models.Model):
    """
    Model representing authorities.
    """

    STATUS_CHOICES = [
        ("NOT_TRAINED", "Not Trained"),
        ("TRAINING", "Training"),
        ("COMPLETE", "Complete"),
        ("GETTING_DATA", "Getting Data"),
    ]

    name = models.CharField(max_length=200, unique=True)
    color = models.CharField(max_length=7, blank=True)
    pid = models.PositiveIntegerField(blank=True, default=0)
    percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    theoretical_precision = models.DecimalField(
        max_digits=5, decimal_places=2, default=0
    )
    practical_precision = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="NOT_TRAINED"
    )
    last_training_date = models.DateTimeField(null=True, blank=True)
    active = models.BooleanField(default=False)
    disabled = models.BooleanField(default=False)
    native = models.BooleanField(default=False)
    auto_sync = models.BooleanField(default=True)
    num_documents_classified = models.PositiveIntegerField(default=0)

    def save(self, *args, **kwargs):
        from .neural_network.text_classifier import TextClassifier

        if self.pk:  # Check if the instance already exists in the database
            original_instance = Authorities.objects.get(pk=self.pk)
            if original_instance.active != self.active:  # Check if 'active' has changed
                try_again = True
                while try_again:
                    try:
                        with db_mutex(str(self.pk)):
                            if not self.active:
                                settings.TEXT_CLASSIFIERS[str(self.pk)] = None
                            else:
                                settings.TEXT_CLASSIFIERS[
                                    str(self.pk)
                                ] = TextClassifier(
                                    authority_id=str(self.pk),
                                    loaded_at=datetime.now(timezone.utc),
                                )
                            gc.collect()
                    except DBMutexError:
                        try_again = True

        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.name)

    class Meta:
        db_table = "categories_authorities"


class Categories(MPTTModel):
    """
    Model representing categories.
    """

    name = models.CharField(max_length=200)
    authority = models.ForeignKey(Authorities, on_delete=models.CASCADE, default=1)
    link = models.CharField(max_length=200)
    parent = TreeForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="children",
        db_index=True,
    )
    label_index = models.BigIntegerField(null=True)
    deprecated = models.BooleanField(default=False)
    related_categories = models.ManyToManyField("self", blank=True)
    searched_for_datasets = models.BooleanField(default=False)

    level = models.PositiveBigIntegerField(default=0)
    lft = models.PositiveBigIntegerField(default=0)
    rght = models.PositiveBigIntegerField(default=0)
    tree_id = models.PositiveBigIntegerField(
        default=0, validators=[MaxValueValidator(99999999)]
    )
    objects = TreeManager()

    def __str__(self):
        return str(self.name)

    class MPTTMeta:
        order_insertion_by = ["id"]

    class Meta:
        db_table = "categories"
        unique_together = (("name", "authority"),)


class Translations(models.Model):
    """
    Model representing translations.
    """

    category = models.ForeignKey(
        Categories, related_name="translations", on_delete=models.CASCADE
    )
    language = models.CharField(max_length=2)
    name = models.CharField(max_length=200)

    class Meta:
        unique_together = ("language", "category")
        db_table = "categories_translations"

    def __str__(self):
        return str(self.name)
