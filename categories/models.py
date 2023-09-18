import gc
from django.db import models
from django.conf import settings
from mptt.models import MPTTModel, TreeForeignKey
from mptt.managers import TreeManager
from datetime import datetime, timezone
from db_mutex.db_mutex import db_mutex


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
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="NOT_TRAINED"
    )
    last_training_date = models.DateTimeField(null=True, blank=True)
    active = models.BooleanField(default=False)
    disabled = models.BooleanField(default=False)
    native = models.BooleanField(default=False)
    num_documents_classified = models.PositiveIntegerField(default=0)
    accuracy = models.PositiveIntegerField(default=0)

    def save(self, *args, **kwargs):
        from .neural_network.text_classifier import TextClassifier

        if self.pk:  # Check if the instance already exists in the database
            original_instance = Authorities.objects.get(pk=self.pk)
            if original_instance.active != self.active:  # Check if 'active' has changed
                with db_mutex(str(self.pk)):
                    if not self.active:
                        settings.TEXT_CLASSIFIERS[str(self.pk)] = None
                    else:
                        settings.TEXT_CLASSIFIERS[str(self.pk)] = TextClassifier(
                            authority_id=str(self.pk),
                            loaded_at=datetime.now(timezone.utc),
                        )
                    gc.collect()

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

    level = models.PositiveIntegerField(default=0)
    lft = models.PositiveIntegerField(default=0)
    rght = models.PositiveIntegerField(default=0)
    tree_id = models.PositiveIntegerField(default=0)
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
