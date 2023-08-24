from django.db import models
from webcolors import CSS3_NAMES_TO_HEX


class Authorities(models.Model):
    """
    Model representing authorities.
    """

    COLOR_CHOICES = [(value, key) for key, value in CSS3_NAMES_TO_HEX.items()]

    STATUS_CHOICES = [
        ("NOT_TRAINED", "Not Trained"),
        ("TRAINING", "Training"),
        ("COMPLETE", "Complete"),
        ("GETTING_DATA", "Getting Data"),
    ]

    name = models.CharField(max_length=200, unique=True)
    color = models.CharField(max_length=7, choices=COLOR_CHOICES, default="#000000")
    percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="NOT_TRAINED"
    )
    last_training_date = models.DateField(null=True, blank=True)
    active = models.BooleanField(default=True)
    native = models.BooleanField(default=False)

    def __str__(self):
        return str(self.name)


class Categories(models.Model):
    """
    Model representing categories.
    """

    name = models.CharField(max_length=200, unique=True)
    link = models.CharField(max_length=200)
    parent_category = models.ForeignKey(
        "self", on_delete=models.CASCADE, null=True, blank=True
    )
    label_index = models.BigIntegerField(
        null=True,
    )
    deprecated = models.BooleanField(default=False)
    related_categories = models.ManyToManyField("self", blank=True)
    searched_for_datasets = models.BooleanField(default=False)
    authority = models.ForeignKey(Authorities, on_delete=models.CASCADE, default=1)

    def __str__(self):
        return str(self.name)

    def children(self):
        """
        Returns the child categories of the current category.
        """
        return Categories.objects.filter(parent_category=self)

    def include_descendants(self):
        """
        Returns the descendants categories of the current category.
        """
        descendants = []
        for child in self.children():
            descendants.append(child.include_descendants())
        return {"name": self.name, "children": descendants}


class Translations(models.Model):
    """
    Model representing translations.
    """

    categories = models.ForeignKey(
        Categories, related_name="translations", on_delete=models.CASCADE
    )
    language = models.CharField(max_length=2)
    name = models.CharField(max_length=200)

    class Meta:
        unique_together = ("language", "name")

    def __str__(self):
        return str(self.name)
