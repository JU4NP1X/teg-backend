from django.db import models


class Authorities(models.Model):
    name = models.CharField(max_length=200, unique=True)

    def __str__(self):
        return self.name


class Categories(models.Model):
    name = models.CharField(max_length=200, unique=True)
    link = models.CharField(max_length=200)
    parent_category = models.ForeignKey(
        "self", on_delete=models.CASCADE, null=True, blank=True
    )
    deprecated = models.BooleanField(default=False)
    related_categories = models.ManyToManyField("self", blank=True)
    searched_for_datasets = models.BooleanField(default=False)
    authority = models.ForeignKey(Authorities, on_delete=models.CASCADE, default=1)

    def __str__(self):
        return self.name


class Translations(models.Model):
    categories = models.ForeignKey(
        Categories, related_name="translations", on_delete=models.CASCADE
    )
    language = models.CharField(max_length=2)
    name = models.CharField(max_length=200)

    class Meta:
        unique_together = ("language", "name")

    def __str__(self):
        return self.name
