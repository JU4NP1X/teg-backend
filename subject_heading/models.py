from django.db import models


class Subject_Headers(models.Model):
    name = models.CharField(max_length=200, unique=True)
    deprecated = models.BooleanField(default=False)
    searched_for_datasets = models.BooleanField(default=False)

    def __str__(self):
        return self.name


class Translations(models.Model):
    subject_header = models.ForeignKey(
        Subject_Headers, related_name="translations", on_delete=models.CASCADE
    )
    language = models.CharField(max_length=2)
    name = models.CharField(max_length=200)

    class Meta:
        unique_together = ("language", "name")

    def __str__(self):
        return self.name
