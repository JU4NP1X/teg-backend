from django.db import models
from categories.models import Categories

class Datasets(models.Model):
    paper_name = models.CharField(max_length=255, unique=True)
    summary = models.TextField()
    categories = models.ManyToManyField(Categories)
    university = models.ForeignKey("Datasets_University", on_delete=models.CASCADE)

    def __str__(self):
        return self.paper_name


class Datasets_University(models.Model):
    name = models.CharField(max_length=100)
    url = models.URLField()
    vid = models.CharField(max_length=30, null=True,)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name

class Datasets_English_Translations(models.Model):
    dataset = models.OneToOneField(Datasets, on_delete=models.CASCADE, primary_key=True)
    paper_name = models.CharField(max_length=500)
    summary = models.TextField()

    def __str__(self):
        return self.dataset.paper_name