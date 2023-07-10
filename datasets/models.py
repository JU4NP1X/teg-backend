from django.db import models
from thesaurus.models import Thesaurus


class Datasets(models.Model):
    paper_name = models.CharField(max_length=500, unique=True)
    summary = models.TextField()
    categories = models.ManyToManyField(Thesaurus)

    def __str__(self):
        return self.paper_name
