from django.db import models

class Thesaurus(models.Model):
    name = models.CharField(max_length=200, unique=True)
    link = models.CharField(max_length=200)
    parent_thesaurus = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    deprecated = models.BooleanField(default=False)
    related_thesauri = models.ManyToManyField('self', blank=True)

    def __str__(self):
        return self.name
class Translations(models.Model):
    thesaurus = models.ForeignKey(Thesaurus, related_name='translations', on_delete=models.CASCADE)
    language = models.CharField(max_length=200)
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name
