from django.db import models
from categories.models import Categories


class Datasets(models.Model):
    """
    Model representing a dataset.

    Attributes:
        paper_name (str): The name of the paper.
        summary (str): The summary of the dataset.
        categories (ManyToManyField): The categories associated with the dataset.
        university (ForeignKey): The university associated with the dataset.
    """

    paper_name = models.CharField(max_length=255, unique=True)
    summary = models.TextField()
    categories = models.ManyToManyField(Categories)
    university = models.ForeignKey("DatasetsUniversity", on_delete=models.CASCADE)

    def __str__(self):
        """
        Returns a string representation of the dataset.
        """
        return str(self.paper_name)

    class Meta:
        db_table = "datasets"


class DatasetsUniversity(models.Model):
    """
    Model representing a university associated with a dataset.

    Attributes:
        name (str): The name of the university.
        url (str): The URL of the university.
        vid (str): The VID of the university.
        active (bool): Indicates if the university is active or not.
    """

    name = models.CharField(max_length=100)
    url = models.URLField()
    vid = models.CharField(
        max_length=30,
        null=True,
    )
    active = models.BooleanField(default=True)

    def __str__(self):
        """
        Returns a string representation of the university.
        """
        return str(self.name)

    class Meta:
        db_table = "datasets_university"


class DatasetsEnglishTranslations(models.Model):
    """
    Model representing English translations of datasets.

    Attributes:
        dataset (OneToOneField): The dataset associated with the translation.
        paper_name (str): The translated name of the paper.
        summary (str): The translated summary of the dataset.
    """

    dataset = models.OneToOneField(Datasets, on_delete=models.CASCADE, primary_key=True)
    paper_name = models.CharField(max_length=500)
    summary = models.TextField()

    def __str__(self):
        """
        Returns a string representation of the translated dataset.
        """
        return str(self.dataset.paper_name)

    class Meta:
        db_table = "datasets_english_translations"
