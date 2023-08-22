"""
Datasets syncronizer
"""
from django.core.management.base import BaseCommand
from django.db.models import Count
from tqdm import tqdm
from categories.models import Categories
from ...models import DatasetsUniversity
from ...sync import DatasetsScraper


class Command(BaseCommand):
    """
    Datasets syncronizer command
    """

    help = "This is the synchronizer of the dataset for the categories model."

    def add_arguments(self, parser):
        """
        Add command line arguments.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
        """
        parser.add_argument(
            "--universities", nargs="*", type=str, help="List of universities"
        )
        parser.add_argument(
            "--reset", action="store_true", help="Reset searched_for_datasets to False"
        )

    def handle(self, *args, **options):
        """
        Handle the command.

        Args:
            *args: Variable length argument list.
            **options: Keyword arguments.
        """
        universities = options["universities"]
        if not universities:
            universities = DatasetsUniversity.objects.filter(active=True).values_list(
                "name", flat=True
            )

        if options["reset"]:
            Categories.objects.update(searched_for_datasets=False)

        # Retrieve the categories objects that have not been searched for datasets or have less than 10 examples
        categories = (
            Categories.objects.filter(deprecated=False, searched_for_datasets=False)
            .annotate(cuenta=Count("datasets"))
            .filter(cuenta__lt=10)
        )

        categories_progress = tqdm(categories)
        for categorie in categories_progress:
            categories_progress.set_description(f"Scraping category '{categorie.name}'")
            scraper = DatasetsScraper(categorie, universities)
            scraper.scrape()
            categorie.searched_for_datasets = True
            categorie.save()

        DatasetsScraper.pass_english_text()
        DatasetsScraper.create_missing_translations()
