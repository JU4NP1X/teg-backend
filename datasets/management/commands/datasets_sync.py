from django.core.management.base import BaseCommand
from tqdm import tqdm
from categories.models import Categories
from ...models import Datasets_University
from ...sync import DatasetsScraper
from django.db.models import Q


class Command(BaseCommand):
    help = "This is the synchronizer of the dataset for the categories model."

    def add_arguments(self, parser):
        parser.add_argument(
            "universities", nargs="*", type=str, help="List of universities"
        )

    def handle(self, *args, **options):
        universities = options["universities"]
        if not universities:
            universities = Datasets_University.objects.filter(active=True).values_list(
                "name", flat=True
            )

        # Retrieve the categories objects that have not been searched for datasets
        categories_objects = Categories.objects.filter(
            ~Q(datasets__isnull=False),
            ~Q(related_categories__datasets__isnull=False)
        )
        # Iterate over the objects and print the name of each one
        for categories_object in tqdm(
            categories_objects, desc="Scraping categories objects"
        ):
            scraper = DatasetsScraper(categories_object.name, universities)
            scraper.scrape()

            # Set the value of searched_for_datasets to True
            categories_object.searched_for_datasets = True
            categories_object.save()

        DatasetsScraper.pass_english_text()
        DatasetsScraper.create_missing_translations()
