from django.core.management.base import BaseCommand
from tqdm import tqdm
from categories.models import Categories
from ...models import Datasets_University
from ...sync import DatasetsScraper
from django.db.models import Count


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

        # Retrieve the categories objects that have not been searched for datasets or have less than 10 examples
        categories = (
            Categories.objects.filter(deprecated=False, searched_for_datasets= False)
            .annotate(cuenta=Count("datasets") + Count("related_categories__datasets"))
            .filter(cuenta__lt=10)
        )

        categories_progress = tqdm(categories)
        for categorie in categories_progress:
            categories_progress.set_description(f"Scraping category '{categorie.name}'")
            scraper = DatasetsScraper(categorie, universities)
            scraper.scrape()

            categorie.save()

        # Close the progress bar

        DatasetsScraper.pass_english_text()
        DatasetsScraper.create_missing_translations()
