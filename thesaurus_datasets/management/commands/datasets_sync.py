from django.core.management.base import BaseCommand
from itertools import islice
from django.db.models import Count
from tqdm import tqdm
from thesaurus.models import Thesaurus
from ...models import Datasets_University
from ...sync import DatasetsScraper


class Command(BaseCommand):
    help = "This is the synchronizer of the dataset for the thesaurus model."

    def add_arguments(self, parser):
        parser.add_argument(
            "universities", nargs="*", type=str, help="List of universities"
        )

    def handle(self, *args, **options):
        universities = options["universities"]
        if not universities:
            universities = Datasets_University.objects.values_list("name", flat=True)

        chunk_size = 100

        # Retrieve the thesaurus objects that have not been searched for datasets
        thesaurus_objects = (
            Thesaurus.objects.filter(searched_for_datasets=False)
            .annotate(num_datasets=Count("datasets"))
            .order_by("num_datasets", "id")[:500]
            .iterator()
        )
        while True:
            chunk = list(islice(thesaurus_objects, chunk_size))
            if not chunk:
                # All objects have been searched, set searched_for_datasets to False and retrieve them again
                Thesaurus.objects.update(searched_for_datasets=False)
                thesaurus_objects = (
                    Thesaurus.objects.filter(searched_for_datasets=False)
                    .annotate(num_datasets=Count("datasets"))
                    .order_by("num_datasets", "id")[:500]
                    .iterator()
                )
                # Check if there are objects left to search again
                if not thesaurus_objects:
                    break

            # Iterate over the objects and print the name of each one
            for thesaurus_object in tqdm(chunk, desc="Scraping thesaurus objects"):
                scraper = DatasetsScraper(thesaurus_object.name, universities)
                scraper.scrape()

                # Set the value of searched_for_datasets to True
                thesaurus_object.searched_for_datasets = True
                thesaurus_object.save()

        DatasetsScraper.create_missing_translations()
