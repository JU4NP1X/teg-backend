from django.core.management.base import BaseCommand
from itertools import islice
from tqdm import tqdm
from thesaurus.models import Thesaurus
from ...sync import DatasetsScraper
from ...models import Datasets


class Command(BaseCommand):
    help = "This is the sincronizer of the dataset for the thesaurus model"

    def handle(self, *args, **options):
        chunk_size = 100

        thesaurus_objects = Thesaurus.objects.filter(
            searched_for_datasets=False
        ).iterator()

        while True:
            chunk = list(islice(thesaurus_objects, chunk_size))
            if not chunk:
                break

            # Itera sobre los objetos y haz print del nombre de cada uno
            for thesaurus_object in tqdm(chunk, desc="Scraping thesaurus objects"):
                scraper = DatasetsScraper(thesaurus_object.name)
                scraper.scrape()

                # Establece el valor de searched_for_datasets a True
                thesaurus_object.searched_for_datasets = True
                thesaurus_object.save()
