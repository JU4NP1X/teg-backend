from django.core.management.base import BaseCommand
from itertools import islice
from tqdm import tqdm
from .models import Thesaurus
from .sync import DatasetsScraper

class Command(BaseCommand):
    help = 'This is the sincronizer of the dataset for the thesaurus model'

    def handle(self, *args, **options):
        scraper = DatasetsScraper()
        chunk_size = 100
        thesaurus_objects = Thesaurus.objects.all().iterator()

        while True:
            chunk = list(islice(thesaurus_objects, chunk_size))
            if not chunk:
                break

            # Itera sobre los objetos y haz print del nombre de cada uno
            for thesaurus_object in tqdm(chunk, desc="Scraping thesaurus objects"):
                scraper.scrape(thesaurus_object.name)