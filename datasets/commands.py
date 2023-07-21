from django.core.management.base import BaseCommand
from itertools import islice
from tqdm import tqdm
from .models import Categories
from .sync import DatasetsScraper

class Command(BaseCommand):
    help = 'This is the sincronizer of the dataset for the categories model'

    def handle(self, *args, **options):
        scraper = DatasetsScraper()
        chunk_size = 100
        categories_objects = Categories.objects.all().iterator()

        while True:
            chunk = list(islice(categories_objects, chunk_size))
            if not chunk:
                break

            for categories_object in tqdm(chunk, desc="Scraping categories objects"):
                scraper.scrape(categories_object.name)