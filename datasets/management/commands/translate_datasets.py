from django.core.management.base import BaseCommand
from itertools import islice
from django.db.models import Count
from tqdm import tqdm
from categories.models import Categories
from ...models import Datasets_University
from ...sync import DatasetsScraper


class Command(BaseCommand):
    help = "This is the translator of the categories."

    def handle(self, *args, **options):
        DatasetsScraper.pass_english_text()
        DatasetsScraper.create_missing_translations()
