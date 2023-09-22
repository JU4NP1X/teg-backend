from django.core.management.base import BaseCommand
from ...models import Authorities
from ...sync import start_scraping


class Command(BaseCommand):
    help = "This is the sincronizer for the categories model"

    def handle(self, *args, **options):
        start_scraping()
