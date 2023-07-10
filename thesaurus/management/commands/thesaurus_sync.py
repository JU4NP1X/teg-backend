from django.core.management.base import BaseCommand
from ...sync import start_scraping
from ...models import Translations

class Command(BaseCommand):
    help = "This is the sincronizer for the thesaurus model"

    def handle(self, *args, **options):
        #Translations.objects.all().delete()
        start_scraping()
        

