from django.core.management.base import BaseCommand
from ...sync import scrap_unesco


class Command(BaseCommand):
    help = "This is the sincronizer for the categories model"

    def handle(self, *args, **options):
        # Translations.objects.all().delete()
        scrap_unesco()
