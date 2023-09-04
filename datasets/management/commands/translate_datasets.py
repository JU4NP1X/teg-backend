from django.core.management.base import BaseCommand
from ...sync import CreateMissingTranslations


class Command(BaseCommand):
    """
    Django management command for translating categories.

    This command translates the categories using the OneSearchScraper class.
    """

    help = "This is the translator of the categories."

    def handle(self, *args, **options):
        """
        Handle method for the management command.

        This method calls the pass_english_text() and create_missing_translations() methods of the OneSearchScraper class.
        """
        CreateMissingTranslations.pass_english_text()
        CreateMissingTranslations.translate_all()
