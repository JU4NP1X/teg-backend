"""
Datasets syncronizer
"""
import os
import signal
import sys
from django.core.management.base import BaseCommand
from django.db.models import Count
from tqdm import tqdm
from categories.models import Authorities, Categories
from ...models import DatasetsUniversity
from ...sync import OneSearchScraper, CreateMissingTranslations, GoogleScholarScraper


class Command(BaseCommand):
    """
    Datasets syncronizer command
    """

    help = "This is the synchronizer of the dataset for the categories model."

    def add_arguments(self, parser):
        """
        Add command line arguments.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
        """
        parser.add_argument(
            "--universities", nargs="*", type=str, help="List of universities"
        )
        parser.add_argument(
            "--authorities", nargs="*", type=int, help="List of athorities"
        )
        parser.add_argument(
            "--reset", action="store_true", help="Reset searched_for_datasets to False"
        )

    def handle(self, *args, **options):
        """
        Handle the command.

        Args:
            *args: Variable length argument list.
            **options: Keyword arguments.
        """

        def signal_handler(signal, frame):
            print("\nSynchronization interrupted by user.")
            # Aquí puedes agregar el código para manejar la interrupción o cerrar adecuadamente
            Authorities.objects.filter(id=authority.id).update(
                status="COMPLETE", percentage=0, pid=0
            )
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        pid = os.getpid()

        universities = options["universities"]
        if not universities:
            universities = (
                DatasetsUniversity.objects.filter(active=True)
                .exclude(name="google_scholar")
                .exclude(name="uc_university")
                .values_list("name", flat=True)
            )

        authorities = options["authorities"]
        if not authorities:
            authorities = Authorities.objects.filter(auto_sync=True)
        else:
            authorities = Authorities.objects.filter(id__in=authorities)

        for authority in tqdm(authorities):
            if options["reset"]:
                Categories.objects.filter(authority__id=authority.id).update(
                    searched_for_datasets=False
                )
            if authority.pid != 0:
                continue
            Authorities.objects.filter(id=authority.id).update(
                status="GETTING_DATA", percentage=0, pid=pid
            )

            categories = (
                Categories.objects.filter(deprecated=False, searched_for_datasets=False)
                .annotate(data_count=Count("datasets"))
                .filter(data_count__lt=10, authority__id=authority.id)
            )
            total_categories = len(categories)
            categories_progress = tqdm(categories)
            progress_counter = 0

            for category in categories_progress:
                progress_counter += 1
                Authorities.objects.filter(id=authority.id).update(
                    status="GETTING_DATA",
                    percentage=(progress_counter / total_categories) * 100 * 0.8,
                    pid=pid,
                )
                categories_progress.set_description(
                    f"Scraping category '{category.name}'"
                )
                scraper = OneSearchScraper(category, universities)
                scraper.scrape()
                Categories.objects.filter(id=category.id).update(
                    searched_for_datasets=True
                )

            categories = (
                Categories.objects.filter(deprecated=False, searched_for_datasets=True)
                .annotate(data_count=Count("datasets"))
                .filter(data_count__lt=10, authority__id=authority.id)
            )

            total_categories_google = len(categories)
            categories_progress = tqdm(categories)
            progress_counter_google = 0

            scraper = GoogleScholarScraper()
            for category in categories_progress:
                progress_counter_google += 1
                Authorities.objects.filter(id=authority.id).update(
                    status="GETTING_DATA",
                    percentage=80
                    + (progress_counter_google / total_categories_google) * 20,
                    pid=pid,
                )

                categories_progress.set_description(
                    f"Scraping category '{category.name}'"
                )
                scraper.search_and_save_datasets(category)

            CreateMissingTranslations.pass_english_text()
            CreateMissingTranslations.translate_all()

            Authorities.objects.filter(id=authority.id).update(
                status="COMPLETE", percentage=0, pid=0
            )

        print("Synchronization completed successfully.")
