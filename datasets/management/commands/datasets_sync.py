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
from ...sync import OneSearchScraper


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
            universities = DatasetsUniversity.objects.filter(active=True).values_list(
                "name", flat=True
            )

        authorities = options["authorities"]
        if not authorities:
            authorities = Authorities.objects.filter(active=True)
        else:
            authorities = Authorities.objects.filter(active=True, id__in=authorities)
        if options["reset"]:
            Categories.objects.update(searched_for_datasets=False)
        for authority in tqdm(authorities):
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

            for categorie in categories_progress:
                progress_counter += 1
                Authorities.objects.filter(id=authority.id).update(
                    status="GETTING_DATA",
                    percentage=(progress_counter / total_categories) * 100,
                    pid=pid,
                )
                categories_progress.set_description(
                    f"Scraping category '{categorie.name}'"
                )
                scraper = OneSearchScraper(categorie, universities)
                scraper.scrape()
                Categories.objects.filter(id=categorie.id).update(
                    searched_for_datasets=True
                )

            OneSearchScraper.pass_english_text()
            OneSearchScraper.create_missing_translations()
            Authorities.objects.filter(id=authority.id).update(
                status="COMPLETE", percentage=0, pid=0
            )

        print("Synchronization completed successfully.")
