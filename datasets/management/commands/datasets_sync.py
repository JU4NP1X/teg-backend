from django.core.management.base import BaseCommand
from itertools import islice
from tqdm import tqdm
from thesaurus.models import Thesaurus
from ...models import Datasets_University
from ...sync import DatasetsScraper


class Command(BaseCommand):
    help = "This is the synchronizer of the dataset for the thesaurus model."

    def add_arguments(self, parser):
        parser.add_argument(
            "universities", nargs="*", type=str, help="List of universities"
        )

    def handle(self, *args, **options):
        universities = options["universities"]
        if not universities:
            universities = Datasets_University.objects.values_list("name", flat=True)

        chunk_size = 100


        thesaurus_objects = Thesaurus.objects.filter(
            searched_for_datasets=False
        ).iterator()

        while True:
            chunk = list(islice(thesaurus_objects, chunk_size))
            if not chunk:
                # Todos los objetos han sido buscados, establece searched_for_datasets en False y vuelve a obtenerlos
                Thesaurus.objects.update(
                    searched_for_datasets=False
                )
                thesaurus_objects = Thesaurus.objects.filter(
                    searched_for_datasets=False
                ).iterator()
                # Verifica si hay objetos sin buscar de nuevo
                if not thesaurus_objects:
                    break

            # Itera sobre los objetos y haz print del nombre de cada uno
            for thesaurus_object in tqdm(chunk, desc="Scraping thesaurus objects"):
                scraper = DatasetsScraper(thesaurus_object.name, universities)
                scraper.scrape()

                # Establece el valor de searched_for_datasets a True
                thesaurus_object.searched_for_datasets = True
                thesaurus_object.save()

        DatasetsScraper.create_missing_translations()
