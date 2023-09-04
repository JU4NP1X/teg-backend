import csv
from django.core.management.base import BaseCommand
from categories.models import Categories
from datasets.models import Datasets


class Command(BaseCommand):
    help = "Importa datos de un archivo CSV y crea o actualiza los registros correspondientes en la base de datos"

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Ruta al archivo CSV")

    def handle(self, *args, **options):
        csv_file = options["csv_file"]

        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                paper_name = row["paper_name"]
                summary = row["summary"]
                category_name = row["name"]

                # Verificar si el paper_name ya existe en la base de datos
                dataset, created = Datasets.objects.get_or_create(
                    paper_name=paper_name, summary=summary
                )

                # Si el dataset es nuevo, buscar la categoría por su nombre
                category = Categories.objects.get(name=category_name, authority__id=1)
                dataset.categories.add(category)
                dataset.save()

        self.stdout.write(self.style.SUCCESS("Los datos se importaron correctamente"))
