from django.core.management.base import BaseCommand
import pandas as pd
from categories.neural_network.data_processer import Data_Processer
import os

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


class Command(BaseCommand):
    help = "Exporta un DataFrame a un archivo CSV"

    def handle(self, *args, **options):
        # Obtén los datos que deseas exportar a CSV
        preprocesser = (
            Data_Processer()
        )  # Reemplaza "TuModelo" con el nombre de tu modelo
        train_set, test_set = preprocesser.preprocess_data()

        # Especifica la ruta y el nombre del archivo CSV
        train_set.to_csv(BASE_DIR + "/train_set.csv", index=False)
        test_set.to_csv(BASE_DIR + "/test_set.csv", index=False)

        self.stdout.write(
            self.style.SUCCESS(
                "El DataFrame se ha exportado exitosamente a un archivo CSV."
            )
        )
