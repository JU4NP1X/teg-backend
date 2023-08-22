"""
Django management command for exporting a DataFrame to a CSV file.
"""
import os
from django.core.management.base import BaseCommand
from categories.neural_network.data_processer import DataProcesser

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


class Command(BaseCommand):
    """
    Django management command for exporting a DataFrame to a CSV file.
    """

    help = "Export a DataFrame to a CSV file"

    def handle(self, *args, **options):
        """
        Handle the command.

        Args:
            *args: Variable length argument list.
            **options: Keyword arguments.
        """
        # Specify the path and name of the CSV file
        preprocesser = (
            DataProcesser()
        )  # Replace "DataProcesser" with the name of your model
        train_set, test_set = preprocesser.preprocess_data()

        train_set.to_csv(BASE_DIR + "/train_set.csv", index=False)
        test_set.to_csv(BASE_DIR + "/test_set.csv", index=False)

        self.stdout.write(
            self.style.SUCCESS("The DataFrame has been exported to CSV successfully.")
        )
