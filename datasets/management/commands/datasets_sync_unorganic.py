from django.core.management.base import BaseCommand
import os
from categories.models import Categories
from ...sync import DatasetsScraper
from ...models import Datasets


class Command(BaseCommand):
    help = "This is the artificial dataset creation for the categories model"

    def handle(self, *args, **options):
        print(
                'Give me a json object with a resume of an science article with the next subject: Abortion. This is the structure of the json  {"title": "", "Description": "" }'
        )