from django.core.management.base import BaseCommand, CommandError
from categories.neural_network import TextClassifier


class Command(BaseCommand):
    help = 'Trains a text classifier and saves the model'

    def handle(self, *args, **kwargs):
        text_classifier = TextClassifier()
        text_classifier.train()