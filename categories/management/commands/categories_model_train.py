from django.core.management.base import BaseCommand
from ...neural_network.roberta_classifier import Classifier

class Command(BaseCommand):
    help = 'Trains a text classifier and saves the model'

    def handle(self, *args, **kwargs):
        text_classifier = Classifier(False)
        text_classifier.train()
        text_classifier.save_model()