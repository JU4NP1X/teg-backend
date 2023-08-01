from django.core.management.base import BaseCommand
from ...neural_network.roberta_classifier import Classifier


class Command(BaseCommand):
    help = "Trains a text classifier and saves the model"

    def add_arguments(self, parser):
        parser.add_argument(
            "--from_pretrained",
            action="store_true",
            help="Load model from pretrained weights",
        )

    def handle(self, *args, **kwargs):
        text_classifier = Classifier(False)
        from_checkpoint = kwargs.get("from_pretrained", False)
        text_classifier.train(from_checkpoint=from_checkpoint)
        text_classifier.save_model()
