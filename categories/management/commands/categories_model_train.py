from django.core.management.base import BaseCommand
from ...neural_network.roberta_classifier import Classifier
import os

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


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
        best_model_checkpoint = None
        best_model_params = None

        if from_checkpoint:
            checkpoint_path = os.path.join(BASE_DIR, "lightning_logs/version_0")
            best_model_checkpoint = f"{checkpoint_path}/checkpoints/model.ckpt"
            best_model_params = f"{checkpoint_path}/hparams.yaml"

        text_classifier.train(best_model_checkpoint, best_model_params)
        text_classifier.save_categories()
