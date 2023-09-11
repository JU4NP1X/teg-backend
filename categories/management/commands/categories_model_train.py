import os
from django.core.management.base import BaseCommand
from ...neural_network.roberta_classifier import Classifier
from categories.models import Authorities

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


class Command(BaseCommand):
    help = "Trains a text classifier and saves the model"

    def add_arguments(self, parser):
        parser.add_argument(
            "--from_pretrained",
            action="store_true",
            help="Load model from pretrained weights",
        )
        parser.add_argument(
            "--authorities",
            nargs="+",
            type=int,
            help="List of authority ids",
        )

    def handle(self, *args, **kwargs):
        authority_ids = kwargs.get("authorities", [])
        from_checkpoint = kwargs.get("from_pretrained", False)
        best_model_checkpoint = None
        best_model_params = None
        pid = os.getpid()

        if from_checkpoint:
            checkpoint_path = os.path.join(BASE_DIR, "lightning_logs/version_0")
            best_model_checkpoint = f"{checkpoint_path}/checkpoints/model.ckpt"
            best_model_params = f"{checkpoint_path}/hparams.yaml"

        for authority_id in authority_ids:
            authority = Authorities.objects.get(id=authority_id)
            if authority.pid != 0:
                self.stdout.write(
                    self.style.WARNING(
                        f"Authority {authority_id} is already training or getting data. Skipping..."
                    )
                )
                continue

            authority.status = "TRAINING"
            authority.pid = pid
            authority.save()
            text_classifier = Classifier(authority_id, False)
            text_classifier.train(best_model_checkpoint, best_model_params)
            text_classifier.save_categories()
