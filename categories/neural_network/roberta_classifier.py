import os
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
from decimal import Decimal
from categories.models import Categories, Authorities
from categories.neural_network.data_processer import DataProcesser
from categories.neural_network.data_module import DataModule
from categories.neural_network.model import CategoriesClassifier


BASE_DIR = os.path.dirname(os.path.realpath(__name__))


def create_pretrained_copy(tokenizer_path, tokenizer_name):
    """
    Creates a pretrained copy of a tokenizer if it doesn't already exist.

    Args:
        tokenizer_path (str): The path to save the pretrained tokenizer.
        tokenizer_name (str): The name of the tokenizer.

    Returns:
        None
    """
    if not os.path.exists(tokenizer_path):
        model = AutoTokenizer.from_pretrained(tokenizer_name)
        model.save_pretrained(tokenizer_path)


class Classifier:
    """
    A class that represents a classifier for categorizing data.

    Attributes:
        best_model_path (str): The path to save the best model.
        max_len (int): The maximum length of input sequences.
        n_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for training.
        best_model_checkpoint (str): The path to save the best model checkpoint.
        model_name (str): The name of the model.
        tokenizer_name (str): The name of the tokenizer.
        tokenizer (AutoTokenizer): The pretrained tokenizer.
        df (DataProcesser): The data processer object.
        categories (list): The list of categories.
        batch_size (int): The batch size for training.
        config (dict): The configuration parameters for the model.
        logs_path (str): The path to save the logs.
    """

    def __init__(self, authority_id, trained=True):
        """
        Initializes a new instance of the Classifier class.

        Args:
            trained (bool): Whether to use a trained model or not.

        Returns:
            None
        """
        self.authority_id = authority_id
        self.best_model_path = (
            f"{os.path.join(BASE_DIR, 'trained_model')}/{self.authority_id }"
        )
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))
        self.n_epochs = int(os.environ.get("CATEGORIES_EPOCHS", 20))
        self.learning_rate = float(os.environ.get("CATEGORIES_LEARNING_RATE", 1e-07))
        self.best_model_checkpoint = f"{self.best_model_path}/model.ckpt"
        self.model_name = "distilroberta-base"
        self.tokenizer_name = "roberta-base"
        self.model = None

        if trained:
            self.model_name = self.best_model_checkpoint

        tokenizer_path = os.path.join(BASE_DIR, self.tokenizer_name)
        create_pretrained_copy(tokenizer_path, self.tokenizer_name)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.df = DataProcesser(self.authority_id)

        self.categories = self.df.get_categories(trained)
        categories_names = [label_name for _, label_name in self.categories]
        self.batch_size = int(os.environ.get("CATEGORIES_BATCH_SIZE", 5))

        self.config = {
            "model_name": self.model_name,
            "n_labels": len(categories_names),
            "batch_size": self.batch_size,
            "lr": self.learning_rate,
            "warmup": 0.2,
            "weight_decay": 0.001,
            "n_epochs": self.n_epochs,
        }

        self.logs_path = os.path.join(BASE_DIR, "lightning_logs")

    def train(self, checkpoint_path=None, params_path=None):
        """
        Trains the classifier model.

        Args:
            checkpoint_path (str): The path to the model checkpoint.
            params_path (str): The path to the model parameters.

        Returns:
            None
        """
        train_df, val_df = self.df.preprocess_data()
        categories_names = [label_name for _, label_name in self.categories]

        data_module = DataModule(
            train_data=train_df,
            val_data=val_df,
            attributes=categories_names,
            max_length=self.max_len,
            batch_size=self.batch_size,
        )
        data_module.setup()

        # Antes de iniciar el entrenamiento, establece el estado de la autoridad en "TRAINING" y el porcentaje en 0
        authority = Authorities.objects.get(id=self.authority_id)
        authority.status = "TRAINING"
        authority.percentage = 0
        authority.save()
        self.config["train_size"] = len(train_df)
        if checkpoint_path is not None and params_path is not None:
            self.model = CategoriesClassifier.load_from_checkpoint(
                checkpoint_path,
                hparams_file=params_path,
                pos_weights=self.df.weights,
                learning_rate=self.learning_rate,
            )
        else:
            self.model = CategoriesClassifier(self.config, self.df.weights)
        self.df = None
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            mode="min",
            monitor="validation_loss",
            filename="model",
        )

        trainer = pl.Trainer(
            max_epochs=self.config["n_epochs"],
            num_sanity_val_steps=1,
            callbacks=[
                checkpoint_callback,
                TrainingProgressCallback(self.authority_id),
            ],
        )

        trainer.fit(self.model, datamodule=data_module, ckpt_path=checkpoint_path)

        if not trainer.interrupted:
            # Save the best model
            best_checkpoint = checkpoint_callback.best_model_path
            if best_checkpoint:
                shutil.copy(best_checkpoint, self.best_model_checkpoint)
                shutil.copy(
                    f"{self.logs_path}/version_0/hparams.yaml",
                    f"{self.best_model_path}/hparams.yaml",
                )

            # Delete the logs
            logs_dir = trainer.logger.root_dir
            shutil.rmtree(logs_dir, ignore_errors=True)
            os.makedirs(logs_dir, exist_ok=True)

            # Cuando el entrenamiento haya terminado, establece el porcentaje en 0 y cambia el estado a "COMPLETE"
            authority.status = "COMPLETE"
            authority.percentage = 0
            authority.save()

    def save_categories(self):
        """
        Saves the categories.

        Returns:
            None
        """
        Categories.objects.all().update(label_index=None)
        index = 0
        for label_id, _ in self.categories:
            category = Categories.objects.get(id=label_id)
            category.label_index = index
            category.save()
            index += 1


class TrainingProgressCallback(pl.Callback):
    def __init__(self, authority_id):
        self.authority_id = authority_id

    def on_validation_epoch_end(self, trainer, pl_module):
        # Calcula el porcentaje de entrenamiento completado
        current_epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs
        percentage = (current_epoch + 1) / total_epochs * 100

        # Calcula la pérdida promedio en la validación
        avg_loss = trainer.callback_metrics["validation_loss"]
        theoretical_precision = Decimal(1 - avg_loss.item()) * 100
        # Calcula la precisión teórica en base a la pérdida
        # Actualiza el porcentaje en la base de datos
        authority = Authorities.objects.get(id=self.authority_id)
        authority.percentage = percentage
        authority.status = "TRAINING"
        authority.theoretical_precision = theoretical_precision
        authority.save()
