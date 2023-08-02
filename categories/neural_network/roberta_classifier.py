from transformers import AutoTokenizer
from .data_processer import Data_Processer
from .data_module import Data_Module
from .model import Categories_Classifier
import pytorch_lightning as pl
import os
from ..models import Categories
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil
import torch
import pytorch_lightning.loggers as logger

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


def create_pretrained_copy(tokenizer_path, tokenizer_name):
    if not os.path.exists(tokenizer_path):
        model = AutoTokenizer.from_pretrained(tokenizer_name)
        model.save_pretrained(tokenizer_path)


class Classifier:
    def __init__(self, trained=True):
        self.best_model_path = os.path.join(BASE_DIR, "trained_model")
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))
        self.n_epochs = int(os.environ.get("CATEGORIES_EPOCHS", 20))
        self.learning_rate = float(os.environ.get("CATEGORIES_LEARNING_RATE", 1e-07))
        self.best_model_checkpoint = f"{self.best_model_path}/model.ckpt"
        self.model_name = "distilroberta-base"
        self.tokenizer_name = "roberta-base"

        if trained:
            self.model_name = self.best_model_checkpoint

        tokenizer_path = os.path.join(BASE_DIR, self.tokenizer_name)
        create_pretrained_copy(tokenizer_path, self.tokenizer_name)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.df = Data_Processer()

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
        train_df, val_df = self.df.preprocess_data()
        categories_names = [label_name for _, label_name in self.categories]

        data_module = Data_Module(
            train_data=train_df,
            val_data=val_df,
            attributes=categories_names,
            max_length=self.max_len,
            batch_size=self.batch_size,
        )
        data_module.setup()
        self.config["train_size"] = len(data_module.train_dataloader())
        if checkpoint_path is not None and params_path is not None:
            self.model = Categories_Classifier.load_from_checkpoint(
                checkpoint_path, hparams_file=params_path, pos_weights=self.df.weights, learning_rate=self.learning_rate
            )
        else:
            self.model = Categories_Classifier(self.config, self.df.weights)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            mode="min",
            monitor="validation_loss",
            filename="model",
        )

        trainer = pl.Trainer(
            max_epochs=self.config["n_epochs"],
            num_sanity_val_steps=1,
            callbacks=[checkpoint_callback],
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

    def save_categories(self):
        Categories.objects.all().update(label_index=None)
        index = 0
        for label_id, _ in self.categories:
            category = Categories.objects.get(id=label_id)
            category.label_index = index
            category.save()
            index += 1
