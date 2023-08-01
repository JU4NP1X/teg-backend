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


class Classifier:
    def __init__(self, trained=True):
        self.model_path = os.environ.get(
            "CATEGORIES_MODEL_PATH", "/home/juan/projects/teg/backend"
        )
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))
        self.n_epochs = int(os.environ.get("CATEGORIES_EPOCHS", 20))
        if trained:
            self.model_name = f"{self.model_path}/trained_model"
            self.tokenizer_name = f"{self.model_path}/trained_tokenizer"
        else:
            self.model_name = "distilroberta-base"
            self.tokenizer_name = "roberta-base"

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.df = Data_Processer()

        self.categories = self.df.get_categories(trained)
        categories_names = [label_name for _, label_name in self.categories]
        self.batch_size = int(os.environ.get("CATEGORIES_BATCH_SIZE", 5))

        self.config = {
            "model_name": self.model_name,
            "n_labels": len(categories_names),
            "batch_size": self.batch_size,
            "lr": 1.5e-6,
            "warmup": 0.2,
            "weight_decay": 0.001,
            "n_epochs": self.n_epochs,
        }
        self.model = Categories_Classifier(self.config)

        self.logs_path = os.environ.get(
            "CATEGOIRES_LOGS_PATH", "/home/juan/projects/teg/backend/lightning_logs"
        )

    def train(self, from_checkpoint=False):
        train_df, val_df = Data_Processer().preprocess_data()
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
        checkpoint_path = None
        checkpoint = None
        if from_checkpoint:
            checkpoint_path = self.logs_path
            checkpoint = f"{checkpoint_path}/version_0/checkpoints/model.ckpt"
        self.model = Categories_Classifier(self.config, checkpoint_path=checkpoint_path)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            mode="min",
            monitor="validation_loss",
            filename="model",
            save_weights_only=True,  # Guardar solo los pesos del modelo
        )

        trainer = pl.Trainer(
            max_epochs=self.config["n_epochs"],
            num_sanity_val_steps=1,
            callbacks=[checkpoint_callback],
        )

        logs_dir = trainer.logger.root_dir
        shutil.rmtree(logs_dir, ignore_errors=True)
        os.makedirs(logs_dir, exist_ok=True)
        # Set float32 matrix multiplication precision
        torch.set_float32_matmul_precision("medium")
        trainer.fit(self.model, datamodule=data_module, ckpt_path=checkpoint)

    def save_model(self):
        model_path = f"{self.model_path}/trained_model"
        tokenizer_path = f"{self.model_path}/trained_tokenizer"
        self.model.pretrained_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        Categories.objects.all().update(label_index=None)
        index = 0
        for label_id, _ in self.categories:
            category = Categories.objects.get(id=label_id)
            category.label_index = index
            category.save()
            index += 1
