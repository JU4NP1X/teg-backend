from transformers import AutoTokenizer
from .data_processer import Data_Processer
from .data_module import Data_Module
from .model import Categories_Classifier
import pytorch_lightning as pl
import numpy as np
import os
from ..models import Categories


class Classifier:
    def __init__(self, trained=True):
        self.model_path = os.environ.get(
            "CATEGORIES_MODEL_PATH", "/home/juan/projects/teg/backend"
        )
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))
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
        self.batch_size = int(os.environ.get("CATEGORIES_TRAIN_BATCH_SIZE", 5))

        self.config = {
            "model_name": self.model_name,
            "n_labels": len(categories_names),
            "batch_size": self.batch_size,
            "lr": 1.5e-6,
            "warmup": 0.2,
            "weight_decay": 0.001,
            "n_epochs": 10,
        }
        self.model = Categories_Classifier(self.config)

    def train(self):
        df = Data_Processer()
        train_df, val_df = df.preprocess_data()
        categories = df.get_categories()
        categories_names = [label_name for _, label_name in categories]

        data_module = Data_Module(
            train_data=train_df,
            val_data=val_df,
            attributes=categories_names,
            max_length=self.max_len,
            batch_size=self.batch_size,
        )
        data_module.setup()

        self.config["train_size"] = len(data_module.train_dataloader())
        print(data_module.train_dataloader())
        self.model = Categories_Classifier(self.config)
        trainer = pl.Trainer(max_epochs=self.config["n_epochs"], num_sanity_val_steps=1)

        trainer.fit(self.model, data_module)

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
