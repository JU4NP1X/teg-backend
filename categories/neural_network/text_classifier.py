import torch
import os
import pytorch_lightning as pl
from transformers import AutoTokenizer
from ..models import Categories
from .model import Categories_Classifier
import numpy as np
from .data_processer import Data_Processer
from .data_module import Data_Module
import pandas as pd

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


def create_pretrained_copy(tokenizer_path, tokenizer_name):
    if not os.path.exists(tokenizer_path):
        model = AutoTokenizer.from_pretrained(tokenizer_name)
        model.save_pretrained(tokenizer_path)


class TextClassifier:
    def __init__(self):
        self.best_model_path = os.path.join(BASE_DIR, "trained_model")
        self.best_model_checkpoint = f"{self.best_model_path}/model.ckpt"
        self.best_model_params = f"{self.best_model_path}/hparams.yaml"
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))

        self.df = Data_Processer()
        self.categories = self.df.get_categories(True)

        self.model = Categories_Classifier.load_from_checkpoint(
            self.best_model_checkpoint,
            hparams_file=self.best_model_params,
            pos_weights=np.ones(len(self.categories)),
        )

    def classify_text(self, text):
        self.model.eval()

        categories_names = [label_name for _, label_name in self.categories]
        # Crear el dataframe
        data = {"CONTEXT": [text]}
        data.update({column: 0 for column in categories_names})

        df = pd.DataFrame(data)
        data_loader = Data_Module(
            None,
            val_data=df,
            max_length=self.max_len,
            attributes=categories_names,
        )

        trainer = pl.Trainer()
        predictions = trainer.predict(
            self.model,
            datamodule=data_loader,
            ckpt_path=self.best_model_checkpoint,
        )[0]
        input_probs = torch.sigmoid(predictions)
        final_output = input_probs.cpu().numpy()[0]
        close_indexes = np.where(final_output >= 0.5)[0]
        categories_list = list(
            Categories.objects.filter(label_index__in=list(close_indexes)).values()
        )
        return categories_list
