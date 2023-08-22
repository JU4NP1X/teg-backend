import torch
import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from .data_processer import DataProcesser
from .data_module import DataModule
from .model import CategoriesClassifier
from ..models import Categories


BASE_DIR = os.path.dirname(os.path.realpath(__name__))


def create_pretrained_copy(tokenizer_path, tokenizer_name):
    """
    Creates a pretrained copy of the tokenizer.

    Args:
        tokenizer_path (str): The path to save the pretrained tokenizer.
        tokenizer_name (str): The name of the pretrained tokenizer.

    Returns:
        None
    """
    if not os.path.exists(tokenizer_path):
        model = AutoTokenizer.from_pretrained(tokenizer_name)
        model.save_pretrained(tokenizer_path)


def get_descendants(categories):
    """
    Retrieves the descendants of the given categories.

    Args:
        categories (list): A list of categories.

    Returns:
        list: A list of descendants for each category.
    """
    descendants = []
    for category in categories:
        descendants.append(category.include_descendants())
    return descendants


class TextClassifier:
    def __init__(self):
        """
        Initializes the TextClassifier object.

        Returns:
            None
        """
        self.best_model_path = os.path.join(BASE_DIR, "trained_model")
        self.best_model_checkpoint = f"{self.best_model_path}/model.ckpt"
        self.best_model_params = f"{self.best_model_path}/hparams.yaml"
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))

        self.df = DataProcesser()
        self.categories = self.df.get_categories(True)

        self.model = CategoriesClassifier.load_from_checkpoint(
            self.best_model_checkpoint,
            hparams_file=self.best_model_params,
            pos_weights=np.ones(len(self.categories)),
        )

    def classify_text(self, text):
        """
        Classifies the given text into categories.

        Args:
            text (str): The text to be classified.

        Returns:
            list: A list of categories that the text belongs to.
        """
        self.model.eval()

        categories_names = [label_name for _, label_name in self.categories]
        # Crear el dataframe
        data = {"CONTEXT": [text]}
        data.update({column: 0 for column in categories_names})

        df = pd.DataFrame(data)
        data_loader = DataModule(
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
        filtered_categories = Categories.objects.filter(
            label_index__in=list(close_indexes)
        )
        categories_list = get_descendants(filtered_categories)
        return categories_list
