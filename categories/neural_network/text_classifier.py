import os
import torch
import yaml
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from categories.neural_network.data_processer import DataProcesser
from categories.neural_network.model import CategoriesClassifier
from categories.models import Categories


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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(tokenizer_path)


class TextClassifier:
    """
    Text Classifier api service
    """

    def __init__(self, authority_id, loaded_at):
        """
        Initializes the TextClassifier object.

        Returns:
            None
        """
        self.best_model_path = os.path.join(BASE_DIR, "trained_model")
        self.best_model_weights = (
            f"{self.best_model_path}/{authority_id}/model_weights.pt"
        )
        tokenizer_path = os.path.join(BASE_DIR, "roberta-large")

        self.best_model_params = f"{self.best_model_path}/{authority_id}/hparams.yaml"
        with open(self.best_model_params, "r") as file:
            yaml_dict = yaml.safe_load(file)

        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))
        self.loaded_at = loaded_at
        self.categories = Categories.objects.filter(
            authority__id=authority_id, deprecated=False, parent=None
        ).count()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.model = CategoriesClassifier(
            config=yaml_dict["config"],
            pos_weights=np.ones(self.categories),
        )
        self.model.load_state_dict(
            torch.load(self.best_model_weights, map_location=torch.device("cpu"))
        )
        torch.set_default_dtype(torch.float32)
        self.model.eval()

    def classify_text(self, text):
        """
        Classifies the given text into categories.

        Args:
            text (str): The text to be classified.

        Returns:
            list: A list of categories that the text belongs to.
        """
        encoded_input = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)[1]

        input_probs = torch.sigmoid(output)
        final_output = input_probs.cpu().numpy()[0]
        close_indexes = np.where(final_output >= 0.5)[0]
        filtered_categories = Categories.objects.filter(
            label_index__in=list(close_indexes)
        )
        return filtered_categories
