import torch
import os
from transformers import AutoTokenizer
from ..models import Categories
from .model import Categories_Classifier
import numpy as np
from .data_processer import Data_Processer


class TextClassifier:
    def __init__(self):
        self.model_path = os.environ.get(
            "CATEGORIES_MODEL_PATH", "/home/juan/projects/teg/backend"
        )
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 300))
        self.batch_size = int(os.environ.get("CATEGORIES_BATCH_SIZE"))
        self.learning_rate = float(os.environ.get("CATEGORIES_LEARNING_RATE", 1.5e-6))
        self.epochs = int(os.environ.get("CATEGORIES_EPOCHS", 2))

        self.df = Data_Processer()
        self.categories = self.df.get_categories(True)

        self.categories = Categories.objects.all().exclude(label_index__isnull=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{self.model_path}/trained_tokenizer"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        categories_names = [label_name for _, label_name in self.categories]

        self.config = {
            "model_name": f"{self.model_path}/trained_model",
            "n_labels": len(categories_names),
            "batch_size": self.batch_size,
            "lr": self.learning_rate,
            "warmup": 0.2,
            "weight_decay": 0.001,
            "n_epochs": self.epochs,
        }
        self.model = Categories_Classifier(self.config)

    def classify_text(self, text):
        encodings = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device, dtype=torch.long)
            attention_mask = encodings["attention_mask"].to(
                self.device, dtype=torch.long
            )
            token_type_ids = encodings["token_type_ids"].to(
                self.device, dtype=torch.long
            )
            output = self.model(input_ids, attention_mask, token_type_ids)
            final_output = torch.sigmoid(output).cpu().detach().numpy()[0]
            max_index = np.argmax(final_output)
            differences = np.abs(final_output - final_output[max_index])
            close_indices = np.where(differences <= 0.04)[0]
        categories_list = list(
            Categories.objects.filter(label_index__in=close_indices).values()
        )
        return categories_list
