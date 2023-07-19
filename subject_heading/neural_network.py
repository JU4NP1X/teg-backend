
from neural_networks.text_classification import TextClassifier as BaseTextClassifier
from neural_networks.text_classification import GroupConcat
from .models import Subject_Headers
import pandas as pd
import os
import torch
from django.db.models.functions import Concat
from django.db.models import Value, CharField, Subquery, OuterRef
from thesaurus_datasets.models import Datasets_English_Translations
import shutil


class TextClassifier(BaseTextClassifier):
    def __init__(self):
        super().__init__()
        self.model_path = os.environ.get(
            "SUBJECT_HEADING_MODEL_PATH", "/home/juan/projects/teg/backend"
        )
        self.max_len = int(os.environ.get("SUBJECT_HEADING_MAX_LEN", 256))
        self.train_batch_size = int(
            os.environ.get("SUBJECT_HEADING_TRAIN_BATCH_SIZE", 3)
        )
        self.valid_batch_size = int(
            os.environ.get("SUBJECT_HEADING_VALID_BATCH_SIZE", 1)
        )
        self.learning_rate = float(
            os.environ.get("SUBJECT_HEADING_LEARNING_RATE", 1e-05)
        )
        self.epochs = int(os.environ.get("SUBJECT_HEADING_EPOCHS", 2))

    def get_data(self):
        datasets = Datasets_English_Translations.objects.all().annotate(
            CONTEXT=Concat(
                "paper_name", Value(": "), "summary", output_field=CharField()
            ),
            CATEGORIES=Subquery(
                Subject_Headers.objects.filter(datasets=OuterRef("dataset"))
                .values("id")
                .annotate(ids=GroupConcat("id"))
                .values("ids"),
                output_field=CharField(),
            ),
        )

        df = pd.DataFrame.from_records(
            datasets.values(
                "CATEGORIES",
                "CONTEXT",
            )
        )
        df = df.sample(frac=1).reset_index(drop=True)

        # Combine the categories, related thesauri, and parent thesaurus into a single column
        df["LABELS"] = df.apply(
            lambda row: list(
                set((row["CATEGORIES"].split(",") if row["CATEGORIES"] else []))
            ),
            axis=1,
        )

        # Drop the individual category, related thesauri, and parent thesaurus columns
        df.drop(["CATEGORIES"], axis=1, inplace=True)
        df["LABELS"] = df["LABELS"].apply(
            lambda labels: [int(label_id) for label_id in labels if label_id]
        )
        return df

    def get_labels(self):
        return Subject_Headers.objects.values_list("id", "name")

    def save_ckp(self, state, is_best):
        c_path = f"{self.model_path}/subject_heading_curr_ckpt.pt"
        b_path = f"{self.model_path}/subject_heading_best_model.pt"
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, c_path)
        # if it is a best model, min validation loss
        if is_best:
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(c_path, b_path)
