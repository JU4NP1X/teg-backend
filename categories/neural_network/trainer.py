import pandas as pd
import os
import torch
from django.db.models.functions import Concat
from django.db.models import Value, CharField, Subquery, OuterRef
from ..models import Categories
from datasets.models import Datasets_English_Translations
import shutil
from categories.neural_network.text_classification_base import TextClassifier as BaseTextClassifier
from categories.neural_network.text_classification_base import GroupConcat


class TextClassifier(BaseTextClassifier):
    def __init__(self):
        super().__init__()
        self.model_path = os.environ.get(
            "CATEGORIES_MODEL_PATH", "/home/juan/projects/teg/backend"
        )
        self.max_len = int(os.environ.get("CATEGORIES_MAX_LEN", 256))
        self.train_batch_size = int(os.environ.get("CATEGORIES_TRAIN_BATCH_SIZE", 3))
        self.valid_batch_size = int(os.environ.get("CATEGORIES_VALID_BATCH_SIZE", 1))
        self.learning_rate = float(os.environ.get("CATEGORIES_LEARNING_RATE", 1e-05))
        self.epochs = int(os.environ.get("CATEGORIES_EPOCHS", 2))

    def get_categories(self):
        return Categories.objects.filter(deprecated=False).values_list("id", "name")

    def get_data(self):
        datasets = Datasets_English_Translations.objects.filter(
            dataset__categories__deprecated=False
        ).annotate(
            CONTEXT=Concat(
                "paper_name", Value(": "), "summary", output_field=CharField()
            ),
            CATEGORIES=Subquery(
                Categories.objects.filter(
                    datasets=OuterRef("dataset"), deprecated=False
                )
                .values("id")
                .annotate(ids=GroupConcat("id"))
                .values("ids"),
                output_field=CharField(),
            ),
            RELATED_CATEGORIES=Subquery(
                Categories.objects.filter(
                    datasets=OuterRef("dataset"),
                    deprecated=False,
                    related_categories__deprecated=False,
                )
                .values("related_categories__id")
                .annotate(ids=GroupConcat("related_categories__id"))
                .values("ids"),
                output_field=CharField(),
            ),
            PARENT_CATEGORIES=Subquery(
                Categories.objects.filter(
                    datasets=OuterRef("dataset"), deprecated=False
                )
                .values("parent_category__id")
                .annotate(ids=GroupConcat("parent_category__id"))
                .values("ids"),
                output_field=CharField(),
            ),
        )

        df = pd.DataFrame.from_records(
            datasets.values(
                "CATEGORIES", "CONTEXT", "RELATED_CATEGORIES", "PARENT_CATEGORIES"
            )
        )
        df = df.sample(frac=1).reset_index(drop=True)

        # Combine the categories, related thesauri, and parent categories into a single column
        df["CATEGORIES"] = df.apply(
            lambda row: list(
                set(
                    (row["CATEGORIES"].split(",") if row["CATEGORIES"] else [])
                    + (
                        row["RELATED_CATEGORIES"].split(",")
                        if row["RELATED_CATEGORIES"]
                        else []
                    )
                )
            ),
            axis=1,
        )

        # Drop the individual category, related thesauri, and parent categories columns
        df.drop(
            [ "RELATED_CATEGORIES", "PARENT_CATEGORIES"],
            axis=1,
            inplace=True,
        )
        df["CATEGORIES"] = df["CATEGORIES"].apply(
            lambda labels: [int(label_id) for label_id in labels if label_id]
        )
        return df

    def save_ckp(self, state, is_best):
        c_path = f"{self.model_path}/categories_curr_ckpt.pt"
        b_path = f"{self.model_path}/categories_best_model.pt"
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, c_path)
        # if it is a best model, min validation loss
        if is_best:
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(c_path, b_path)
