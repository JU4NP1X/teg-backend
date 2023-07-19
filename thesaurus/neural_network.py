import pandas as pd
import os
import torch
from django.db.models.functions import Concat
from django.db.models import Value, CharField, Subquery, OuterRef
from .models import Thesaurus
from thesaurus_datasets.models import Datasets_English_Translations
import shutil
from neural_networks.text_classification import TextClassifier as BaseTextClassifier
from neural_networks.text_classification import GroupConcat


class TextClassifier(BaseTextClassifier):
    def __init__(self):
        super().__init__()
        self.model_path = os.environ.get(
            "THESAURUS_MODEL_PATH", "/home/juan/projects/teg/backend"
        )
        self.max_len = int(os.environ.get("THESAURUS_MAX_LEN", 256))
        self.train_batch_size = int(os.environ.get("THESAURUS_TRAIN_BATCH_SIZE", 3))
        self.valid_batch_size = int(os.environ.get("THESAURUS_VALID_BATCH_SIZE", 1))
        self.learning_rate = float(os.environ.get("THESAURUS_LEARNING_RATE", 1e-05))
        self.epochs = int(os.environ.get("THESAURUS_EPOCHS", 2))

    def get_labels(self):
        return Thesaurus.objects.values_list("id", "name")

    def get_data(self):
        datasets = Datasets_English_Translations.objects.all().annotate(
            CONTEXT=Concat(
                "paper_name", Value(": "), "summary", output_field=CharField()
            ),
            CATEGORIES=Subquery(
                Thesaurus.objects.filter(datasets=OuterRef("dataset"))
                .values("id")
                .annotate(ids=GroupConcat("id"))
                .values("ids"),
                output_field=CharField(),
            ),
            RELATED_THESAURI=Subquery(
                Thesaurus.objects.filter(datasets=OuterRef("dataset"))
                .values("related_thesauri__id")
                .annotate(ids=GroupConcat("related_thesauri__id"))
                .values("ids"),
                output_field=CharField(),
            ),
            PARENT_THESAURUS=Subquery(
                Thesaurus.objects.filter(datasets=OuterRef("dataset"))
                .values("parent_thesaurus__id")
                .annotate(ids=GroupConcat("parent_thesaurus__id"))
                .values("ids"),
                output_field=CharField(),
            ),
        )

        df = pd.DataFrame.from_records(
            datasets.values(
                "CATEGORIES", "CONTEXT", "RELATED_THESAURI", "PARENT_THESAURUS"
            )
        )
        df = df.sample(frac=1).reset_index(drop=True)

        # Combine the categories, related thesauri, and parent thesaurus into a single column
        df["LABELS"] = df.apply(
            lambda row: list(
                set(
                    (row["CATEGORIES"].split(",") if row["CATEGORIES"] else [])
                    + (
                        row["RELATED_THESAURI"].split(",")
                        if row["RELATED_THESAURI"]
                        else []
                    )
                    + (
                        row["PARENT_THESAURUS"].split(",")
                        if row["PARENT_THESAURUS"]
                        else []
                    )
                )
            ),
            axis=1,
        )

        # Drop the individual category, related thesauri, and parent thesaurus columns
        df.drop(
            ["CATEGORIES", "RELATED_THESAURI", "PARENT_THESAURUS"], axis=1, inplace=True
        )
        df["LABELS"] = df["LABELS"].apply(
            lambda labels: [int(label_id) for label_id in labels if label_id]
        )
        return df

    def save_ckp(self, state, is_best):
        c_path = f"{self.model_path}/thesaurus_curr_ckpt.pt"
        b_path = f"{self.model_path}/thesaurus_best_model.pt"
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, c_path)
        # if it is a best model, min validation loss
        if is_best:
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(c_path, b_path)
