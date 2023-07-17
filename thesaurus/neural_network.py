import pandas as pd
import os
import torch
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder
from django.db.models.functions import Concat
from django.db.models import Value, CharField, Subquery, OuterRef, Func
import numpy as np
from translate import Translator
import torch.nn.functional as F
from .models import Thesaurus
from datasets.models import Datasets
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from langdetect import detect
import shutil

THESAURUS_MODEL_PATH = os.environ.get(
    "THESAURUS_MODEL_PATH", "/home/juan/projects/teg/backend"
)
THESAURUS_MAX_LEN = os.environ.get("THESAURUS_MAX_LEN", 300)
THESAURUS_TRAIN_BATCH_SIZE = os.environ.get("THESAURUS_TRAIN_BATCH_SIZE", 3)
THESAURUS_VALID_BATCH_SIZE = os.environ.get("THESAURUS_VALID_BATCH_SIZE", 1)
THESAURUS_LEARNING_RATE = os.environ.get("THESAURUS_LEARNING_RATE", 1e-05)
THESAURUS_EPOCHS = os.environ.get("THESAURUS_EPOCHS", 10)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class GroupConcat(Func):
    function = "GROUP_CONCAT"
    template = "%(function)s(%(expressions)s)"


class BERTClass(torch.nn.Module):
    def __init__(self, size):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased", return_dict=True
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, size)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


class CustomDataset:
    def __init__(self, df, outputs):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df["CONTEXT"]
        self.targets = self.df[outputs].values

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        title = str(self.title[idx])
        title = " ".join(title.split())
        try:
            origin_aplha2 = detect(title)
            translator = Translator(from_lang=origin_aplha2, to_lang="en")
            translated_text = translator.translate(title)
        except Exception as e:
            print(f"Error translating text: {e}")
            translated_text = title

        inputs = self.tokenizer.encode_plus(
            translated_text,
            None,
            max_length=THESAURUS_MAX_LEN,
            padding="max_length",
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "token_type_ids": inputs["token_type_ids"].flatten(),
            "targets": torch.FloatTensor(self.targets[idx]),
        }


class TextClassifier:
    def __init__(self):
        self.categories = None
        self.num_outputs = None
        self.outputs = []
        self.num_inputs = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        self.model_path = THESAURUS_MODEL_PATH

    def preprocess_data(self):
        datasets = (
            Datasets.objects.all()
            .order_by("-id")[:100]
            .annotate(
                CONTEXT=Concat(
                    "paper_name", Value(": "), "summary", output_field=CharField()
                ),
                CATEGORIES=Subquery(
                    Thesaurus.objects.filter(datasets=OuterRef("pk"))
                    .values("id")
                    .annotate(ids=GroupConcat("id"))
                    .values("ids"),
                    output_field=CharField(),
                ),
            )
        )
        # Make the list of possible results
        label_ids_and_names = Thesaurus.objects.values_list("id", "name")

        # Extract only the IDs
        label_ids = [label_id for label_id, _ in label_ids_and_names]
        # Fit the MultiLabelBinarizer using the IDs only
        self.mlb.fit([label_ids])

        # Convert Datasets objects to a pandas DataFrame and shuffle the data
        df = pd.DataFrame.from_records(datasets.values("CATEGORIES", "CONTEXT"))
        df = df.sample(frac=1).reset_index(drop=True)
        labels = df["CATEGORIES"].apply(
            lambda x: [int(label_id) for label_id in x.split(",") if label_id]
        )
        binary_labels = pd.DataFrame(
            self.mlb.transform(labels),
            columns=[label_name for _, label_name in label_ids_and_names],
        )
        df.drop(
            labels=[
                "CATEGORIES",
            ],
            axis=1,
            inplace=True,
        )
        df = pd.concat([df, binary_labels], axis=1)

        # Split the data into training, validation, and test sets
        train_df = df.sample(frac=0.8, random_state=200).reset_index(drop=True)
        val_df = df.drop(train_df.index).reset_index(drop=True)
        self.outputs = binary_labels.columns.values.tolist()
        self.num_outputs = len(label_ids)
        print(train_df.head())
        print(val_df.head())
        return train_df, val_df

    def create_model(self):
        # Definir la arquitectura del modelo
        self.model = BERTClass(self.num_outputs)
        self.model.to(self.device)

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def train_model(
        self,
        n_epochs,
        training_loader,
        validation_loader,
    ):
        val_targets = []
        val_outputs = []

        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf
        for epoch in range(1, n_epochs + 1):
            train_loss = 0
            valid_loss = 0

            self.model.train()
            print(
                "############# Epoch {}: Training Start   #############".format(epoch)
            )
            for batch_idx, data in enumerate(training_loader):
                # print('yyy epoch', batch_idx)
                ids = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )
                targets = data["targets"].to(self.device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                # if batch_idx%5000==0:
                #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print('before loss data in training', loss.item(), train_loss)
                train_loss = train_loss + (
                    (1 / (batch_idx + 1)) * (loss.item() - train_loss)
                )
                # print('after loss data in training', loss.item(), train_loss)

            print(
                "############# Epoch {}: Training End     #############".format(epoch)
            )

            print(
                "############# Epoch {}: Validation Start   #############".format(epoch)
            )
            ######################
            # validate the model #
            ######################

            self.model.eval()

            with torch.no_grad():
                for batch_idx, data in enumerate(validation_loader, 0):
                    ids = data["input_ids"].to(self.device, dtype=torch.long)
                    mask = data["attention_mask"].to(self.device, dtype=torch.long)
                    token_type_ids = data["token_type_ids"].to(
                        self.device, dtype=torch.long
                    )
                    targets = data["targets"].to(self.device, dtype=torch.float)
                    outputs = self.model(ids, mask, token_type_ids)

                    loss = self.loss_fn(outputs, targets)
                    valid_loss = valid_loss + (
                        (1 / (batch_idx + 1)) * (loss.item() - valid_loss)
                    )
                    val_targets.extend(targets.cpu().detach().numpy().tolist())
                    val_outputs.extend(
                        torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                    )

            print(
                "############# Epoch {}: Validation End     #############".format(epoch)
            )
            # calculate average losses
            # print('before cal avg train loss', train_loss)
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            # print training/validation statistics
            print(
                "Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}".format(
                    epoch, train_loss, valid_loss
                )
            )

            # create checkpoint variable and add important data
            checkpoint = {
                "epoch": epoch + 1,
                "valid_loss_min": valid_loss,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            # save checkpoint
            self.save_ckp(checkpoint, False)

            ## TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    )
                )
                # save checkpoint as best model
                self.save_ckp(checkpoint, True)
                valid_loss_min = valid_loss

            print("############# Epoch {}  Done   #############\n".format(epoch))

    def load_ckp(self, checkpoint_fpath):
        # load check point
        checkpoint = torch.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        self.model.load_state_dict(checkpoint["state_dict"])
        # initialize optimizer from checkpoint to optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint["valid_loss_min"]
        # return model, optimizer, epoch value, min validation loss
        return checkpoint["epoch"], valid_loss_min.item()

    def save_ckp(self, state, is_best):
        c_path = f"{self.model_path}/curr_ckpt.pt"
        b_path = f"{self.model_path}/best_model.pt"
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, c_path)
        # if it is a best model, min validation loss
        if is_best:
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(c_path, b_path)

    def retrain_model(self, model_path, new_num_outputs):
        self.model = TextClassifier(self.num_inputs, new_num_outputs)
        self.model.load_state_dict(torch.load(model_path))

        # Aquí necesitarías definir tu nuevo bucle de entrenamiento

    def train(self):
        train_df, val_df = self.preprocess_data()
        train_data = CustomDataset(train_df, self.outputs)
        val_data = CustomDataset(val_df, self.outputs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=THESAURUS_LEARNING_RATE
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=THESAURUS_TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=THESAURUS_VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        self.train_model(THESAURUS_EPOCHS, train_data_loader, val_data_loader)
