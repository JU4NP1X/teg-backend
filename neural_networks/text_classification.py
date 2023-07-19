import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from django.db.models import Func
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
import shutil
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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
    def __init__(self, df, outputs, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len
        self.title = df["CONTEXT"]
        self.targets = self.df[outputs].values

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        title = str(self.title[idx])
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            max_length=self.max_len,
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
        self.model_path = "/home/juan/projects/teg/backend"
        self.max_len = 256
        self.train_batch_size = 3
        self.valid_batch_size = 1
        self.learning_rate = 1e-05
        self.epochs = 2

    def get_data(self):
        pass

    def get_labels(self):
        pass

    # Really better solution, but slower
    def balance_data(self, df):
        # Primero, vamos a aplanar la lista de etiquetas y contar la frecuencia de cada una
        labels = [label for sublist in df["LABELS"].tolist() for label in sublist]
        counter = Counter(labels)

        # Encuentra la categoría mínima para balancear
        min_category = min(counter, key=counter.get)
        print("min_cat", min_category)

        # Crea un nuevo DataFrame vacío para almacenar los datos balanceados
        balanced_df = pd.DataFrame(columns=df.columns)

        # Itera sobre cada fila en el DataFrame original
        for _, row in df.iterrows():
            # Si la fila contiene la categoría mínima, añádela al DataFrame balanceado
            if min_category in row["LABELS"]:
                balanced_df = pd.concat([balanced_df, row.to_frame().T])
            else:
                worst_proportion_in_row = float("inf")
                for category in row["LABELS"]:
                    if counter[category] < worst_proportion_in_row:
                        worst_proportion_in_row = counter[category]
                # Si no, añade la fila al DataFrame balanceado con una probabilidad igual a la proporción de la categoría mínima
                if worst_proportion_in_row <= 5 or (
                    (np.random.rand() * (counter[min_category] + 5))
                    > (np.random.rand() * worst_proportion_in_row)
                ):
                    balanced_df = pd.concat([balanced_df, row.to_frame().T])

        # Mezcla el DataFrame balanceado para asegurar que los datos estén distribuidos aleatoriamente
        balanced_df = shuffle(balanced_df)

        # Resetea los índices del DataFrame balanceado
        balanced_df.reset_index(drop=True, inplace=True)
        self.plot_category_counts(df, balanced_df)
        return balanced_df

    # This balancer is really bad
    def balance_data2(self, df):
        # Primero, vamos a aplanar la lista de etiquetas y contar la frecuencia de cada una
        labels = [label for sublist in df["LABELS"].tolist() for label in sublist]
        counter = Counter(labels)

        # Encuentra la categoría mínima para balancear
        min_category = min(counter, key=counter.get)
        print("min_cat", min_category)

        # Crea un nuevo DataFrame vacío para almacenar los datos balanceados
        balanced_df = pd.DataFrame(columns=df.columns)

        # Itera sobre cada fila en el DataFrame original
        for index, row in df.iterrows():
            # Si la fila contiene la categoría mínima, añádela al DataFrame balanceado
            if min_category in row["LABELS"]:
                balanced_df = pd.concat([balanced_df, row.to_frame().T])
            else:
                # Si no, añade la fila al DataFrame balanceado con una probabilidad igual a la proporción de la categoría mínima
                if np.random.rand() < counter[min_category] / len(row["LABELS"]):
                    balanced_df = pd.concat([balanced_df, row.to_frame().T])

        # Mezcla el DataFrame balanceado para asegurar que los datos estén distribuidos aleatoriamente
        balanced_df = shuffle(balanced_df)

        # Resetea los índices del DataFrame balanceado
        balanced_df.reset_index(drop=True, inplace=True)
        self.plot_category_counts(df, balanced_df)
        return balanced_df

    def plot_category_counts(self, df, balanced_df):
        # Cuenta la frecuencia de cada categoría en el DataFrame original
        original_counts = df["LABELS"].explode().value_counts().sort_index()

        # Cuenta la frecuencia de cada categoría en el DataFrame balanceado
        balanced_counts = balanced_df["LABELS"].explode().value_counts().sort_index()

        # Crea una figura y dos subplots para mostrar los gráficos antes y después del balanceo
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Grafica el número de coincidencias por categoría antes del balanceo
        ax1.bar(original_counts.index, original_counts.values)
        ax1.set_title("Antes del balanceo")
        ax1.set_xlabel("Categoría")
        ax1.set_ylabel("Número de coincidencias")

        # Grafica el número de coincidencias por categoría después del balanceo
        ax2.bar(balanced_counts.index, balanced_counts.values)
        ax2.set_title("Después del balanceo")
        ax2.set_xlabel("Categoría")
        ax2.set_ylabel("Número de coincidencias")

        # Ajusta los espacios entre los subplots
        plt.tight_layout()

        # Guarda la gráfica como una imagen en formato PNG
        plt.savefig("category_counts.png")

    def preprocess_data(self):
        df = self.get_data()
        # Make the list of possible results
        label_ids_and_names = self.get_labels()

        # Extract only the IDs
        label_ids = [label_id for label_id, _ in label_ids_and_names]
        # Fit the MultiLabelBinarizer using the IDs only
        self.mlb.fit([label_ids])

        df = self.balance_data(df)

        binary_labels = pd.DataFrame(
            self.mlb.transform(df["LABELS"]),
            columns=[label_name for _, label_name in label_ids_and_names],
        )
        df = pd.concat([df, binary_labels], axis=1)

        # Split the data into training, validation, and test sets
        train_df = df.sample(frac=0.8, random_state=200).reset_index(drop=True)
        val_df = df.drop(train_df.index).reset_index(drop=True)
        self.outputs = binary_labels.columns.values.tolist()
        self.num_outputs = len(label_ids)
        print(len(df))
        print(train_df.head(), len(df))
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
        training_loader,
        validation_loader,
    ):
        val_targets = []
        val_outputs = []

        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf
        for epoch in range(1, self.epochs + 1):
            train_loss = 0
            valid_loss = 0

            self.model.train()

            print("Training {}".format(epoch))

            for batch_idx, data in enumerate(training_loader, 0):
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

            ######################
            # validate the model #
            ######################
            print("Validation {}".format(epoch))

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

            # calculate average losses
            # print('before cal avg train loss', train_loss)
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            # print training/validation statistics
            print(
                "Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}".format(
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
        train_data = CustomDataset(train_df, self.outputs, self.max_len)
        val_data = CustomDataset(val_df, self.outputs, self.max_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.valid_batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.train_model(train_data_loader, val_data_loader)
