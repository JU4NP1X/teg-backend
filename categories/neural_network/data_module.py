from transformers import AutoTokenizer
import pytorch_lightning as pl
from .dataset_constructor import Dataset_Tensor
import os
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


def create_pretrained_copy(tokenizer_path, tokenizer_name):
    if not os.path.exists(tokenizer_path):
        model = AutoTokenizer.from_pretrained(tokenizer_name)
        model.save_pretrained(tokenizer_path)


class Data_Module(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        attributes,
        max_length,
        batch_size: int = 16,
    ):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.attributes = attributes
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer_name = "roberta-base"
        tokenizer_path = os.path.join(BASE_DIR, self.tokenizer_name)
        create_pretrained_copy(tokenizer_path, self.tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = Dataset_Tensor(
                self.train_data,
                outputs=self.attributes,
                max_len=self.max_length,
                tokenizer=self.tokenizer,
            )
            self.val_dataset = Dataset_Tensor(
                self.val_data,
                outputs=self.attributes,
                max_len=self.max_length,
                tokenizer=self.tokenizer,
            )
        if stage in ("test", "predict"):
            self.val_dataset = Dataset_Tensor(
                self.val_data,
                outputs=self.attributes,
                max_len=self.max_length,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
