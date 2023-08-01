from transformers import AutoTokenizer
import pytorch_lightning as pl
from .dataset_constructor import Dataset_Tensor

from torch.utils.data import DataLoader


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
        self.model_name = "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
        if stage == "predict":
            self.val_dataset = Dataset_Tensor(
                self.val_data,
                outputs=self.attributes,
                max_len=self.max_length,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        data = DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )
        print(data)
        return data

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
