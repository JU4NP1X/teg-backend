"""
This is the tensor constructor of the dataset
"""
from torch.utils.data import Dataset
import torch


class DatasetTensor(Dataset):
    """
    This is the class tensor constructor of the dataset
    """

    def __init__(self, df, outputs, max_len, tokenizer):
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len
        self.summary = df["CONTEXT"]
        self.targets = self.df[outputs].values

    def __len__(self):
        return len(self.summary)

    def __getitem__(self, index):
        summary = str(self.summary[index])
        attributes = torch.FloatTensor(self.targets[index])
        tokens = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
        )
        return {
            "input_ids": tokens.input_ids.flatten(),
            "attention_mask": tokens.attention_mask.flatten(),
            "labels": attributes,
        }
