"""
Trainer model of the categories classifier
"""
import torch
import math
import os
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW

BASE_DIR = os.path.dirname(os.path.realpath(__name__))


def create_pretrained_copy(model_path, model_name):
    """
    Creates a pretrained copy of a model if it doesn't already exist.

    Args:
        model_path (str): The path to save the pretrained model.
        model_name (str): The name of the model.

    Returns:
        None
    """
    if not os.path.exists(model_path):
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(model_path)


class CustomLoss(pl.LightningModule):
    """
    Custom loss function for the classifier.

    Args:
        pos_weight (torch.Tensor): The positive weight for the loss function.

    Returns:
        None
    """

    def __init__(self, pos_weight):
        super(CustomLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target, device):
        """
        Computes the custom loss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
            device (torch.device): The device to use.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Apply sigmoid to convert logits to probabilities
        input_probs = torch.sigmoid(input)
        # Get the list of values that are 1 and 0
        ones_mask = target.clone().detach().to(torch.bool)
        zeros_mask = ones_mask.clone() == 0

        # Only is important the mean of the 1 that are correct or not
        ones_loss = torch.masked_select(
            torch.ones_like(input_probs) - input_probs, ones_mask
        )

        # Assign higher weight to cases where expected output is 0 and predicted output is 1
        zeros_loss = torch.masked_select(input_probs, zeros_mask)

        # Combine the two losses
        loss = zeros_loss.mean() + ones_loss.mean()

        return loss / 2


class CategoriesClassifier(pl.LightningModule):
    """
    A classifier model for categorizing data.

    Args:
        config (dict): The configuration parameters for the model.
        pos_weights (torch.Tensor): The positive weights for the loss function.
        learning_rate (float): The learning rate for training.

    Returns:
        None
    """

    def __init__(self, config: dict, pos_weights, learning_rate=None):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weights", "learning_rate"])
        pos_weights = torch.tensor(pos_weights)
        if learning_rate:
            config["lr"] = learning_rate
        print(config)
        self.config = config
        model_path = os.path.join(BASE_DIR, config["model_name"])

        create_pretrained_copy(model_path, config["model_name"])
        self.pretrained_model = AutoModel.from_pretrained(model_path, return_dict=True)
        self.hidden_1 = torch.nn.Linear(
            self.pretrained_model.config.hidden_size,
            self.pretrained_model.config.hidden_size * 2,
        )

        self.hidden_2 = torch.nn.Linear(
            self.pretrained_model.config.hidden_size * 2,
            self.pretrained_model.config.hidden_size,
        )

        self.classifier = torch.nn.Linear(
            self.pretrained_model.config.hidden_size, self.config["n_labels"]
        )
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_func = CustomLoss(pos_weight=pos_weights)
        self.dropout = nn.Dropout()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            labels (torch.Tensor): The labels tensor.

        Returns:
            tuple: The loss and logits.
        """
        output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden_1(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden_2(pooled_output)

        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = 0
        if labels is not None:
            logits = logits.to(labels.device)
            loss = self.loss_func(
                logits.view(-1, self.config["n_labels"]),
                labels.view(-1, self.config["n_labels"]),
                labels.device,
            )
        return loss, logits

    def training_step(self, batch, batch_index):
        """
        Training step of the model.

        Args:
            batch (dict): The batch data.
            batch_index (int): The batch index.

        Returns:
            dict: The loss, predictions, and labels.
        """
        loss, outputs = self(**batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        """
        Validation step of the model.

        Args:
            batch (dict): The batch data.
            batch_index (int): The batch index.

        Returns:
            dict: The validation loss, predictions, and labels.
        """
        loss, outputs = self(**batch)
        self.log("validation_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def test_step(self, batch, batch_index):
        """
        Test step of the model.

        Args:
            batch (dict): The batch data.
            batch_index (int): The batch index.

        Returns:
            dict: The test loss, predictions, and labels.
        """
        loss, outputs = self(**batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def predict_step(self, batch, batch_index):
        """
        Prediction step of the model.

        Args:
            batch (dict): The batch data.
            batch_index (int): The batch index.

        Returns:
            torch.Tensor: The predictions.
        """
        _, outputs = self(**batch)
        return outputs

    def configure_optimizers(self):
        """
        Configures the optimizers and schedulers for training.

        Returns:
            list: The optimizers.
            list: The schedulers.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        total_steps = self.config["train_size"] / self.config["batch_size"]
        warmup_steps = math.floor(total_steps * self.config["warmup"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        return [optimizer], [scheduler]
