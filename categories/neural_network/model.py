import pytorch_lightning as pl
from transformers import AutoModel, get_cosine_schedule_with_warmup
import torch.nn as nn
import torch
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
from torch.optim import AdamW


class Categories_Classifier(pl.LightningModule):
    def __init__(self, config: dict, checkpoint_path=None):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(
            config["model_name"], return_dict=True
        )
        self.hidden = torch.nn.Linear(
            self.pretrained_model.config.hidden_size,
            self.pretrained_model.config.hidden_size,
        )
        self.classifier = torch.nn.Linear(
            self.pretrained_model.config.hidden_size, self.config["n_labels"]
        )
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.dropout = nn.Dropout()
        self.save_hyperparameters()

        if checkpoint_path is not None:
            checkpoint = f"{checkpoint_path}/version_0/checkpoints/model.ckpt"
            hparams_file = f"{checkpoint_path}/version_0/hparams.yaml"
            self.load_from_checkpoint(checkpoint, hparams_file=hparams_file)

    def forward(self, input_ids, attention_mask, labels=None):
        # roberta layer
        output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled_output = torch.mean(output.last_hidden_state, 1)
        # final logits
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.loss_func(
                logits.view(-1, self.config["n_labels"]),
                labels.view(-1, self.config["n_labels"]),
            )
        return loss, logits

    def training_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        self.log("validation_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def predict_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        return outputs

    def configure_optimizers(self):
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

    # def validation_epoch_end(self, outputs):
    #   losses = []
    #   for output in outputs:
    #     loss = output['val_loss'].detach().cpu()
    #     losses.append(loss)
    #   avg_loss = torch.mean(torch.stack(losses))
    #   self.log("avg_val_loss", avg_loss)
