import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
)
from torch.optim import AdamW
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from print_color import print
from tqdm.std import tqdm

from sits_siam.original import (
    OriginalSITSBert,
    OriginalSITSBertMissingMaskFix,
)
import random
import sys
import os

from sits_siam.augment import (
    AddMissingMask,
    Pipeline,
    ToPytorchTensor,
    IncreaseSequenceLength,
    LimitSequenceLength,
)

from sits_siam.utils import AgriGEELiteDataset
import torch.nn as nn

import geopandas as gpd
import pandas as pd


def setup_seed():
    seed = 123

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def beautify_prints():
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)


setup_seed()
beautify_prints()

transforms = Pipeline(
    [
        LimitSequenceLength(127),
        IncreaseSequenceLength(127),
        AddMissingMask(),
        OriginalSITSBertMissingMaskFix(),
        ToPytorchTensor(),
    ]
)

train_dataset = AgriGEELiteDataset(
    "data/agl/gdf.parquet", "data/agl/df_sits.parquet", transform=transforms
)
val_dataset = train_dataset
test_dataset = train_dataset

# print(train_dataset[0])
# exit(0)


class OriginalSITSBertTrainer(pl.LightningModule):
    def __init__(self, max_seq_len=127, num_classes=int(train_dataset.num_classes)):
        super(OriginalSITSBertTrainer, self).__init__()
        pretrain_path = "weights/original_sits_bert/checkpoint.bert.pth"
        self.model = OriginalSITSBert(num_classes=18, pretrain_path=pretrain_path)
        self.model.classification.linear = nn.Linear(256, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.train_f1 = MulticlassCohenKappa(num_classes=num_classes)
        self.val_f1 = MulticlassCohenKappa(num_classes=num_classes)
        self.test_f1 = MulticlassCohenKappa(num_classes=num_classes)
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, input):
        x = input["x"]
        doy = input["doy"]
        mask = input["mask"]

        outputs = self.model(x, doy, mask)

        return outputs

    def training_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        train_oa_score = self.train_oa(outputs, targets)
        train_f1_score = self.train_f1(outputs, targets)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_oa", train_oa_score, prog_bar=True)
        self.log("train_kappa", train_f1_score, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        val_oa_score = self.val_oa(outputs, targets)
        val_f1_score = self.val_f1(outputs, targets)

        # Log loss and oa score
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_oa", val_oa_score, prog_bar=True, sync_dist=True)
        self.log("val_kappa", val_f1_score, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        test_oa_score = self.test_oa(outputs, targets)
        test_f1_score = self.test_f1(outputs, targets)
        self.test_cm(outputs, targets)

        # Log loss and oa score
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_oa", test_oa_score, sync_dist=True)
        self.log("test_kappa", test_f1_score, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        self.confusion_matrix = self.test_cm.compute()
        self.test_cm.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=10, mode="min", verbose=True
)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    precision="16-mixed",
    devices=[1, 2],
    callbacks=[checkpoint_callback, early_stopping_callback],
)

model = OriginalSITSBertTrainer()

trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)


def predict_and_save_predictions(
    model, dataloader, output_path="predictions.npz", device="cuda"
):
    model.eval()
    model.to(device)

    all_true = []
    all_preds = []
    all_scores = []

    with torch.inference_mode():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch)

            preds = torch.argmax(outputs, dim=1)

            all_true.append(batch["y"].cpu().numpy())  # shape: (batch_size,)
            all_preds.append(preds.cpu().numpy())  # shape: (batch_size,)
            all_scores.append(outputs.cpu().numpy())  # shape: (batch_size, num_classes)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_score = np.concatenate(all_scores, axis=0)

    # np.savez(output_path, y_true=y_true, y_pred=y_pred, y_score=y_score)

    # print(f"Predictions saved to {output_path}")


model = model.eval()

# torch.save({"sbert": model.model.sbert.state_dict(), "classification": model.model.classification.state_dict()}, f"{base_folder}weights.pth")
# predict_and_save_predictions(model, test_dataloader, f"{base_folder}results.npz")
