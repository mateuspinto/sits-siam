import random
import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score
from torch.optim import AdamW, Adam
import numpy as np

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

from print_color import print

from sits_siam.backbone import TransformerBackbone
from sits_siam.head import BertHead, ClassifierHead
from sits_siam.utils import SitsDataset
from sits_siam.bottleneck import PoolingBottleneck, NDVIWord2VecBottleneck
from sits_siam.augment import (
    AddNDVIWeights,
    RandomChanSwapping,
    RandomChanRemoval,
    RandomAddNoise,
    RandomTempSwapping,
    RandomTempShift,
    RandomTempRemoval,
    AddMissingMask,
    Normalize,
    Pipeline,
    ToPytorchTensor,
)

median = [
    0.0656,
    0.0948,
    0.1094,
    0.1507,
    0.2372,
    0.2673,
    0.2866,
    0.2946,
    0.2679,
    0.1985,
]
iqd = [0.0456, 0.0536, 0.0946, 0.0769, 0.0851, 0.1053, 0.1066, 0.1074, 0.1428, 0.1376]

train_transforms = Pipeline(
    [
        # AddNDVIWeights(),
        RandomChanSwapping(),
        RandomChanRemoval(),
        RandomAddNoise(0.02),
        RandomTempSwapping(max_distance=3),
        RandomTempShift(),
        AddMissingMask(),
        Normalize(
            a=median,
            b=iqd,
        ),
        ToPytorchTensor(),
    ]
)

val_transforms = Pipeline(
    [
        # AddNDVIWeights(),
        AddMissingMask(),
        Normalize(
            a=median,
            b=iqd,
        ),
        ToPytorchTensor(),
    ]
)

whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")

train_df = whole_df[whole_df.use_bert == 2].reset_index(drop=True)
val_df = whole_df[whole_df.use_bert == 1].reset_index(drop=True)
test_df = whole_df[whole_df.use_bert == 0].reset_index(drop=True)

print(f"Train df={len(train_df)}, Val df={len(val_df)}, Test df={len(test_df)}")
train_dataset = SitsDataset(train_df, max_seq_len=45, transform=train_transforms)
val_dataset = SitsDataset(val_df, max_seq_len=45, transform=val_transforms)
test_dataset = SitsDataset(test_df, max_seq_len=45, transform=val_transforms)

del train_df
del val_df
del test_df
del whole_df


class TransformerClassifier(pl.LightningModule):
    def __init__(self, max_seq_len=40, num_classes=13):
        super(TransformerClassifier, self).__init__()
        self.backbone = TransformerBackbone(max_seq_len=max_seq_len)
        # self.backbone = torch.load("backbone.pt", map_location=torch.device('cpu'))
        self.bottleneck = PoolingBottleneck()
        self.classifier = ClassifierHead(num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, input):
        x = input["x"]
        doy = input["doy"]
        mask = input["mask"]
        # weight = input["weight"]

        features = self.backbone(x, doy, mask)
        features = self.bottleneck(features)
        outputs = self.classifier(features)
        return outputs

    def training_step(self, batch, batch_idx):
        targets = batch["y"]
        outputs = self(batch)

        loss = self.criterion(outputs, targets)

        # Log loss and F1 score
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["y"]
        outputs = self(batch)
        loss = self.criterion(outputs, targets)

        # Calculate F1 score
        preds = torch.argmax(outputs, dim=1)
        f1_score = self.val_f1(preds, targets)

        # Log loss and F1 score
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        targets = batch["y"]
        outputs = self(batch)
        loss = self.criterion(outputs, targets)

        # Calculate F1 score
        preds = torch.argmax(outputs, dim=1)
        f1_score = self.test_f1(preds, targets)

        # Log loss and F1 score
        self.log("test_loss", loss)
        self.log("test_f1", f1_score)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        return optimizer


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1024, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1024, shuffle=False
)
test_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1024, shuffle=False
)

early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)


trainer = pl.Trainer(
    max_epochs=100,
    # log_every_n_steps=5,
    devices="auto",
    # accelerator="gpu",
    # strategy="ddp",
    # sync_batchnorm=False,
    # use_distributed_sampler=True,
    # precision='16-mixed',
    callbacks=[checkpoint_callback],
)

model = TransformerClassifier()

trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

model = TransformerClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
model = model.eval()

trainer.test(model=model, dataloaders=test_dataloader)
