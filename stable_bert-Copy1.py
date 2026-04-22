import math
import os
import random
import sys
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report, brier_score_loss
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)
from tqdm.std import tqdm

from sits_siam.augment import (
    AddMissingMask,
    IncreaseSequenceLength,
    LimitSequenceLength,
    Pipeline,
    ToPytorchTensor,
    Normalize,
)
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, positions):
        return self.pe[positions]

class SITSBertClassifier(nn.Module):
    def __init__(
        self,
        num_features=10,
        num_classes=13, # not used in pre-training
        seq_len=64,
        hidden=256,
        n_layers=3,
        n_heads=8,
        dropout=0.1,
        max_len=366
    ):
        super().__init__()
        self.embed_dim = hidden // 2
        self.input_proj = nn.Linear(num_features, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=n_layers
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, doy, mask):
        feat_emb = self.input_proj(x)
        pos_emb = self.pos_encoder(doy)
        embed = torch.cat([feat_emb, pos_emb], dim=-1)
        src_key_padding_mask = mask
        x_enc = self.transformer(embed, src_key_padding_mask=src_key_padding_mask)
        # Pooling foi movido para fora para que o KNN callback possa usar o backbone diretamente
        return x_enc
        
def setup_seed():
    seed = 42

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
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

train_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/train.npz", transform=transforms)
val_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/val.npz", transform=transforms)
test_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/test.npz", transform=transforms)


class OriginalSITSBertTrainer(pl.LightningModule):
    def __init__(self, max_seq_len=127, num_classes=int(train_dataset.num_classes), 
                 max_epochs=100, batch_size=8*512, train_dataset_size=None,
                 num_warmup_epochs=10, base_lr=1e-6, max_lr=1e-3, k=4, gamma=0.5):
        super(OriginalSITSBertTrainer, self).__init__()
        self.backbone = SITSBertClassifier()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier_layer = nn.Linear(256, num_classes)
        
        self.criterion = nn.CrossEntropyLoss(weight=train_dataset.get_class_weights())
        self.train_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        
        # Parâmetros de treinamento
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.k = k  # Número de divisões do treinamento para redução de LR
        self.gamma = gamma  # Fator de redução do LR

    def forward(self, input):
        x = input["x"]
        doy = input["doy"]
        mask = input["mask"]

        outputs = self.backbone(x, doy, mask)
        # pooling ao longo do tempo (seq_len)
        pooled = self.pool(outputs.permute(0, 2, 1)).squeeze(-1) # [B, hidden]
        logits = self.classifier_layer(pooled)
        
        return logits

    def training_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        train_oa_score = self.train_oa(outputs, targets)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_oa", train_oa_score, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        val_oa_score = self.val_oa(outputs, targets)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_oa", val_oa_score, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
    # Pega o valor da LR do primeiro param group
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        test_oa_score = self.test_oa(outputs, targets)
        self.test_cm(outputs, targets)

        # Log loss and oa score
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_oa", test_oa_score, sync_dist=True)
        #self.log("test_f1", test_f1_score, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        self.confusion_matrix = self.test_cm.compute()
        self.test_cm.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        train_steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_training_steps = train_steps_per_epoch * self.max_epochs
        num_warmup_steps = train_steps_per_epoch * self.num_warmup_epochs
        
        # define warmup parameters equivalentes ao código anterior
        warmup_factor = 1.0 / 1000        # começa com 1/1000 do LR
        warmup_iters = min(1000, num_warmup_steps - 1)  # igual ao n_iters - 1
    
        def lr_lambda(current_step):
            if current_step < warmup_iters:
                # Interpola linearmente de warmup_factor até 1.0
                alpha = current_step / float(max(1, warmup_iters))
                return warmup_factor * (1 - alpha) + alpha * 1.0
            # Após o warmup, mantém LR constante (1.0 × max_lr)
            return 1.0
    
        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }]

batch_size = 8 * 512
max_epochs = 200 #200
num_warmup_epochs = 20 #20

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True
)

mlflow_logger = MLFlowLogger(experiment_name="texas")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)

  
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu",
    precision="16-mixed",
    # devices=[0, 1],
    callbacks=[checkpoint_callback],
    logger=mlflow_logger
)

model = OriginalSITSBertTrainer(
    max_epochs=max_epochs,
    batch_size=batch_size,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=num_warmup_epochs,
    k=4,  # Divide o treinamento em 4 partes
    gamma=0.5  # Reduz LR pela metade a cada 25%
)

model.backbone.load_state_dict(torch.load("siam_texas_new_bert.pth"))
    
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)


def predict_and_save_predictions(
    model, dataloader, device="cuda"
):
    model.eval()
    model.to(device)

    all_true = []
    all_preds = []
    all_proba = []

    with torch.inference_mode():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch)

            preds = torch.argmax(outputs, dim=1)
            proba = F.softmax(outputs, dim=1)
            
            all_true.append(batch["y"].cpu().numpy())  # shape: (batch_size,)
            all_preds.append(preds.cpu().numpy())  # shape: (batch_size,)
            all_proba.append(proba.cpu().numpy()) # shape: (batch_size, num_classes)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_proba = np.concatenate(all_proba, axis=0)


    class_names = train_dataset.get_class_names()
    y_true = np.array([class_names[num] for num in y_true])
    y_pred = np.array([class_names[num] for num in y_pred])

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    # brier_score = brier_score_loss(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("f1_micro", f1_micro)
    print("f1_macro", f1_macro)
    # print("brier score", brier_score)
    print(report)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "results.npz")
        np.savez(file_path, y_true=y_true, y_pred=y_pred, y_proba=y_proba)

        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

trainer.validate(model=model, dataloaders=val_dataloader)
model = OriginalSITSBertTrainer.load_from_checkpoint(checkpoint_callback.best_model_path)
model = model.eval()

predict_and_save_predictions(model, test_dataloader)
