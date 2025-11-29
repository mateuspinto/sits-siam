import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np

from lightly.loss import NegativeCosineSimilarity
from lightly.utils.debug import std_of_l2_normalized
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

#from sits_siam.backbone import TransformerBackbone
from sits_siam.utils import SitsFinetuneDatasetFromNpz, SitsPretrainDatasetFromNpz
#from sits_siam.bottleneck import PoolingBottleneck
from sits_siam.augment import (
    RandomAddNoise,
    RandomTempSwapping,
    RandomTempShift,
    RandomTempRemoval,
    RandomCloudAugmentation,
    RandomChanSwapping,
    RandomChanRemoval,
    AddMissingMask,
    Normalize,
    Pipeline,
    IncreaseSequenceLength,
    AddCorruptedSample,
    LimitSequenceLength,
    ToPytorchTensor,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score

# from pytorch_optimizer import Lamb

# disable scientific notation pytorch, keep 3 numbers after decimal
torch.set_printoptions(precision=3, sci_mode=False)

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

aug_transform = Pipeline(
    [
        LimitSequenceLength(127),
        IncreaseSequenceLength(127),
        # RandomAddNoise(0.02), #
        RandomTempSwapping(max_distance=3), #
        RandomTempShift(), #
        RandomTempRemoval(), #
        AddCorruptedSample(),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

val_transform = Pipeline(
    [
        LimitSequenceLength(127),
        IncreaseSequenceLength(127),
        # RandomAddNoise(0.02), #
        # RandomTempSwapping(max_distance=3), #
        # RandomTempShift(), #
        # RandomTempRemoval(), #
        AddCorruptedSample(),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

# train_dataset = SitsFinetuneDatasetFromNpz("data/california_001_001_998/test.npz", transform=aug_transform)
train_dataset = SitsPretrainDatasetFromNpz("../pretrain_from_former", world_size=2, transform=val_transform)
val_dataset = SitsFinetuneDatasetFromNpz("data/california_001_001_998/val.npz", transform=val_transform)


knn_train_dataset = SitsFinetuneDatasetFromNpz("data/california_001_001_998/train.npz", transform=val_transform)
knn_val_dataset = SitsFinetuneDatasetFromNpz("data/california_001_001_998/val.npz", transform=val_transform)

class TransformerClassifier(pl.LightningModule):
    def __init__(self, max_seq_len=140, max_epochs=100, batch_size=8*512, train_dataset_size=None, num_warmup_epochs=10, base_lr=1e-6, max_lr=1e-4, k=4, gamma=0.5):
        super(TransformerClassifier, self).__init__()
        self.backbone = SITSBertClassifier()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output_head = nn.Linear(256, 10)

        self.criterion = nn.MSELoss(reduction="none")

        # Parâmetros de treinamento
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.k = k  # Número de divisões do treinamento para redução de LR
        self.gamma = gamma  # Fator de redução do LR

    def forward(self, batch):
        x = batch["x"]
        doy = batch["doy"]
        mask = batch["mask"]

        f = self.backbone(x, doy, mask)
        f = self.pool(x_enc.permute(0, 2, 1)).squeeze(-1) # [B, hidden]
        return f

    def forward_corrupted(self, batch):
        x = batch["corrupted_x"]
        doy = batch["doy"]
        mask = batch["mask"]

        f = self.backbone(x, doy, mask)
        f = self.output_head(f)
        return f

    def training_step(self, batch, batch_idx):
        pred = self.forward_corrupted(batch)
        # Target: batch["x"]
        loss = self.criterion(pred, batch["x"].float())   # [B, num_features]
        mask = batch["corrupted_mask"].unsqueeze(-1)      # [B, 1]
        loss = (loss * mask.float()).sum() / mask.sum()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Pega o valor da LR do primeiro param group
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pred = self.forward_corrupted(batch)
        loss = self.criterion(pred, batch["x"].float())   # [B, seq_len, num_features]
        mask = batch["corrupted_mask"].unsqueeze(-1)      # [B, seq_len, 1]
        loss = (loss * mask.float()).sum() / mask.sum()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred = self.forward_corrupted(batch)
        loss = self.criterion(pred, batch["x"].float())
        mask = batch["corrupted_mask"].unsqueeze(-1)
        loss = (loss * mask.float()).sum() / mask.sum()
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(
        #    self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
        #)

        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
        # optimizer = Lamb(self.parameters(), lr=1e-4)

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

batch_size = 4 * 512
max_epochs = 300
num_warmup_epochs = 20

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

knn_train_dataloader = torch.utils.data.DataLoader(
    knn_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
knn_val_dataloader = torch.utils.data.DataLoader(
    knn_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

class KNNCallback(pl.Callback):
    def __init__(self, train_dataloader, val_dataloader, every_n_epochs=10, num_classes=13, k=5):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.every_n_epochs = every_n_epochs
        self.num_classes = num_classes
        self.k = k


    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        device = pl_module.device

        # 1. Extrair embeddings do conjunto de treino
        train_feats = []
        train_labels = []
        with torch.no_grad():
            for batch in self.train_dataloader:
                x = batch["x"].to(device)
                doy = batch["doy"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y"].to(device)

                feats = pl_module.backbone(x, doy, mask)
                feats = pl_module.pool(feats.permute(0, 2, 1)).squeeze(-1)
                train_feats.append(feats.cpu().numpy())
                train_labels.append(y.cpu().numpy())
        train_feats = np.concatenate(train_feats, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # 2. Extrair embeddings do conjunto de validação
        val_feats = []
        val_labels = []
        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch["x"].to(device)
                doy = batch["doy"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y"].to(device)

                feats = pl_module.backbone(x, doy, mask)
                feats = pl_module.pool(feats.permute(0, 2, 1)).squeeze(-1)
                val_feats.append(feats.cpu().numpy())
                val_labels.append(y.cpu().numpy())
        val_feats = np.concatenate(val_feats, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        # 3. Treinar KNN nos embeddings de treino, avaliar nos de validação
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(train_feats, train_labels)
        preds = knn.predict(val_feats)

        # 4. Métricas
        acc = accuracy_score(val_labels, preds)
        f1_macro = f1_score(val_labels, preds, average="macro", zero_division=0)
        f1_micro = f1_score(val_labels, preds, average="micro", zero_division=0)

        pl_module.log("knn_acc", acc, prog_bar=True, logger=True, sync_dist=True)
        pl_module.log("knn_f1_macro", f1_macro, prog_bar=True, logger=True, sync_dist=True)
        pl_module.log("knn_f1_micro", f1_micro, prog_bar=True, logger=True, sync_dist=True)
        
        pl_module.train()
        
mlflow_logger = MLFlowLogger(experiment_name="CALIFORNIARECONSTRUCT")
knn_callback = KNNCallback(train_dataloader=knn_train_dataloader, val_dataloader=knn_val_dataloader, every_n_epochs=2, num_classes=13, k=5)
checkpoint_callback = ModelCheckpoint(
    monitor="knn_f1_macro", filename="best_model", save_top_k=1, mode="max"
)

trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[knn_callback, checkpoint_callback], accelerator='gpu', precision="16-mixed", devices=[0, 1], logger=mlflow_logger, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
model = TransformerClassifier(max_epochs=max_epochs,
    batch_size=batch_size,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=num_warmup_epochs,
    k=4,  # Divide o treinamento em 4 partes
    gamma=0.5  # Reduz LR pela metade a cada 25%
)

trainer.validate(model=model, dataloaders=val_dataloader)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model = TransformerClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

torch.save(model.backbone.state_dict(), "bert_all_new_bert.pth")

# Saving pytorch model backbone state dict
# torch.save(model.backbone.state_dict(), "weights/fastsiam_texas.pth")
