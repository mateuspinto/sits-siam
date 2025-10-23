import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np

from lightly.loss import NegativeCosineSimilarity
from lightly.utils.debug import std_of_l2_normalized
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

#from sits_siam.backbone import TransformerBackbone
from sits_siam.utils import SitsFinetuneDatasetFromNpz
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
    LimitSequenceLength,
    ToPytorchTensor,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score

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
        return self.pe[positions]  # [batch, seq_len, d_model]

class SITSBertClassifier(nn.Module):
    def __init__(
        self,
        num_features=10,
        num_classes=13,
        seq_len=64,
        hidden=256,
        n_layers=3,
        n_heads=8,
        dropout=0.1,
        max_len=366
    ):
        super().__init__()
        # SBERT original: concat features + positional encoding
        self.embed_dim = hidden // 2

        self.input_proj = nn.Linear(num_features, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len)
        
        # O embedding final terá dimensão hidden (concatenação)
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
        # Max pooling para classificação
        self.pool = nn.AdaptiveMaxPool1d(1)
        # self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, doy, mask):
        """
        x: [batch, seq_len, num_features]
        doy: [batch, seq_len]
        mask: [batch, seq_len] (1=valido, 0=padding)
        """
        
        feat_emb = self.input_proj(x)                     # [B, L, hidden//2]
        pos_emb = self.pos_encoder(doy)                   # [B, L, hidden//2]
        embed = torch.cat([feat_emb, pos_emb], dim=-1)    # [B, L, hidden]
        # PyTorch usa mask: True=padded, False=valido
        #src_key_padding_mask = (mask == 0)                # [B, L]
        src_key_padding_mask = mask
        x_enc = self.transformer(embed, src_key_padding_mask=src_key_padding_mask)
        # pooling ao longo do tempo (seq_len)
        pooled = self.pool(x_enc.permute(0, 2, 1)).squeeze(-1) # [B, hidden]
        #logits = self.classifier(pooled)
        return pooled

class FastSiamMultiViewTransform(object):
    def __init__(
        self,
        n_views: int = 4,
    ):
        self.n_views = n_views
        self.transform = Pipeline(
            [
                IncreaseSequenceLength(140),
                RandomTempShift(),
                RandomAddNoise(),
                RandomTempRemoval(),
                RandomTempSwapping(max_distance=3),
                #RandomChanRemoval(0.1),
                #RandomChanSwapping(0.1),
                AddMissingMask(),
                Normalize(),
                ToPytorchTensor(),
            ]
        )

    def __call__(self, sample: np.ndarray):
        return [
            self.transform({k: v.copy() for k, v in sample.items()})
            for _ in range(self.n_views)
        ]

train_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/test.npz", transform=FastSiamMultiViewTransform(),)
val_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/val.npz", transform=FastSiamMultiViewTransform(),)

knn_transform = Pipeline(
    [
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

knn_train_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/train.npz", transform=knn_transform)
knn_val_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/val.npz", transform=knn_transform)

class TransformerClassifier(pl.LightningModule):
    def __init__(self, max_seq_len=140, max_epochs=100, batch_size=8*512, train_dataset_size=None, num_warmup_epochs=10, base_lr=1e-6, max_lr=1e-3, k=4, gamma=0.5):
        super(TransformerClassifier, self).__init__()
        self.model = SITSBertClassifier(num_classes=13)
        self.projection_head = SimSiamProjectionHead(256, 512, 1024)
        self.prediction_head = SimSiamPredictionHead(1024, 512, 1024)

        self.criterion = NegativeCosineSimilarity()

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

        f = self.model(x, doy, mask)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        views = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_collapse",
            std_of_l2_normalized(ps[0].detach()),
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
    # Pega o valor da LR do primeiro param group
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        views = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "val_collapse",
            std_of_l2_normalized(ps[0].detach()),
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        views = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_collapse", std_of_l2_normalized(ps[0].detach()), prog_bar=True)
        return loss

    def configure_optimizers(self):
            # Adicione weight_decay para regularização!
            optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr, weight_decay=1e-5) 
    
            train_steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
            total_training_steps = train_steps_per_epoch * self.max_epochs
            num_warmup_steps = train_steps_per_epoch * self.num_warmup_epochs
    
            def lr_lambda(current_step):
                # Fase de aquecimento (Warmup)
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                # Fase de decaimento (Cosine Decay)
                progress = float(current_step - num_warmup_steps) / float(max(1, total_training_steps - num_warmup_steps))
                # Garante que o progresso não exceda 1.0
                progress = min(progress, 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
    
            scheduler = LambdaLR(optimizer, lr_lambda)
            
            return [optimizer], [{
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }]

batch_size = 4 * 512
max_epochs = 200
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

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
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

                feats = pl_module.model(x, doy, mask)
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

                feats = pl_module.model(x, doy, mask)
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
        
mlflow_logger = MLFlowLogger(experiment_name="TEXASPRETRAIN")
knn_callback = KNNCallback(train_dataloader=knn_train_dataloader, val_dataloader=knn_val_dataloader, every_n_epochs=1, num_classes=13, k=5)


trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback, knn_callback], accelerator='gpu', precision="16-mixed", devices=[0, 1], logger=mlflow_logger)

model = TransformerClassifier(max_epochs=max_epochs,
    batch_size=batch_size,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=num_warmup_epochs,
    k=4,  # Divide o treinamento em 4 partes
    gamma=0.5  # Reduz LR pela metade a cada 25%
)

# trainer.validate(model=model, dataloaders=val_dataloader)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model = TransformerClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

# Saving pytorch model backbone state dict
torch.save(model.backbone.state_dict(), "weights/fastsiam_texas.pth")
