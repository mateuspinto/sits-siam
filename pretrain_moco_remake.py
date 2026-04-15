import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import math
import copy

# Lightly imports for MoCo
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from lightly.utils.debug import std_of_l2_normalized


from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score

# --- Seus imports e classes de modelo/dataset ---
# (Mantive suas classes originais inalteradas, pois elas formam a base)
from sits_siam.utils import SitsFinetuneDatasetFromNpz
from sits_siam.augment import (
    RandomAddNoise,
    RandomTempSwapping,
    RandomTempShift,
    RandomTempRemoval,
    AddMissingMask,
    Normalize,
    Pipeline,
    IncreaseSequenceLength,
    LimitSequenceLength,
    ToPytorchTensor,
)

from torch.optim.lr_scheduler import LambdaLR

# disable scientific notation pytorch, keep 3 numbers after decimal
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_float32_matmul_precision('high')

# ==============================================================================
# 1. SUAS CLASSES DE MODELO E POSITIONAL ENCODING (SEM ALTERAÇÕES)
# ==============================================================================

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

# ==============================================================================
# 2. NOVA TRANSFORMAÇÃO PARA O MOCO (GERA 2 VIEWS)
# ==============================================================================
class MoCoSITSTransform:
    def __init__(self):
        # A pipeline de aumento de dados é aplicada de forma independente para cada view
        self.transform = Pipeline(
            [
                IncreaseSequenceLength(140),
                RandomTempShift(),
                RandomAddNoise(),
                RandomTempRemoval(),
                RandomTempSwapping(max_distance=3),
                AddMissingMask(),
                Normalize(),
                ToPytorchTensor(),
            ]
        )

    def __call__(self, sample: np.ndarray):
        # Cria duas views (query e key) da mesma amostra
        view1 = self.transform({k: v.copy() for k, v in sample.items()})
        view2 = self.transform({k: v.copy() for k, v in sample.items()})
        return [view1, view2]


# ==============================================================================
# 3. NOVO MÓDULO PYTORCH LIGHTNING PARA O MOCO
# ==============================================================================
class SITS_MoCo(pl.LightningModule):
    def __init__(self, max_seq_len=140, max_epochs=100, batch_size=8*512, train_dataset_size=None, num_warmup_epochs=10, base_lr=1e-6, max_lr=1e-4, k=4, gamma=0.5):
        super().__init__()
        
        # Backbone (Query Encoder)
        self.backbone = SITSBertClassifier(hidden=256) # Sua arquitetura de modelo
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # MoCo requer uma cabeça de projeção
        # A entrada (256) deve ser igual à saída do seu backbone
        self.projection_head = MoCoProjectionHead(input_dim=256, hidden_dim=512, output_dim=128)

        # Momentum Encoder (Key Encoder)
        # É uma cópia do encoder principal, mas seus pesos são atualizados de forma diferente
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Desativa o cálculo de gradientes para o momentum encoder
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Critério de perda contrastiva
        self.criterion = NTXentLoss(memory_bank_size=(4096, 128))

        # Parâmetros de treinamento
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.k = k  # Número de divisões do treinamento para redução de LR
        self.gamma = gamma  # Fator de redução do LR

    def _forward_features(self, sample, model):
        """Helper para passar os dados pelo backbone e fazer o pooling."""
        x = sample["x"]
        doy = sample["doy"]
        mask = sample["mask"]
        features = model(x, doy, mask)
        pooled_features = self.pool(features.permute(0, 2, 1)).squeeze(-1)
        return pooled_features

    def forward(self, x):
        """Forward pass para o Query Encoder"""
        features = self._forward_features(x, self.backbone)
        query = self.projection_head(features)
        return query

    def forward_momentum(self, x):
        """Forward pass para o Key Encoder (Momentum)"""
        features = self._forward_features(x, self.backbone_momentum)
        key = self.projection_head_momentum(features)
        return key.detach() # Importante: destacar para não propagar gradientes

    def training_step(self, batch, batch_idx):
        # 1. Atualiza os pesos do encoder de momentum
        momentum = cosine_schedule(self.current_epoch, max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        
        # O dataloader retorna uma lista com duas views [view1, view2]
        view1, view2 = batch

        # 2. Gera query e key
        query = self.forward(view1)
        key = self.forward_momentum(view2)

        # 3. Calcula a perda
        loss = self.criterion(query, key)
        
        # 4. Logging
        self.log("train_loss", loss, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # O colapso é menos comum no MoCo, mas é bom monitorar a std das features
        # Para isso, precisamos de um forward extra
        self.backbone.eval()
        with torch.no_grad():
            # Pega um batch do dataloader de validação para medir o colapso
            val_loader = self.trainer.val_dataloaders
            if val_loader:
                batch = next(iter(val_loader))
                view1, _ = batch
                view1 = {k: v.to(self.device) for k, v in view1.items()}
                query = self.forward(view1)
                collapse_level = std_of_l2_normalized(query)
                self.log("collapse_level", collapse_level, prog_bar=True)
        self.backbone.train()

    def on_train_epoch_end(self):
        # Pega o valor da LR do primeiro param group
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # O dataloader de validação também deve retornar duas views
        view1, view2 = batch

        # 1. Gera query e key
        # Não há atualização de momentum aqui, usamos os encoders como estão
        query = self.forward(view1)
        key = self.forward_momentum(view2)

        # 2. Calcula a perda de validação
        loss = self.criterion(query, key)

        # 3. Faz o log da perda e da métrica de colapso
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "val_collapse",
            std_of_l2_normalized(query.detach()), # Usamos detach() pois não precisamos de gradientes
            prog_bar=True,
            sync_dist=True,
        )
        return loss
        
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
        # optimizer = Lamb(self.parameters(), lr=self.max_lr)

        train_steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_training_steps = train_steps_per_epoch * self.max_epochs
        num_warmup_steps = train_steps_per_epoch * self.num_warmup_epochs
        
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, num_warmup_steps - 1)
    
        def lr_lambda(current_step):
            if current_step < warmup_iters:
                alpha = current_step / float(max(1, warmup_iters))
                return warmup_factor * (1 - alpha) + alpha * 1.0
            return 1.0
    
        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }]


# ==============================================================================
# 4. CONFIGURAÇÃO DOS DADOS E DATALOADERS
# ==============================================================================
train_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/test.npz", transform=MoCoSITSTransform())
val_dataset = SitsFinetuneDatasetFromNpz("data/texas_001_001_998/val.npz", transform=MoCoSITSTransform())


# Transform para o KNN: sem data augmentation, apenas normalização e padding
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

batch_size = 256 # MoCo beneficia-se de batch sizes maiores
max_epochs = 80
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

# ==============================================================================
# 5. KNN CALLBACK ATUALIZADO
# ==============================================================================
class KNNCallback(pl.Callback):
    def __init__(self, train_dataloader, val_dataloader, every_n_epochs=10, k=5):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.every_n_epochs = every_n_epochs
        self.k = k

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        device = pl_module.device

        # Extrair embeddings do backbone
        def get_embeddings(dataloader):
            feats_list, labels_list = [], []
            with torch.no_grad():
                for batch in dataloader:
                    x = batch["x"].to(device)
                    doy = batch["doy"].to(device)
                    mask = batch["mask"].to(device)
                    y = batch["y"].to(device)

                    # MUDANÇA: Usar pl_module.backbone diretamente
                    feats = pl_module.backbone(x, doy, mask) 
                    feats = pl_module.pool(feats.permute(0, 2, 1)).squeeze(-1)
                    
                    feats_list.append(feats.cpu().numpy())
                    labels_list.append(y.cpu().numpy())
            return np.concatenate(feats_list), np.concatenate(labels_list)

        train_feats, train_labels = get_embeddings(self.train_dataloader)
        val_feats, val_labels = get_embeddings(self.val_dataloader)

        # Treinar e avaliar o KNN
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(train_feats, train_labels)
        preds = knn.predict(val_feats)

        # Métricas
        acc = accuracy_score(val_labels, preds)
        f1_macro = f1_score(val_labels, preds, average="macro", zero_division=0)
        
        pl_module.log("knn_acc", acc, prog_bar=True, logger=True)
        pl_module.log("knn_f1_macro", f1_macro, prog_bar=True, logger=True)
        
        pl_module.train()

# ==============================================================================
# 6. TREINAMENTO
# ==============================================================================
checkpoint_callback = ModelCheckpoint(
    monitor="knn_f1_macro", filename="best_moco_model", save_top_k=1, mode="max" # F1-macro maior é melhor
)
knn_callback = KNNCallback(train_dataloader=knn_train_dataloader, val_dataloader=knn_val_dataloader, every_n_epochs=1)
mlflow_logger = MLFlowLogger(experiment_name="TEXAS_MOCO_PRETRAIN")

trainer = pl.Trainer(
    max_epochs=max_epochs, 
    callbacks=[checkpoint_callback, knn_callback], 
    accelerator='gpu', 
    precision="bf16-mixed", 
    devices=[0, 1],
    strategy="ddp", # MoCo funciona muito bem com DDP
    logger=mlflow_logger
)

model = SITS_MoCo(max_epochs=max_epochs,
    batch_size=batch_size,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=num_warmup_epochs,)

trainer.validate(model=model, dataloaders=val_dataloader)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model = SITS_MoCo.load_from_checkpoint(checkpoint_callback.best_model_path)

torch.save(model.backbone.state_dict(), "moco_texas_new_bert.pth")
