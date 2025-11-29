import math
import os
import random
import sys
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)

from torch.utils.data import WeightedRandomSampler

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
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz

torch.set_float32_matmul_precision('high')

BATCH_SIZE = 8 * 512
MAX_EPOCHS_P1 = 100
MAX_EPOCHS_P2 = 100

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

####################################### TRAINING #######################################

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

        return x_enc

transforms = Pipeline(
    [
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

aug_transforms = Pipeline(
    [
        LimitSequenceLength(140),
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

#train_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/train.npz", transform=transforms)
#val_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/val.npz", transform=transforms)
#test_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/test.npz", transform=transforms)


gdf = gpd.read_parquet("data/agl/gdf.parquet")

unique_mun = gdf['CD_MUN'].unique()

mun_train, mun_temp = train_test_split(
    unique_mun,
    test_size=0.30,
    random_state=42
)


mun_val, mun_test = train_test_split(
    mun_temp,
    test_size=0.50,
    random_state=42
)


gdf_train = gdf[gdf['CD_MUN'].isin(mun_train)].copy()
gdf_val = gdf[gdf['CD_MUN'].isin(mun_val)].copy()
gdf_test = gdf[gdf['CD_MUN'].isin(mun_test)].copy()

print(f"Municípios Treino: {len(mun_train)} - {len(gdf_train)} linhas")
print(f"Municípios Validação: {len(mun_val)} - {len(gdf_val)} linhas")
print(f"Municípios Teste: {len(mun_test)} - {len(gdf_test)} linhas")

train_dataset = AgriGEELiteDataset(
    gdf_train, "data/agl/df_sits.parquet", transform=aug_transforms, timestamp_processing="days_after_start"
)

val_dataset = AgriGEELiteDataset(
    gdf_val, "data/agl/df_sits.parquet", transform=transforms, timestamp_processing="days_after_start"
)

test_dataset = AgriGEELiteDataset(
    gdf_test, "data/agl/df_sits.parquet", transform=transforms, timestamp_processing="days_after_start"
)


class Phase1_Classifier(pl.LightningModule):
    def __init__(self, num_classes, train_dataset, max_epochs=100, batch_size=512, num_warmup_epochs=10, base_lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Arquitetura de Classificação
        self.backbone = SITSBertClassifier()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier_layer = nn.Linear(256, num_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.train_dataset_size = len(train_dataset)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, x, doy, mask):
        features = self.backbone(x, doy, mask)
        pooled = self.pool(features.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier_layer(pooled)

        return logits, pooled

    def training_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        logits, _ = self.forward(x, doy, mask)
        
        loss = self.criterion(logits, y)
        self.log("p1_train_loss", loss, prog_bar=True)
        self.log("p1_train_acc", self.train_acc(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        logits, _ = self.forward(x, doy, mask)
        
        loss = self.criterion(logits, y)
        self.log("p1_val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("p1_val_acc", self.val_acc(logits, y), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        num_warmup_steps = steps_per_epoch * self.num_warmup_epochs
        warmup_steps = num_warmup_steps 

        def lr_lambda(current_step):

            if current_step < warmup_steps:
                warmup_factor = 1.0 / 1000
                alpha = float(current_step) / float(max(1, warmup_steps))
                return warmup_factor * (1 - alpha) + alpha * 1.0
        
            else:
                decay_steps = total_steps - warmup_steps
                step_in_decay = current_step - warmup_steps

                progress = float(step_in_decay) / float(max(1, decay_steps))
                progress = min(1.0, progress) 

                return 0.5 * (1.0 + math.cos(math.pi * progress))
    
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }]


class Phase2_ConfidNet(pl.LightningModule):
    def __init__(self, pretrained_model: Phase1_Classifier, max_epochs=100, batch_size=512, train_dataset_size=None, num_warmup_epochs=10, base_lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])

        self.backbone = pretrained_model.backbone
        self.pool = pretrained_model.pool
        self.classifier_layer = pretrained_model.classifier_layer
        
        self.backbone.eval()
        self.classifier_layer.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier_layer.parameters():
            param.requires_grad = False

        self.confid_net = nn.Sequential(
            nn.Linear(256, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )
        
        self.mse_loss = nn.MSELoss()
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, x, doy, mask):

        with torch.no_grad():
            features = self.backbone(x, doy, mask)
            pooled = self.pool(features.permute(0, 2, 1)).squeeze(-1)
        
        confidence = self.confid_net(pooled)
        
        return confidence, pooled

    def training_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()

        pred_conf, pooled = self.forward(x, doy, mask)
        
        with torch.no_grad():
            self.classifier_layer.eval()
            
            logits = self.classifier_layer(pooled.detach()) 
            probs = F.softmax(logits, dim=1)
            
            tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()

        loss = self.mse_loss(pred_conf.squeeze(), tcp_target)
        
        self.log("p2_conf_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        
        pred_conf, pooled = self.forward(x, doy, mask)
        
        with torch.no_grad():
            logits = self.classifier_layer(pooled)
            probs = F.softmax(logits, dim=1)
            tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()
            
            loss = self.mse_loss(pred_conf.squeeze(), tcp_target)
            
        self.log("p2_val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        num_warmup_steps = steps_per_epoch * self.num_warmup_epochs
        warmup_steps = num_warmup_steps 

        def lr_lambda(current_step):

            if current_step < warmup_steps:
                warmup_factor = 1.0 / 1000
                alpha = float(current_step) / float(max(1, warmup_steps))
                return warmup_factor * (1 - alpha) + alpha * 1.0
        
            else:
                decay_steps = total_steps - warmup_steps
                step_in_decay = current_step - warmup_steps

                progress = float(step_in_decay) / float(max(1, decay_steps))
                progress = min(1.0, progress) 

                return 0.5 * (1.0 + math.cos(math.pi * progress))
    
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }]

sample_weights = train_dataset.get_weights_for_WeightedRandomSampler()

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True # Importante: permite repetir amostras das classes raras
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,   # <--- Adiciona o sampler aqui
    shuffle=False,     # <--- OBRIGATÓRIO: shuffle deve ser False ao usar sampler
    num_workers=12, 
    pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True
)

print("--- INICIANDO FASE 1: Classificação ---")
model_phase1 = Phase1_Classifier(
    num_classes=int(train_dataset.num_classes),
    max_epochs=MAX_EPOCHS_P1,
    batch_size=BATCH_SIZE,
    train_dataset=train_dataset
)

# Carregar pesos pré-treinados do SITS-BERT se houver
# model_phase1.backbone.load_state_dict(torch.load("siam_texas_new_bert.pth"))

mlflow_logger = MLFlowLogger(experiment_name="confidnet")

checkpoint_cb_p1 = ModelCheckpoint(monitor="p1_val_loss", filename="best_classifier", mode="min")
trainer_p1 = pl.Trainer(
    max_epochs=MAX_EPOCHS_P1, 
    accelerator="gpu", 
    callbacks=[checkpoint_cb_p1],
    logger=mlflow_logger
)

trainer_p1.fit(model_phase1, train_dataloader, val_dataloader)
best_model_p1 = Phase1_Classifier.load_from_checkpoint(checkpoint_cb_p1.best_model_path)

print("--- INICIANDO FASE 2: ConfidNet ---")

model_phase2 = Phase2_ConfidNet(
    pretrained_model=best_model_p1,
    max_epochs=MAX_EPOCHS_P2,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset)
)

checkpoint_cb_p2 = ModelCheckpoint(monitor="p2_val_loss", filename="best_confidnet", mode="min")
trainer_p2 = pl.Trainer(
    max_epochs=MAX_EPOCHS_P2, 
    accelerator="gpu", 
    callbacks=[checkpoint_cb_p2],
    # logger=mlflow_logger
)

trainer_p2.fit(model_phase2, train_dataloader, val_dataloader)

print("--- Gerando Predições Finais ---")
best_model_p2 = Phase2_ConfidNet.load_from_checkpoint(
    checkpoint_cb_p2.best_model_path, 
    pretrained_model=best_model_p1
)




def print_pretty_confusion_matrix(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    max_cell_width = max(len(str(np.max(cm))), 1)
    max_label_width = max(len(name) for name in class_names)
    max_header_height = max(len(name) for name in class_names)
    output = []
    
    true_label_pad = max(max_label_width, 4) 
    left_pad_width = true_label_pad + 3 
    left_padding_str = " " * left_pad_width
    
    matrix_content_width = len(class_names) * (max_cell_width + 2) - 2 

    output.append(f"{left_padding_str}{'Predicted':^{matrix_content_width}}")

    for i in range(max_header_height):
        row_str = left_padding_str
        cells = []
        for name in class_names:
            letter = name[i] if i < len(name) else " "
            cells.append(f"{letter:^{max_cell_width}}")
        row_str += "  ".join(cells)
        output.append(row_str)

    
    separator = "-" * matrix_content_width
    output.append(f"{'True':>{true_label_pad}} | {separator}")

    for i, row in enumerate(cm):
        row_label = class_names[i]
        row_str = f"{row_label:>{true_label_pad}} |"
        cells = []
        for val in row:
            cells.append(f"{val:>{max_cell_width}}")
        row_str += "  ".join(cells)
        output.append(row_str)

    print("\n".join(output))


print("\n--- Validating Phase 2 Best Model ---")
trainer_p2.validate(model=best_model_p2, dataloaders=val_dataloader)

print("\n--- Loading Best Phase 2 Model for Inference ---")
best_model_p2.eval()

def predict_and_save_predictions(model, dataloader, dataset, device="cuda"):
    model.eval()
    model.to(device)

    all_true = []
    all_preds = []
    all_proba_max = []
    all_conf = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Predicting"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            x, doy, mask = batch["x"], batch["doy"], batch["mask"]
            
            confidence, pooled = model(x, doy, mask)

            logits = model.classifier_layer(pooled)

            preds = torch.argmax(logits, dim=1)
            proba = F.softmax(logits, dim=1)
            proba_max = proba.max(dim=1).values

            all_true.append(batch["y"].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_proba_max.append(proba_max.cpu().numpy())
            all_conf.append(confidence.cpu().numpy())

    y_true_int = np.concatenate(all_true, axis=0).squeeze()
    y_pred_int = np.concatenate(all_preds, axis=0)
    y_proba_max = np.concatenate(all_proba_max, axis=0)
    y_conf = np.concatenate(all_conf, axis=0).squeeze()

    class_names = dataset.get_class_names() if hasattr(dataset, "get_class_names") else train_dataset.get_class_names()
    class_map = {i: name for i, name in enumerate(class_names)}
    y_true_names = np.array([class_map[num] for num in y_true_int])
    y_pred_names = np.array([class_map[num] for num in y_pred_int])

    acc = accuracy_score(y_true_names, y_pred_names)
    f1_macro = f1_score(y_true_names, y_pred_names, average="macro")
    f1_micro = f1_score(y_true_names, y_pred_names, average="micro")

    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "test_accuracy", acc)
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "test_f1_macro", f1_macro)
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "test_f1_micro", f1_micro)

    def compute_failure_metrics(y_true_correct, y_confidence, prefix):
        y_true_error = (~y_true_correct).astype(int)
        y_true_success = y_true_correct.astype(int)
        
        y_score_error = 1.0 - y_confidence

        auroc = roc_auc_score(y_true_error, y_score_error)

        precision_err, recall_err, _ = precision_recall_curve(y_true_error, y_score_error)
        aupr_error = auc(recall_err, precision_err)

        precision_succ, recall_succ, _ = precision_recall_curve(y_true_success, y_confidence)
        aupr_success = auc(recall_succ, precision_succ)

        fpr, tpr, _ = roc_curve(y_true_error, y_score_error)
        idx = np.argmax(tpr >= 0.95)
        fpr_at_95_tpr = fpr[idx] if idx < len(fpr) else fpr[-1]

        print(f"\n--- Failure Prediction Metrics ({prefix}) ---")
        print(f"AUC: {auroc:.4f}")
        print(f"AUPR-Error: {aupr_error:.4f}")
        print(f"AUPR-Success: {aupr_success:.4f}")
        print(f"FPR-95%-TPR: {fpr_at_95_tpr:.4f}")

        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"{prefix}_AUC", auroc)
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"{prefix}_AUPR_Error", aupr_error)
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"{prefix}_AUPR_Success", aupr_success)
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"{prefix}_FPR_95_TPR", fpr_at_95_tpr)

    is_correct = (y_true_int == y_pred_int)

    compute_failure_metrics(is_correct, y_conf, prefix="conf")

    compute_failure_metrics(is_correct, y_proba_max, prefix="mcp")

    print("\n--- Classification Report ---")
    print(classification_report(y_true_names, y_pred_names))

    if hasattr(dataset, "gdf"):
        print("Merging predictions with original GeoDataFrame...")
        final_gdf = dataset.gdf.copy()
        
        if len(final_gdf) != len(y_pred_names):
            print(f"WARNING: Dataset length ({len(final_gdf)}) differs from predictions ({len(y_pred_names)}).")
            print("Falling back to saving only predictions DataFrame.")
            final_df = pd.DataFrame({
                "y_true": y_true_names,
                "y_pred": y_pred_names,
                "y_proba": y_proba_max,
                "y_conf": y_conf
            })
            filename = "predictions.parquet"
        else:
            final_gdf["pred_class"] = y_pred_names
            final_gdf["pred_proba"] = y_proba_max
            final_gdf["pred_conf"] = y_conf
            final_gdf["true_class"] = y_true_names
            
            final_df = final_gdf
            filename = "predictions_geo.parquet"
            
    else:
        print("Dataset does not have 'gdf' attribute. Saving simple predictions DataFrame.")
        final_df = pd.DataFrame({
            "y_true": y_true_names,
            "y_pred": y_pred_names,
            "y_proba": y_proba_max,
            "y_conf": y_conf
        })
        filename = "predictions.parquet"

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, filename)
        final_df.to_parquet(file_path)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)
        print(f"Artifact saved: {filename}")

predict_and_save_predictions(
    best_model_p2, 
    test_dataloader, 
    test_dataset
)
