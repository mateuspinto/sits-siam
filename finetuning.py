import math

import geopandas as gpd
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import WeightedRandomSampler
from torchmetrics.classification import MulticlassAccuracy

from sits_siam.augment import (
    AddMissingMask,
    IncreaseSequenceLength,
    LimitSequenceLength,
    Normalize,
    Pipeline,
    RandomAddNoise,
    RandomTempRemoval,
    RandomTempShift,
    RandomTempSwapping,
    ToPytorchTensor,
)
from sits_siam.auxiliar import (
    beautify_prints,
    predict_and_save_predictions,
    setup_seed,
    run_gemos,
)
from sits_siam.models import SITSBert, SITSBertPlusPlus
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz

patch_sklearn()
torch.set_float32_matmul_precision("high")
setup_seed()
beautify_prints()

DATASET = "brazil"
TRAIN_SIZE = 70
BATCH_SIZE = 2 * 512
MAX_EPOCHS_P1 = 100
MAX_EPOCHS_P2 = 100
NUM_WARMUP_EPOCHS = 10
BASE_LR = 1e-4
PRETRAIN_PATH = ""

TAGS = {
    "dataset": DATASET,
    "batch_size": BATCH_SIZE,
    "max_epochs_p1": MAX_EPOCHS_P1,
    "max_epochs_p2": MAX_EPOCHS_P2,
    "pretrain_path": PRETRAIN_PATH,
    "num_warmup_epochs": NUM_WARMUP_EPOCHS,
    "base_lr": BASE_LR,
}
RUN_NAME = f"BERTPPSCRATCH-{TRAIN_SIZE}"


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

# train_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/train.npz", transform=transforms)
# val_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/val.npz", transform=transforms)
# test_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/test.npz", transform=transforms)


gdf = gpd.read_parquet("/home/m/Downloads/gdf.parquet")

class_map = (
    gdf[["crop_class", "crop_number"]]
    .drop_duplicates()
    .set_index("crop_number")["crop_class"]
    .to_dict()
)

unique_mun = gdf["CD_MUN"].unique()

mun_train, mun_temp = train_test_split(unique_mun, test_size=0.30, random_state=13)

mun_val, mun_test = train_test_split(mun_temp, test_size=0.50, random_state=13)

gdf_train = gdf[gdf["CD_MUN"].isin(mun_train)].copy()
gdf_val = gdf[gdf["CD_MUN"].isin(mun_val)].copy()
gdf_test = gdf[gdf["CD_MUN"].isin(mun_test)].copy()

print("Distribuição das classes - Treino")
print(gdf_train.crop_class.value_counts())

print("Distribuição das classes - Validação")
print(gdf_val.crop_class.value_counts())

print("Distribuição das classes - Teste")
print(gdf_test.crop_class.value_counts())

print(f"Municípios Treino: {len(mun_train)} - {len(gdf_train)} linhas")
print(f"Municípios Validação: {len(mun_val)} - {len(gdf_val)} linhas")
print(f"Municípios Teste: {len(mun_test)} - {len(gdf_test)} linhas")

train_dataset = AgriGEELiteDataset(
    gdf_train,
    "/home/m/Downloads/df_sits.parquet",
    transform=aug_transforms,
    timestamp_processing="days_after_start",
)

val_dataset = AgriGEELiteDataset(
    gdf_val,
    "/home/m/Downloads/df_sits.parquet",
    transform=transforms,
    timestamp_processing="days_after_start",
)

test_dataset = AgriGEELiteDataset(
    gdf_test,
    "/home/m/Downloads/df_sits.parquet",
    transform=transforms,
    timestamp_processing="days_after_start",
)


class Phase1_Classifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        train_dataset_size,
        max_epochs=100,
        batch_size=512,
        num_warmup_epochs=10,
        base_lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # self.backbone = SITSBert(num_classes=num_classes)
        self.backbone = SITSBertPlusPlus(num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted")

        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, x, doy, mask):
        pooled, logits, _ = self.backbone(x, doy, mask)

        return logits, pooled

    def training_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        logits, _ = self.forward(x, doy, mask)

        loss = self.criterion(logits, y)
        self.log("p1_train_loss", loss, prog_bar=True)
        self.log("p1_train_acc", self.train_acc(logits, y), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("p1_lr", lr, prog_bar=True, on_epoch=True, sync_dist=True)

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

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]


class Phase2_ConfidNet(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: Phase1_Classifier,
        train_dataset_size: int,
        max_epochs=100,
        batch_size=512,
        num_warmup_epochs=10,
        base_lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])

        self.backbone = pretrained_model.backbone
        self.backbone.eval()

        for param in self.backbone.parameters():
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
            nn.Sigmoid(),
        )

        self.mse_loss = nn.MSELoss()
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, x, doy, mask):
        with torch.no_grad():
            pooled, _, _ = self.backbone(x, doy, mask)

        confidence = self.confid_net(pooled)

        return confidence, pooled

    def training_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()

        pred_conf, pooled = self.forward(x, doy, mask)

        with torch.no_grad():
            self.backbone.classifier.eval()

            logits = self.backbone.classifier(pooled.detach())
            probs = F.softmax(logits, dim=1)

            tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()

        loss = self.mse_loss(pred_conf.squeeze(), tcp_target)

        self.log("p2_conf_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("p2_lr", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()

        pred_conf, pooled = self.forward(x, doy, mask)

        with torch.no_grad():
            logits = self.backbone.classifier(pooled)
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

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]


sample_weights = train_dataset.get_weights_for_WeightedRandomSampler()

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

print("--- INICIANDO FASE 1: Classificação ---")
model_phase1 = Phase1_Classifier(
    num_classes=int(train_dataset.num_classes),
    max_epochs=MAX_EPOCHS_P1,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)


with torch.device("meta"):

    def model_fwd():
        return model_phase1(*model_phase1.backbone.example_input_array)

    fwd_flops = measure_flops(model_phase1, model_fwd)
    print(f"Forward FLOPs: {fwd_flops}")


# Carregar pesos pré-treinados do SITS-BERT se houver
# model_phase1.backbone.load_state_dict(torch.load("siam_texas_new_bert.pth"))

mlflow_logger = MLFlowLogger(
    experiment_name=f"{DATASET}-finetuning", tags=TAGS, run_name=RUN_NAME
)

checkpoint_cb_p1 = ModelCheckpoint(
    monitor="p1_val_loss", filename="best_classifier", mode="min"
)

early_stopping_cb_p1 = EarlyStopping(
    monitor="p1_val_loss",
    patience=10,
    mode="min",
)

devicestats_monitor = DeviceStatsMonitor(cpu_stats=False)

trainer_p1 = pl.Trainer(
    max_epochs=MAX_EPOCHS_P1,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu",
    precision="bf16-mixed",
    callbacks=[checkpoint_cb_p1, early_stopping_cb_p1, devicestats_monitor],
    logger=mlflow_logger,
    log_every_n_steps=5,
)

trainer_p1.fit(model_phase1, train_dataloader, val_dataloader)
best_model_p1 = Phase1_Classifier.load_from_checkpoint(checkpoint_cb_p1.best_model_path)

print("--- INICIANDO FASE 2: ConfidNet ---")

model_phase2 = Phase2_ConfidNet(
    pretrained_model=best_model_p1,
    max_epochs=MAX_EPOCHS_P2,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

early_stopping_cb_p2 = EarlyStopping(
    monitor="p2_val_loss",
    patience=10,
    mode="min",
)

checkpoint_cb_p2 = ModelCheckpoint(
    monitor="p2_val_loss", filename="best_confidnet", mode="min"
)
trainer_p2 = pl.Trainer(
    max_epochs=MAX_EPOCHS_P2,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu",
    precision="bf16-mixed",
    callbacks=[checkpoint_cb_p2, early_stopping_cb_p2],
    logger=mlflow_logger,
    log_every_n_steps=5,
)

trainer_p2.fit(model_phase2, train_dataloader, val_dataloader)

print("--- Gerando Predições Finais ---")
best_model_p2 = Phase2_ConfidNet.load_from_checkpoint(
    checkpoint_cb_p2.best_model_path, pretrained_model=best_model_p1
)

print("\n--- Validating Phase 2 Best Model ---")
trainer_p2.validate(model=best_model_p2, dataloaders=val_dataloader)

print("\n--- Loading Best Phase 2 Model for Inference ---")
best_model_p2.eval()


train_gdf = predict_and_save_predictions(
    best_model_p2,
    train_dataloader,
    train_dataset,
    mlflow_logger,
    "train",
    class_map,
    to_print=False,
)

val_gdf = predict_and_save_predictions(
    best_model_p2,
    val_dataloader,
    val_dataset,
    mlflow_logger,
    "val",
    class_map,
    to_print=False,
)

test_gdf = predict_and_save_predictions(
    best_model_p2,
    test_dataloader,
    test_dataset,
    mlflow_logger,
    "test",
    class_map,
    to_print=True,
)

run_gemos(train_gdf, val_gdf, test_gdf, mlflow_logger)
