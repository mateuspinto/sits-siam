import argparse
import copy
import math

import geopandas as gpd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
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
    ReduceToMonthlyMeans,
    ToPytorchTensor,
)
from sits_siam.auxiliar import (
    predict_and_save_predictions,
    setup_seed,
    run_gemos,
    save_pytorch_model,
    split_with_percent_and_class_coverage,
    check_if_already_ran,
)
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz
from sits_siam.models import SITS_MLP_Backbone

patch_sklearn()
torch.set_float32_matmul_precision("high")
setup_seed()
# beautify_prints()


BATCHED_ARGS_PARSER = argparse.ArgumentParser(add_help=False)
BATCHED_ARGS_PARSER.add_argument(
    "--train_percent",
    type=float,
    default=70.0,
)
BATCHED_ARGS_PARSER.add_argument(
    "--dataset",
    type=str,
    choices=["brazil", "california", "texas"],
    default="brazil",
)
BATCHED_ARGS_PARSER.add_argument(
    "--gpu",
    type=int,
    default=0,
)
_parsed_args, _ = BATCHED_ARGS_PARSER.parse_known_args()
TRAIN_PERCENT = float(_parsed_args.train_percent)
GPU_ID = _parsed_args.gpu
DATASET = _parsed_args.dataset
BATCH_SIZE = 2 * 512

MAX_EPOCHS = 100
if TRAIN_PERCENT <= 1:
    MAX_EPOCHS = 200
NUM_WARMUP_EPOCHS = 10
BASE_LR = 1e-4

TAGS = {
    "dataset": str(DATASET),
    "batch_size": str(BATCH_SIZE),
    "max_epochs": str(MAX_EPOCHS),
    "num_warmup_epochs": str(NUM_WARMUP_EPOCHS),
    "base_lr": str(BASE_LR),
    "train_percent": str(TRAIN_PERCENT),
    "model": "MLP",
}
RUN_NAME = f"MLP-{TRAIN_PERCENT}"
EXPERIMENT_NAME = f"{DATASET}-finetuning"

# if check_if_already_ran(EXPERIMENT_NAME, RUN_NAME):
#     print(RUN_NAME, "already ran in", EXPERIMENT_NAME)
#     exit()

transforms = Pipeline(
    [
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        ReduceToMonthlyMeans(),
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
        ReduceToMonthlyMeans(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

if DATASET == "brazil":
    gdf = gpd.read_parquet("data/agl/gdf.parquet")

    class_map = (
        gdf[["crop_class", "crop_number"]]
        .drop_duplicates()
        .set_index("crop_number")["crop_class"]
        .to_dict()
    )

    if abs(TRAIN_PERCENT - 70.0) < 1e-9:
        unique_mun = gdf["CD_MUN"].unique()

        mun_train, mun_temp = train_test_split(
            unique_mun, test_size=0.30, random_state=13
        )

        mun_val, mun_test = train_test_split(mun_temp, test_size=0.50, random_state=13)

        gdf_train = gdf[gdf["CD_MUN"].isin(mun_train)].copy()
        gdf_val = gdf[gdf["CD_MUN"].isin(mun_val)].copy()
        gdf_test = gdf[gdf["CD_MUN"].isin(mun_test)].copy()
    else:
        # New path for small percent modes: 10, 1, 0.1
        gdf_train, gdf_val, gdf_test = split_with_percent_and_class_coverage(
            gdf, percent=TRAIN_PERCENT, max_attempts=500
        )

    train_dataset = AgriGEELiteDataset(
        gdf_train,
        "data/agl/df_sits.parquet",
        transform=aug_transforms,
        timestamp_processing="days_after_start",
    )

    val_dataset = AgriGEELiteDataset(
        gdf_val,
        "data/agl/df_sits.parquet",
        transform=transforms,
        timestamp_processing="days_after_start",
    )

    test_dataset = AgriGEELiteDataset(
        gdf_test,
        "data/agl/df_sits.parquet",
        transform=transforms,
        timestamp_processing="days_after_start",
    )

elif DATASET in {"texas", "california"}:
    if TRAIN_PERCENT == 70:
        split_string = "npz"
    elif TRAIN_PERCENT == 10:
        split_string = "10_10_80"
    elif TRAIN_PERCENT == 1:
        split_string = "1_1_98"
    elif TRAIN_PERCENT == 0.1:
        split_string = "001_001_998"
    else:
        raise ValueError(f"TRAIN_PERCENT {TRAIN_PERCENT} not supported for {DATASET}")

    train_dataset = SitsFinetuneDatasetFromNpz(
        f"data/{DATASET}_{split_string}/train.npz",
        transform=aug_transforms,
    )
    val_dataset = SitsFinetuneDatasetFromNpz(
        f"data/{DATASET}_{split_string}/val.npz",
        transform=transforms,
    )
    test_dataset = SitsFinetuneDatasetFromNpz(
        f"data/{DATASET}_{split_string}/test.npz",
        transform=transforms,
    )

    # Create class_map from dataset
    class_map = {i: str(i) for i in range(int(train_dataset.num_classes))}
else:
    raise ValueError(f"Dataset {DATASET} not recognized.")


class Phase1_Classifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        train_dataset_size,
        max_epochs,
        batch_size,
        num_warmup_epochs,
        base_lr,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = SITS_MLP_Backbone(
            input_channels=10, num_classes=num_classes, time_steps=12, hidden_dim=256
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, x, doy, mask):
        logits, last_emb, all_embs = self.backbone(x, None, None)

        return logits, last_emb, all_embs

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"].squeeze()
        logits, _, _ = self.forward(x, None, None)

        loss = self.criterion(logits, y)
        self.log("p1_train_loss", loss, prog_bar=True)
        self.log("p1_train_acc", self.train_acc(logits, y), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("p1_lr", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"].squeeze()
        logits, _, _ = self.forward(x, None, None)

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
        train_dataset_size,
        max_epochs,
        batch_size,
        num_warmup_epochs,
        base_lr,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])
        self.backbone = pretrained_model.backbone

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Usando apenas a saída final (hidden_dim=256)
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
            logits, last_emb, all_embs = self.backbone(x, doy, mask)

        confidence = self.confid_net(last_emb)

        return confidence, logits, last_emb, all_embs

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"].squeeze()

        confidence, logits, last_emb, _ = self.forward(x, None, None)

        with torch.no_grad():
            self.backbone.classifier.eval()

            logits = self.backbone.classifier(last_emb.detach())
            probs = F.softmax(logits, dim=1)

            tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()

        loss = self.mse_loss(confidence.squeeze(), tcp_target)

        self.log("p2_conf_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("p2_lr", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"].squeeze()

        confidence, logits, last_emb, _ = self.forward(x, None, None)

        with torch.no_grad():
            logits = self.backbone.classifier(last_emb)
            probs = F.softmax(logits, dim=1)
            tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()

            loss = self.mse_loss(confidence.squeeze(), tcp_target)

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


class Phase3_ConfidNetFinetuning(pl.LightningModule):
    def __init__(
        self,
        pretrained_confidnet: Phase2_ConfidNet,
        train_dataset_size,
        max_epochs,
        batch_size,
        num_warmup_epochs,
        base_lr,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_confidnet"])

        self.backbone_frozen = copy.deepcopy(pretrained_confidnet.backbone)
        self.backbone_frozen.eval()
        for param in self.backbone_frozen.parameters():
            param.requires_grad = False

        self.backbone = copy.deepcopy(pretrained_confidnet.backbone)

        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

        self.confid_net = pretrained_confidnet.confid_net

        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.confid_net.parameters():
            param.requires_grad = True

        self.backbone.train()
        self.confid_net.train()

        self.mse_loss = nn.MSELoss()
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, x, doy, mask):
        _, last_emb, _ = self.backbone(x, doy, mask)
        confidence = self.confid_net(last_emb)

        with torch.no_grad():
            logits_frozen, last_emb_frozen, all_embs_frozen = self.backbone_frozen(
                x, doy, mask
            )

        return confidence, logits_frozen, last_emb_frozen, all_embs_frozen

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"].squeeze()

        confidence, logits_frozen, last_emb_frozen, all_embs_frozen = self.forward(
            x, None, None
        )

        logits = self.backbone.classifier(last_emb_frozen)
        probs = F.softmax(logits, dim=1)

        tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()

        loss = self.mse_loss(confidence.squeeze(), tcp_target)

        self.log("p3_conf_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("p3_lr", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"].squeeze()

        confidence, logits_frozen, last_emb_frozen, all_embs_frozen = self.forward(
            x, None, None
        )

        logits = self.backbone.classifier(last_emb_frozen)
        probs = F.softmax(logits, dim=1)
        tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()

        loss = self.mse_loss(confidence.squeeze(), tcp_target)

        self.log("p3_val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr / 10)
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
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow_logger = MLFlowLogger(
    experiment_name=EXPERIMENT_NAME,
    tags=TAGS,
    run_name=RUN_NAME,
    tracking_uri=mlflow.get_tracking_uri(),
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
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu",
    devices=[GPU_ID],
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
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

checkpoint_cb_p2 = ModelCheckpoint(
    monitor="p2_val_loss", filename="best_confidnet", mode="min"
)
early_stopping_cb_p2 = EarlyStopping(
    monitor="p2_val_loss",
    patience=10,
    mode="min",
)

trainer_p2 = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu",
    devices=[GPU_ID],
    precision="bf16-mixed",
    callbacks=[checkpoint_cb_p2, early_stopping_cb_p2, devicestats_monitor],
    logger=mlflow_logger,
    log_every_n_steps=5,
)

trainer_p2.fit(model_phase2, train_dataloader, val_dataloader)

print("--- Carregando melhor modelo da Fase 2 ---")
best_model_p2 = Phase2_ConfidNet.load_from_checkpoint(
    checkpoint_cb_p2.best_model_path, pretrained_model=best_model_p1
)

print("\n--- INICIANDO FASE 3: Finetuning ConfidNet Completa ---")

model_phase3 = Phase3_ConfidNetFinetuning(
    pretrained_confidnet=best_model_p2,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

checkpoint_cb_p3 = ModelCheckpoint(
    monitor="p3_val_loss", filename="best_confidnet_finetuned", mode="min"
)
early_stopping_cb_p3 = EarlyStopping(
    monitor="p3_val_loss",
    patience=10,
    mode="min",
)

trainer_p3 = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu",
    devices=[GPU_ID],
    precision="bf16-mixed",
    callbacks=[checkpoint_cb_p3, early_stopping_cb_p3, devicestats_monitor],
    logger=mlflow_logger,
    log_every_n_steps=5,
)

trainer_p3.fit(model_phase3, train_dataloader, val_dataloader)

print("--- Gerando Predições Finais ---")
best_model_p3 = Phase3_ConfidNetFinetuning.load_from_checkpoint(
    checkpoint_cb_p3.best_model_path, pretrained_confidnet=best_model_p2
)

print("\n--- Validating Phase 3 Best Model ---")
trainer_p3.validate(model=best_model_p3, dataloaders=val_dataloader)

print("\n--- Loading Best Phase 3 Model for Inference ---")
best_model_p3.eval()

if DATASET == "brazil":
    train_val_dataset = AgriGEELiteDataset(
        gdf_train,
        "data/agl/df_sits.parquet",
        transform=transforms,
        timestamp_processing="days_after_start",
    )
elif DATASET in {"texas", "california"}:
    if TRAIN_PERCENT == 70:
        split_string = "npz"
    elif TRAIN_PERCENT == 10:
        split_string = "10_10_80"
    elif TRAIN_PERCENT == 1:
        split_string = "1_1_98"
    elif TRAIN_PERCENT == 0.1:
        split_string = "001_001_998"
    else:
        raise ValueError(f"TRAIN_PERCENT {TRAIN_PERCENT} not supported for {DATASET}")

    train_val_dataset = SitsFinetuneDatasetFromNpz(
        f"data/{DATASET}_{split_string}/train.npz",
        transform=transforms,
    )
else:
    raise ValueError(f"Dataset {DATASET} not recognized.")

train_val_dataloader = torch.utils.data.DataLoader(
    train_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

train_gdf = predict_and_save_predictions(
    best_model_p3,
    train_dataloader,
    train_dataset,
    mlflow_logger,
    "train_aug",
    class_map,
    to_print=False,
)

train_val_gdf = predict_and_save_predictions(
    best_model_p3,
    train_val_dataloader,
    train_val_dataset,
    mlflow_logger,
    "train",
    class_map,
    to_print=False,
)

val_gdf = predict_and_save_predictions(
    best_model_p3,
    val_dataloader,
    val_dataset,
    mlflow_logger,
    "val",
    class_map,
    to_print=False,
)

test_gdf = predict_and_save_predictions(
    best_model_p3,
    test_dataloader,
    test_dataset,
    mlflow_logger,
    "test",
    class_map,
    to_print=True,
)

run_gemos(train_gdf, train_val_gdf, val_gdf, test_gdf, mlflow_logger)
save_pytorch_model(best_model_p3, mlflow_logger)
