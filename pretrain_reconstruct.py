import math
import argparse

import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearnex import patch_sklearn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import WeightedRandomSampler

from sits_siam.augment import (
    AddCorruptedSample,
    AddMissingMask,
    IncreaseSequenceLength,
    LimitSequenceLength,
    Normalize,
    Pipeline,
    RandomTempRemoval,
    RandomTempShift,
    RandomTempSwapping,
    ToPytorchTensor,
    RandomChanRemoval,
)
from sits_siam.auxiliar import (
    KNNCallback,
    setup_seed,
    save_pytorch_model,
    split_with_percent_and_class_coverage,
)
from sits_siam.models import (
    SITSBert,
    SITS_LSTM,
    SITSBertPlusPlus,
    SITSConvNext,
    SITSMamba,
)
from sits_siam.utils import SitsFinetuneDatasetFromNpz, AgriGEELiteDataset

patch_sklearn()
setup_seed()
# beautify_prints()
torch.set_float32_matmul_precision("high")

DATASET = "brazil"
BATCH_SIZE = 2 * 512
MAX_EPOCHS = 400
NUM_WARMUP_EPOCHS = 20
BASE_LR = 1e-4

BATCHED_ARGS_PARSER = argparse.ArgumentParser(add_help=False)
BATCHED_ARGS_PARSER.add_argument(
    "--model_name",
    type=str,
    choices=["MAMBA", "BERT", "BERTPP", "LSTM", "CNN"],
    default="MAMBA",
)
_parsed_args, _ = BATCHED_ARGS_PARSER.parse_known_args()
MODEL_NAME = _parsed_args.model_name

TAGS = {
    "dataset": DATASET,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "num_warmup_epochs": NUM_WARMUP_EPOCHS,
    "base_lr": BASE_LR,
    "model_name": MODEL_NAME,
}
RUN_NAME = "-".join(str(value) for value in TAGS.values())
EXPERIMENT_NAME = f"pretrain-{DATASET}"
RUN_NAME = f"{MODEL_NAME}-reconstruct"

aug_transforms = Pipeline(
    [
        LimitSequenceLength(127),
        IncreaseSequenceLength(127),
        RandomTempSwapping(max_distance=3),
        RandomTempShift(),
        RandomTempRemoval(),
        # RandomChanRemoval(0.2),
        AddMissingMask(),
        Normalize(),
        AddCorruptedSample(),
        ToPytorchTensor(),
    ]
)

val_transforms = Pipeline(
    [
        LimitSequenceLength(127),
        IncreaseSequenceLength(127),
        AddMissingMask(),
        Normalize(),
        AddCorruptedSample(),
        ToPytorchTensor(),
    ]
)

if DATASET in {"california", "texas"}:
    train_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/test.npz",
        transform=aug_transforms,
    )
    val_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/val.npz",
        transform=val_transforms,
    )

    knn_train_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/train.npz",
        transform=val_transforms,
    )
    knn_val_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/val.npz",
        transform=val_transforms,
    )
elif DATASET == "brazil":
    gdf = gpd.read_parquet("/home/m/Downloads/gdf.parquet")

    class_map = (
        gdf[["crop_class", "crop_number"]]
        .drop_duplicates()
        .set_index("crop_number")["crop_class"]
        .to_dict()
    )

    gdf_train, gdf_val, gdf_test = split_with_percent_and_class_coverage(
        gdf, percent=1, max_attempts=500
    )

    train_dataset = AgriGEELiteDataset(
        gdf_test,  # Test set used for training in pretraining
        "/home/m/Downloads/df_sits.parquet",
        transform=aug_transforms,
        timestamp_processing="days_after_start",
    )

    val_dataset = AgriGEELiteDataset(
        gdf_val,
        "/home/m/Downloads/df_sits.parquet",
        transform=val_transforms,
        timestamp_processing="days_after_start",
    )

    knn_train_dataset = AgriGEELiteDataset(
        gdf_val,
        "/home/m/Downloads/df_sits.parquet",
        transform=val_transforms,
        timestamp_processing="days_after_start",
    )

    knn_val_dataset = AgriGEELiteDataset(
        gdf_train,
        "/home/m/Downloads/df_sits.parquet",
        transform=val_transforms,
        timestamp_processing="days_after_start",
    )
else:
    raise ValueError(f"Dataset {DATASET} not recognized.")


class TransformerClassifier(pl.LightningModule):
    def __init__(
        self,
        train_dataset_size: int,
        max_epochs: int,
        batch_size: int,
        num_warmup_epochs: int,
        base_lr: float,
    ):
        super(TransformerClassifier, self).__init__()
        BACKBONES = {
            "BERT": SITSBert,
            "BERTPP": SITSBertPlusPlus,
            "LSTM": SITS_LSTM,
            "CNN": SITSConvNext,
            "MAMBA": SITSMamba,
        }
        self.backbone = BACKBONES[MODEL_NAME](num_classes=1)
        self.criterion = nn.MSELoss(reduction="none")

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, batch):
        x = batch["x"]
        doy = batch["doy"]
        mask = batch["mask"]

        pooled, _, _ = self.backbone(x, doy, mask)

        return pooled

    def forward_corrupted(self, batch):
        x = batch["corrupted_x"]
        doy = batch["doy"]
        mask = batch["mask"]

        _, _, reconstructed = self.backbone(x, doy, mask)
        return reconstructed

    def training_step(self, batch, batch_idx):
        pred = self.forward_corrupted(batch)

        loss = self.criterion(pred, batch["x"].float())
        mask = batch["corrupted_mask"].unsqueeze(-1)
        loss = (loss * mask.float()).sum() / mask.sum()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pred = self.forward_corrupted(batch)
        loss = self.criterion(pred, batch["x"].float())
        mask = batch["corrupted_mask"].unsqueeze(-1)
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

knn_train_dataloader = torch.utils.data.DataLoader(
    knn_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
knn_val_dataloader = torch.utils.data.DataLoader(
    knn_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


mlflow_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME)

knn_callback = KNNCallback(
    train_dataloader=knn_train_dataloader,
    val_dataloader=knn_val_dataloader,
    every_n_epochs=2,
    num_classes=knn_train_dataset.num_classes,
    k=3,
)
checkpoint_callback = ModelCheckpoint(
    monitor="knn_f1_weighted", filename="best_model", save_top_k=1, mode="max"
)
early_stopping_callback = EarlyStopping(
    monitor="knn_f1_weighted",
    patience=40,
    mode="max",
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    callbacks=[knn_callback, checkpoint_callback, early_stopping_callback],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=mlflow_logger,
)
model = TransformerClassifier(
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

trainer.validate(model=model, dataloaders=val_dataloader)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model = TransformerClassifier.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

save_pytorch_model(model.backbone, mlflow_logger)
