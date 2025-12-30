import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearnex import patch_sklearn
from torch.optim.lr_scheduler import LambdaLR

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
)
from sits_siam.auxiliar import KNNCallback, beautify_prints, setup_seed
from sits_siam.models import SITSBert
from sits_siam.utils import SitsFinetuneDatasetFromNpz

patch_sklearn()
setup_seed()
beautify_prints()
torch.set_float32_matmul_precision("high")

DATASET = "texas"
BATCH_SIZE = 2 * 512
MAX_EPOCHS = 400
NUM_WARMUP_EPOCHS = 20
BASE_LR = 1e-4

TAGS = {
    "dataset": DATASET,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "num_warmup_epochs": NUM_WARMUP_EPOCHS,
    "base_lr": BASE_LR,
}
RUN_NAME = "-".join(str(value) for value in TAGS.values())


aug_transform = Pipeline(
    [
        LimitSequenceLength(127),
        IncreaseSequenceLength(127),
        RandomTempSwapping(max_distance=3),
        RandomTempShift(),
        RandomTempRemoval(),
        AddMissingMask(),
        Normalize(),
        AddCorruptedSample(),
        ToPytorchTensor(),
    ]
)

val_transform = Pipeline(
    [
        LimitSequenceLength(127),
        IncreaseSequenceLength(127),
        AddMissingMask(),
        Normalize(),
        AddCorruptedSample(),
        ToPytorchTensor(),
    ]
)

train_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/test.npz",
    transform=aug_transform,
)
val_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/val.npz",
    transform=val_transform,
)


knn_train_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/train.npz",
    transform=val_transform,
)
knn_val_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/val.npz",
    transform=val_transform,
)


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
        self.backbone = SITSBert(num_classes=1)
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


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
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


mlflow_logger = MLFlowLogger(experiment_name="california-pretrain")

knn_callback = KNNCallback(
    train_dataloader=knn_train_dataloader,
    val_dataloader=knn_val_dataloader,
    every_n_epochs=2,
    num_classes=knn_train_dataset.num_classes,
    k=3,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    mode="min",
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    callbacks=[knn_callback, checkpoint_callback, early_stopping_callback],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=mlflow_logger,
    gradient_clip_val=5.0,
    gradient_clip_algorithm="norm",
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

torch.save(model.backbone.state_dict(), "bert_all_new_bert.pth")
