import math

import lightning.pytorch as pl
import numpy as np
import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.utils.debug import std_of_l2_normalized
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearnex import patch_sklearn
from torch.optim.lr_scheduler import LambdaLR

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
from sits_siam.auxiliar import KNNCallback, beautify_prints, setup_seed
from sits_siam.models import SITSBert
from sits_siam.utils import SitsFinetuneDatasetFromNpz, SitsPretrainDatasetFromNpz


patch_sklearn()
setup_seed()
beautify_prints()
torch.set_float32_matmul_precision("high")

DATASET = "texas"
BATCH_SIZE = 2 * 512
MAX_EPOCHS = 100
NUM_WARMUP_EPOCHS = 20
BASE_LR = 1e-4
NUM_VIEWS = 2

TAGS = {
    "dataset": DATASET,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "num_warmup_epochs": NUM_WARMUP_EPOCHS,
    "base_lr": BASE_LR,
    "num_views": NUM_VIEWS,
}
RUN_NAME = "-".join(str(value) for value in TAGS.values())


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


train_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/test.npz",
    transform=FastSiamMultiViewTransform(n_views=NUM_VIEWS),
)
val_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/val.npz",
    transform=FastSiamMultiViewTransform(n_views=NUM_VIEWS),
)

knn_transform = Pipeline(
    [
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

knn_train_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/train.npz",
    transform=knn_transform,
)
knn_val_dataset = SitsFinetuneDatasetFromNpz(
    "/mnt/c/Users/m/Downloads/grsl/california_01_01_998/val.npz",
    transform=knn_transform,
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
        self.projection_head = SimSiamProjectionHead(256, 512, 1024)
        self.prediction_head = SimSiamPredictionHead(1024, 512, 1024)

        self.criterion = NegativeCosineSimilarity()

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

        z = self.projection_head(pooled)
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


checkpoint_callback = ModelCheckpoint(
    monitor="knn_f1_macro", filename="best_model", save_top_k=1, mode="max"
)
knn_callback = KNNCallback(
    train_dataloader=knn_train_dataloader,
    val_dataloader=knn_val_dataloader,
    every_n_epochs=2,
    k=3,
)
early_stopping_callback = EarlyStopping(
    monitor="knn_f1_macro",
    patience=10,
    mode="max",
)
mlflow_logger = MLFlowLogger(experiment_name="TEXASPRETRAIN")

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    callbacks=[checkpoint_callback, knn_callback, early_stopping_callback],
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
model = TransformerClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

torch.save(model.backbone.state_dict(), "siam_texas_new_bert.pth")
# Saving pytorch model backbone state dict
# torch.save(model.backbone.state_dict(), "weights/fastsiam_texas.pth")
