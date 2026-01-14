import math
import argparse
import copy

import geopandas as gpd
import lightning.pytorch as pl
import numpy as np
import torch
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
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
from sits_siam.auxiliar import (
    KNNCallback,
    beautify_prints,
    setup_seed,
    check_if_already_ran,
    TSNECallback,
    split_with_percent_and_class_coverage,
    save_pytorch_model
)
from sits_siam.models import (
    SITSBert,
    SITSBertPlusPlus,
    SITS_LSTM,
    SITSConvNext,
    SITSMamba,
)
from sits_siam.utils import SitsFinetuneDatasetFromNpz, SitsPretrainDatasetFromNpz, AgriGEELiteDataset


patch_sklearn()
setup_seed()
beautify_prints()
torch.set_float32_matmul_precision("high")

BATCH_SIZE = 2 * 512
MAX_EPOCHS = 100
NUM_WARMUP_EPOCHS = 20
BASE_LR = 1e-4
NUM_VIEWS = 2

BATCHED_ARGS_PARSER = argparse.ArgumentParser(add_help=False)
BATCHED_ARGS_PARSER.add_argument(
    "--model_name",
    type=str,
    choices=["MAMBA", "BERT", "BERTPP", "LSTM", "CNN"],
    default="BERT",
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
MODEL_NAME = _parsed_args.model_name
DATASET = _parsed_args.dataset
GPU_ID = _parsed_args.gpu

TAGS = {
    "dataset": DATASET,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "num_warmup_epochs": NUM_WARMUP_EPOCHS,
    "base_lr": BASE_LR,
    "num_views": NUM_VIEWS,
    "model_name": MODEL_NAME,
}

EXPERIMENT_NAME = f"{DATASET}-pretrain"
RUN_NAME = f"{MODEL_NAME}-MoCo"

# if check_if_already_ran(EXPERIMENT_NAME, RUN_NAME):
#     print(RUN_NAME, "already ran in", EXPERIMENT_NAME)
#     exit()


class MoCoMultiViewTransform(object):
    def __init__(
        self,
        n_views: int = 2,
    ):
        self.n_views = n_views
        self.transform = Pipeline(
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

    def __call__(self, sample: np.ndarray):
        return [
            self.transform({k: v.copy() for k, v in sample.items()})
            for _ in range(self.n_views)
        ]


knn_transform = Pipeline(
    [
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

if DATASET in {"california", "texas"}:
    train_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/test.npz",
        transform=MoCoMultiViewTransform(n_views=NUM_VIEWS),
    )
    val_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/val.npz",
        transform=MoCoMultiViewTransform(n_views=NUM_VIEWS),
    )

    knn_train_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/train.npz",
        transform=knn_transform,
    )
    knn_val_dataset = SitsFinetuneDatasetFromNpz(
        f"/mnt/c/Users/m/Downloads/grsl/{DATASET}_01_01_998/val.npz",
        transform=knn_transform,
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
        gdf_test,
        "/home/m/Downloads/df_sits.parquet",
        transform=MoCoMultiViewTransform(n_views=NUM_VIEWS),
        timestamp_processing="days_after_start",
    )

    val_dataset = AgriGEELiteDataset(
        gdf_val,
        "/home/m/Downloads/df_sits.parquet",
        transform=MoCoMultiViewTransform(n_views=NUM_VIEWS),
        timestamp_processing="days_after_start",
    )

    knn_train_dataset = AgriGEELiteDataset(
        gdf_val,
        "/home/m/Downloads/df_sits.parquet",
        transform=knn_transform,
        timestamp_processing="days_after_start",
    )

    knn_val_dataset = AgriGEELiteDataset(
        gdf_train,
        "/home/m/Downloads/df_sits.parquet",
        transform=knn_transform,
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
        self.projection_head = MoCoProjectionHead(input_dim=self.backbone.hidden_dim, hidden_dim=512, output_dim=128)

        # Momentum Encoder (cópia do encoder principal)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Desativa gradientes para o momentum encoder
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=(4096, 128))

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

    def forward_moco(self, batch, use_momentum=False):
        x = batch["x"]
        doy = batch["doy"]
        mask = batch["mask"]

        if use_momentum:
            pooled, _, _ = self.backbone_momentum(x, doy, mask)
            projection = self.projection_head_momentum(pooled)
            return projection.detach()
        else:
            pooled, _, _ = self.backbone(x, doy, mask)
            projection = self.projection_head(pooled)
            return projection

    def training_step(self, batch, batch_idx):
        # Atualiza os pesos do encoder de momentum
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        # O dataloader retorna uma lista com duas views [view1, view2]
        view1, view2 = batch

        # Gera query e key
        query = self.forward_moco(view1, use_momentum=False)
        key = self.forward_moco(view2, use_momentum=True)

        # Calcula a perda
        loss = self.criterion(query, key)

        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_collapse",
            std_of_l2_normalized(query.detach()),
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # O dataloader de validação também retorna duas views
        view1, view2 = batch

        # Gera query e key
        query = self.forward_moco(view1, use_momentum=False)
        key = self.forward_moco(view2, use_momentum=True)

        # Calcula a perda de validação
        loss = self.criterion(query, key)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "val_collapse",
            std_of_l2_normalized(query.detach()),
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        view1, view2 = batch

        query = self.forward_moco(view1, use_momentum=False)
        key = self.forward_moco(view2, use_momentum=True)

        loss = self.criterion(query, key)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_collapse", std_of_l2_normalized(query.detach()), prog_bar=True)
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
    monitor="knn_f1_weighted", filename="best_model", save_top_k=1, mode="max"
)
knn_callback = KNNCallback(
    train_dataloader=knn_train_dataloader,
    val_dataloader=knn_val_dataloader,
    every_n_epochs=2,
    num_classes=knn_train_dataset.num_classes,
    k=3,
)
early_stopping_callback = EarlyStopping(
    monitor="knn_f1_weighted",
    patience=10,
    mode="max",
)

mlflow_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME)

tsne_callback = TSNECallback(
    train_dataset=knn_val_dataset,
    mlflow_logger=mlflow_logger,
    num_samples=1000,
    every_n_epochs=30
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    callbacks=[tsne_callback, checkpoint_callback, knn_callback, early_stopping_callback],
    accelerator="gpu",
    devices=[GPU_ID],
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
    base_lr=BASE_LR
)

save_pytorch_model(model.backbone, mlflow_logger)

