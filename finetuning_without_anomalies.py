"""
finetuning_without_anomalies.py

Pipeline:
  1. KFold anomaly detection on train+val (Phase1 + per-class GMM)
  2. Remove anomalies from train+val
  3. Train full 3-phase model on clean data
  4. Evaluate on test set with anomalies (all) and without (GMM-filtered via run_gemos)
"""

import copy
import math
import argparse
import warnings
import sys
import logging
import tempfile
import os

IS_TTY = sys.stdout.isatty()  # False when stdout redirected to log file

# ---------------------------------------------------------------------------
# Thread limits — script runs twice in parallel (one per GPU).
# Each process gets at most half the available CPU cores.
# Must be set BEFORE importing numpy / torch so OMP/MKL honour them.
# ---------------------------------------------------------------------------
_HALF_CORES = str(max(1, os.cpu_count() // 2))
os.environ.setdefault("OMP_NUM_THREADS",  _HALF_CORES)
os.environ.setdefault("MKL_NUM_THREADS",  _HALF_CORES)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _HALF_CORES)
os.environ.setdefault("NUMEXPR_NUM_THREADS",  _HALF_CORES)
NUM_WORKERS = max(1, int(_HALF_CORES) // 2)  # DataLoader workers per process

# Suppress Lightning's "install litlogger" tip and other INFO noise
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearnex import patch_sklearn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import WeightedRandomSampler
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

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
    load_pretrain_weights,
    predict_and_save_predictions,
    setup_seed,
    run_gemos,
    save_pytorch_model,
    split_with_percent_and_class_coverage,
)
from sits_siam.models import (
    SITSBert,
    SITSBertPlusPlus,
    SITS_LSTM,
    SITSConvNext,
    SITSMamba,
)
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz

patch_sklearn()
torch.set_float32_matmul_precision("high")
torch.set_num_threads(int(_HALF_CORES))
torch.set_num_interop_threads(max(1, int(_HALF_CORES) // 2))
setup_seed()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
BATCHED_ARGS_PARSER = argparse.ArgumentParser(add_help=False)
BATCHED_ARGS_PARSER.add_argument("--train_percent", type=float, default=70.0)
BATCHED_ARGS_PARSER.add_argument(
    "--model_name", type=str,
    choices=["MAMBA", "BERT", "BERTPP", "LSTM", "CNN"], default="BERT",
)
BATCHED_ARGS_PARSER.add_argument(
    "--dataset", type=str,
    choices=["brazil", "california", "texas"], default="brazil",
)
BATCHED_ARGS_PARSER.add_argument(
    "--pretrain", type=str,
    choices=["off", "reconstruct", "MoCo", "PMSN", "FastSiam"], default="off",
)
BATCHED_ARGS_PARSER.add_argument("--gpu", type=int, default=0)
BATCHED_ARGS_PARSER.add_argument("--n_folds", type=int, default=5)

_parsed_args, _ = BATCHED_ARGS_PARSER.parse_known_args()
TRAIN_PERCENT = float(_parsed_args.train_percent)
GPU_ID = _parsed_args.gpu
DATASET = _parsed_args.dataset
MODEL_NAME = _parsed_args.model_name
PRETRAIN = _parsed_args.pretrain
N_FOLDS = _parsed_args.n_folds

BATCH_SIZE = 2 * 512
MAX_EPOCHS = 100
KFOLD_MAX_EPOCHS = 50
if TRAIN_PERCENT <= 1:
    MAX_EPOCHS = 200
    KFOLD_MAX_EPOCHS = 100
NUM_WARMUP_EPOCHS = 10
BASE_LR = 1e-4

EXPERIMENT_NAME = f"{DATASET}-finetuning-wo-anomalies"
RUN_NAME = f"{MODEL_NAME}-{TRAIN_PERCENT}"
if PRETRAIN != "off":
    RUN_NAME += f"-{PRETRAIN}"

TAGS = {
    "dataset": DATASET,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "pretrain": PRETRAIN,
    "num_warmup_epochs": NUM_WARMUP_EPOCHS,
    "base_lr": BASE_LR,
    "train_percent": TRAIN_PERCENT,
    "model_name": MODEL_NAME,
    "n_folds": N_FOLDS,
}

import mlflow
mlflow.set_experiment(EXPERIMENT_NAME)  # creates on server if not exists

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
transforms = Pipeline([
    LimitSequenceLength(140),
    IncreaseSequenceLength(140),
    AddMissingMask(),
    Normalize(),
    ToPytorchTensor(),
])

aug_transforms = Pipeline([
    LimitSequenceLength(140),
    IncreaseSequenceLength(140),
    RandomTempShift(),
    RandomAddNoise(),
    RandomTempRemoval(),
    RandomTempSwapping(max_distance=3),
    AddMissingMask(),
    Normalize(),
    ToPytorchTensor(),
])

# ---------------------------------------------------------------------------
# Lightweight dataset helpers
# ---------------------------------------------------------------------------

class _SubsetWithWeights(torch.utils.data.Dataset):
    """
    Index-based subset of any dataset that exposes .ys.
    Provides get_weights_for_WeightedRandomSampler() and .num_classes.
    Optionally carries a reference to parent's .gdf (subset rows) so that
    predict_and_save_predictions can merge spatial attributes.
    """

    def __init__(self, parent, indices, gdf_subset=None):
        self.parent = parent
        self.indices = np.asarray(indices)
        self.ys = parent.ys[self.indices]
        self.num_classes = parent.num_classes
        if gdf_subset is not None:
            self.gdf = gdf_subset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]

    def get_weights_for_WeightedRandomSampler(self):
        classes, counts = np.unique(self.ys, return_counts=True)
        weight_map = dict(zip(classes, 1.0 / counts))
        weights = np.array([weight_map[t] for t in self.ys])
        return torch.from_numpy(weights).double()


class NpzSubset(torch.utils.data.Dataset):
    """Dataset backed by pre-sliced numpy arrays (for Texas / California)."""

    def __init__(self, ts, doys, ys, transform=None):
        self.ts = ts.astype(np.float16)
        self.doys = doys.astype(np.int16)
        self.ys = ys.astype(np.int16)
        self.transform = transform
        self.num_classes = int(np.max(ys) + 1)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx):
        sample = {"x": self.ts[idx], "doy": self.doys[idx], "y": self.ys[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_weights_for_WeightedRandomSampler(self):
        classes, counts = np.unique(self.ys, return_counts=True)
        weight_map = dict(zip(classes, 1.0 / counts))
        weights = np.array([weight_map[t] for t in self.ys])
        return torch.from_numpy(weights).double()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
if DATASET == "brazil":
    gdf = gpd.read_parquet("data/agl/gdf.parquet")
    class_map = (
        gdf[["crop_class", "crop_number"]]
        .drop_duplicates()
        .set_index("crop_number")["crop_class"]
        .to_dict()
    )
    NUM_CLASSES = int(gdf["crop_class"].nunique())

    if abs(TRAIN_PERCENT - 70.0) < 1e-9:
        unique_mun = gdf["CD_MUN"].unique()
        mun_train, mun_temp = train_test_split(unique_mun, test_size=0.30, random_state=13)
        mun_val, mun_test = train_test_split(mun_temp, test_size=0.50, random_state=13)
        gdf_train = gdf[gdf["CD_MUN"].isin(mun_train)].copy()
        gdf_val = gdf[gdf["CD_MUN"].isin(mun_val)].copy()
        gdf_test = gdf[gdf["CD_MUN"].isin(mun_test)].copy()
    else:
        gdf_train, gdf_val, gdf_test = split_with_percent_and_class_coverage(
            gdf, percent=TRAIN_PERCENT, max_attempts=500
        )

    N_TRAIN = len(gdf_train)
    sits_df = pd.read_parquet("data/agl/df_sits.parquet")

    # Combined trainval gdf — preserve original index so AgriGEELiteDataset
    # can match indexnum values to sits_df
    gdf_trainval = pd.concat([gdf_train, gdf_val])

    # Build the two base datasets ONCE; KFold uses lightweight _SubsetWithWeights
    print("Building trainval datasets (done once, shared across folds)...")
    trainval_aug = AgriGEELiteDataset(
        gdf_trainval, sits_df,
        transform=aug_transforms,
        timestamp_processing="days_after_start",
    )
    trainval_noaug = AgriGEELiteDataset(
        gdf_trainval, sits_df,
        transform=transforms,
        timestamp_processing="days_after_start",
    )

    test_dataset = AgriGEELiteDataset(
        gdf_test, sits_df,
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

    _train_npz = np.load(f"data/{DATASET}_{split_string}/train.npz")
    _val_npz = np.load(f"data/{DATASET}_{split_string}/val.npz")
    _test_npz = np.load(f"data/{DATASET}_{split_string}/test.npz")

    train_ts = _train_npz["ts"].astype(np.float16)
    train_doys = _train_npz["doys"].astype(np.int16)
    train_ys = _train_npz["ys"].astype(np.int16)

    val_ts = _val_npz["ts"].astype(np.float16)
    val_doys = _val_npz["doys"].astype(np.int16)
    val_ys = _val_npz["ys"].astype(np.int16)

    N_TRAIN = len(train_ts)

    tv_ts = np.concatenate([train_ts, val_ts], axis=0)
    tv_doys = np.concatenate([train_doys, val_doys], axis=0)
    tv_ys = np.concatenate([train_ys, val_ys], axis=0)

    NUM_CLASSES = int(np.max(tv_ys) + 1)

    _ref = SitsFinetuneDatasetFromNpz(f"data/{DATASET}_{split_string}/train.npz")
    _class_names = _ref.get_class_names()
    class_map = {i: _class_names[i] for i in range(len(_class_names))}

    # Full trainval datasets (aug and noaug) backed by combined arrays
    trainval_aug = NpzSubset(tv_ts, tv_doys, tv_ys, transform=aug_transforms)
    trainval_noaug = NpzSubset(tv_ts, tv_doys, tv_ys, transform=transforms)

    test_dataset = NpzSubset(
        _test_npz["ts"], _test_npz["doys"], _test_npz["ys"], transform=transforms
    )
else:
    raise ValueError(f"Dataset {DATASET} not recognized.")


# ---------------------------------------------------------------------------
# Model classes (identical to finetuning.py)
# ---------------------------------------------------------------------------

def _lr_lambda_factory(train_dataset_size, batch_size, max_epochs, num_warmup_epochs, base_lr):
    steps_per_epoch = math.ceil(train_dataset_size / batch_size)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = steps_per_epoch * num_warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            warmup_factor = 1.0 / 1000
            alpha = float(current_step) / float(max(1, warmup_steps))
            return warmup_factor * (1 - alpha) + alpha * 1.0
        decay_steps = total_steps - warmup_steps
        step_in_decay = current_step - warmup_steps
        progress = min(1.0, float(step_in_decay) / float(max(1, decay_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


class Phase1_Classifier(pl.LightningModule):
    def __init__(self, num_classes, train_dataset_size, max_epochs=100,
                 batch_size=512, num_warmup_epochs=10, base_lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        BACKBONES = {
            "BERT": SITSBert, "BERTPP": SITSBertPlusPlus,
            "LSTM": SITS_LSTM, "CNN": SITSConvNext, "MAMBA": SITSMamba,
        }
        self.backbone = BACKBONES[MODEL_NAME](num_classes=num_classes)

        if PRETRAIN != "off":
            self.backbone = load_pretrain_weights(DATASET, PRETRAIN, MODEL_NAME, self.backbone)

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
        lr_lambda = _lr_lambda_factory(
            self.train_dataset_size, self.batch_size,
            self.max_epochs, self.num_warmup_epochs, self.base_lr,
        )
        return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}]


class Phase2_ConfidNet(pl.LightningModule):
    def __init__(self, pretrained_model: Phase1_Classifier, train_dataset_size: int,
                 max_epochs=100, batch_size=512, num_warmup_epochs=10, base_lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])

        self.backbone = pretrained_model.backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.confid_net = nn.Sequential(
            nn.Linear(self.backbone.hidden_dim, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 1), nn.Sigmoid(),
        )
        self.mse_loss = nn.MSELoss()
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr

    def forward(self, x, doy, mask):
        with torch.no_grad():
            pooled, logits, _ = self.backbone(x, doy, mask)
        confidence = self.confid_net(pooled)
        return confidence, logits, pooled, pooled

    def training_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        confidence, logits, last_emb, _ = self.forward(x, doy, mask)
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
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        confidence, logits, last_emb, _ = self.forward(x, doy, mask)
        with torch.no_grad():
            logits = self.backbone.classifier(last_emb)
            probs = F.softmax(logits, dim=1)
            tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()
        loss = self.mse_loss(confidence.squeeze(), tcp_target)
        self.log("p2_val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        lr_lambda = _lr_lambda_factory(
            self.train_dataset_size, self.batch_size,
            self.max_epochs, self.num_warmup_epochs, self.base_lr,
        )
        return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}]


class Phase3_ConfidNetFinetuning(pl.LightningModule):
    def __init__(self, pretrained_confidnet: Phase2_ConfidNet, train_dataset_size: int,
                 max_epochs=100, batch_size=512, num_warmup_epochs=10, base_lr=1e-4):
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
        pooled, _, _ = self.backbone(x, doy, mask)
        confidence = self.confid_net(pooled)
        with torch.no_grad():
            pooled_frozen, logits_frozen, _ = self.backbone_frozen(x, doy, mask)
        return confidence, logits_frozen, pooled_frozen, pooled_frozen

    def training_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        confidence, logits_frozen, _, _ = self.forward(x, doy, mask)
        probs = F.softmax(logits_frozen, dim=1)
        tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()
        loss = self.mse_loss(confidence.squeeze(), tcp_target)
        self.log("p3_conf_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("p3_lr", lr, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        confidence, logits_frozen, _, _ = self.forward(x, doy, mask)
        probs = F.softmax(logits_frozen, dim=1)
        tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()
        loss = self.mse_loss(confidence.squeeze(), tcp_target)
        self.log("p3_val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr / 10)
        lr_lambda = _lr_lambda_factory(
            self.train_dataset_size, self.batch_size,
            self.max_epochs, self.num_warmup_epochs, self.base_lr,
        )
        return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}]


# ---------------------------------------------------------------------------
# KFold helpers
# ---------------------------------------------------------------------------

def _make_dataloader(dataset, batch_size, sampler=None, shuffle=False, num_workers=NUM_WORKERS):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=shuffle, num_workers=num_workers, pin_memory=True,
    )


def _make_sampler(dataset):
    weights = dataset.get_weights_for_WeightedRandomSampler()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.inference_mode()
def _get_embeddings(model, dataloader, device):
    """Return (preds, labels, embeddings) arrays from a Phase1_Classifier."""
    model.eval().to(device)
    preds_l, labels_l, embs_l = [], [], []
    for batch in tqdm(dataloader, leave=False, desc="  embeddings", disable=not IS_TTY):
        x = batch["x"].to(device)
        doy = batch["doy"].to(device)
        mask = batch["mask"].to(device)
        logits, pooled = model(x, doy, mask)
        preds_l.append(torch.argmax(logits, dim=1).cpu().numpy())
        labels_l.append(batch["y"].cpu().numpy().squeeze())
        embs_l.append(pooled.cpu().float().numpy())
    return (
        np.concatenate(preds_l),
        np.concatenate(labels_l),
        np.concatenate(embs_l),
    )


def _fit_gmm_robust(X, n_components, cls_label):
    """
    Fit GaussianMixture with progressive fallback on convergence/numerical failure.

    Attempt order:
      1. Requested n_components, k-means++ init, reg_covar=1e-4
      2. n_components=1, random init, reg_covar=1e-3  (most convergence issues go away)
      3. n_components=1, random init, reg_covar=1e-1, covariance_type='diag'

    Returns fitted GMM, or None if all attempts fail (caller treats class as non-anomalous).
    """
    from sklearn.exceptions import ConvergenceWarning

    attempts = [
        dict(n_components=n_components, init_params="k-means++",
             max_iter=300, n_init=3, reg_covar=1e-4, covariance_type="full"),
        dict(n_components=1, init_params="random",
             max_iter=500, n_init=5, reg_covar=1e-3, covariance_type="full"),
        dict(n_components=1, init_params="random",
             max_iter=500, n_init=5, reg_covar=1e-1, covariance_type="diag"),
    ]

    for i, kwargs in enumerate(attempts):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                gmm = GaussianMixture(random_state=42, **kwargs).fit(X)

            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                if i < len(attempts) - 1:
                    print(
                        f"  [GMM WARNING] class={cls_label!r}, attempt {i+1}: "
                        f"ConvergenceWarning with {kwargs}. Retrying..."
                    )
                    continue
                else:
                    print(
                        f"  [GMM WARNING] class={cls_label!r}: all attempts converged poorly. "
                        f"Using last fit — class treated as non-anomalous."
                    )
                    return None

            return gmm

        except Exception as exc:
            print(
                f"  [GMM WARNING] class={cls_label!r}, attempt {i+1}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        f"  [GMM WARNING] class={cls_label!r}: all {len(attempts)} attempts failed. "
        f"Skipping — all val samples of this class treated as non-anomalous."
    )
    return None


def _gmm_anomaly_flags(train_preds, train_labels, train_embs, val_preds, val_embs):
    """
    Per-class GMM fitted on correctly-predicted train samples.
    Threshold = 2.5th percentile of train GMM scores.
    Returns boolean anomaly array for val samples.

    If GMM fitting fails for a class, that class's val samples are NOT flagged as anomalies.
    """
    val_anomaly = np.zeros(len(val_preds), dtype=bool)

    for cls in np.unique(train_labels):
        correct_mask = (train_preds == cls) & (train_labels == cls)
        X_cls = train_embs[correct_mask]

        if len(X_cls) < 5:
            # Too few samples — skip, treat as non-anomalous
            continue

        n_comp = min(3, max(1, len(X_cls) // 20))
        gmm = _fit_gmm_robust(X_cls, n_comp, cls)
        if gmm is None:
            continue  # all val samples of this class stay False (non-anomalous)

        try:
            threshold = np.percentile(gmm.score_samples(X_cls), 2.5)
            val_cls_mask = val_preds == cls
            if val_cls_mask.any():
                scores = gmm.score_samples(val_embs[val_cls_mask])
                val_anomaly[val_cls_mask] = scores < threshold
        except Exception as exc:
            print(
                f"  [GMM WARNING] class={cls!r}: scoring failed ({type(exc).__name__}: {exc}). "
                f"Class treated as non-anomalous."
            )

    return val_anomaly


def _train_kfold_phase1(fold_train_ds, fold_val_ds, fold_idx):
    """Train a Phase1_Classifier for one fold. Returns best model."""
    sampler = _make_sampler(fold_train_ds)
    dl_train = _make_dataloader(fold_train_ds, BATCH_SIZE, sampler=sampler)
    dl_val = _make_dataloader(fold_val_ds, BATCH_SIZE)

    model = Phase1_Classifier(
        num_classes=NUM_CLASSES,
        train_dataset_size=len(fold_train_ds),
        max_epochs=KFOLD_MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        num_warmup_epochs=NUM_WARMUP_EPOCHS,
        base_lr=BASE_LR,
    )

    ckpt_dir = tempfile.mkdtemp(prefix=f"kfold_fold{fold_idx}_")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, monitor="p1_val_loss", mode="min", filename="best"
    )
    es_cb = EarlyStopping(monitor="p1_val_loss", patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs=KFOLD_MAX_EPOCHS,
        min_epochs=2 * NUM_WARMUP_EPOCHS,
        accelerator="gpu",
        devices=[GPU_ID],
        precision="bf16-mixed",
        callbacks=[ckpt_cb, es_cb],
        enable_progress_bar=IS_TTY,
        logger=False,  # No MLflow during KFold
        log_every_n_steps=5,
    )
    trainer.fit(model, dl_train, dl_val)
    best_model = Phase1_Classifier.load_from_checkpoint(ckpt_cb.best_model_path)
    return best_model


# ---------------------------------------------------------------------------
# KFold anomaly detection
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"PHASE 1: KFold anomaly detection  ({N_FOLDS} folds, dataset={DATASET})")
print("=" * 60)

device = f"cuda:{GPU_ID}"
anomaly_flags = np.zeros(len(trainval_aug), dtype=bool)

if DATASET == "brazil":
    groups = gdf_trainval["CD_MUN"].values
    positions = np.arange(len(gdf_trainval))
    kf_splits = list(GroupKFold(n_splits=N_FOLDS).split(positions, groups=groups))
else:
    kf_splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(
        np.arange(len(trainval_aug))
    ))

for fold_idx, (fold_train_pos, fold_val_pos) in enumerate(kf_splits):
    print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

    fold_train_ds = _SubsetWithWeights(trainval_aug, fold_train_pos)
    fold_val_ds = _SubsetWithWeights(trainval_noaug, fold_val_pos)
    fold_train_noaug = _SubsetWithWeights(trainval_noaug, fold_train_pos)

    best_model = _train_kfold_phase1(fold_train_ds, fold_val_ds, fold_idx)

    print("  Extracting embeddings for GMM...")
    dl_train_noaug = _make_dataloader(fold_train_noaug, BATCH_SIZE)
    dl_val_noaug = _make_dataloader(fold_val_ds, BATCH_SIZE)

    train_preds, train_labels, train_embs = _get_embeddings(best_model, dl_train_noaug, device)
    val_preds, val_labels, val_embs = _get_embeddings(best_model, dl_val_noaug, device)

    fold_anomaly = _gmm_anomaly_flags(train_preds, train_labels, train_embs, val_preds, val_embs)
    anomaly_flags[fold_val_pos] = fold_anomaly

    n_anom = fold_anomaly.sum()
    print(f"  Anomalies: {n_anom}/{len(fold_anomaly)} ({100*n_anom/max(1,len(fold_anomaly)):.1f}%)")

print(f"\nTotal anomalies in train+val: {anomaly_flags.sum()}/{len(anomaly_flags)} "
      f"({100*anomaly_flags.mean():.1f}%)")

# ---------------------------------------------------------------------------
# Build clean datasets
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 2: Building clean datasets (no anomalies in train+val)")
print("=" * 60)

train_anomaly_flags = anomaly_flags[:N_TRAIN]
val_anomaly_flags = anomaly_flags[N_TRAIN:]

clean_train_idx = np.where(~train_anomaly_flags)[0]
clean_val_idx = N_TRAIN + np.where(~val_anomaly_flags)[0]
clean_trainval_idx = np.where(~anomaly_flags)[0]

N_VAL = len(trainval_aug) - N_TRAIN
clean_val_pos = np.where(~val_anomaly_flags)[0]  # positions within val slice (0-based)

print(f"Clean train: {len(clean_train_idx)}/{N_TRAIN} "
      f"({100*(1-train_anomaly_flags.mean()):.1f}% kept)")
print(f"Clean val:   {len(clean_val_pos)}/{N_VAL} "
      f"({100*(1-val_anomaly_flags.mean()):.1f}% kept)")

if DATASET == "brazil":
    # Subsets using positions within trainval_(aug|noaug) datasets
    train_dataset = _SubsetWithWeights(trainval_aug, clean_train_idx)
    val_dataset = _SubsetWithWeights(trainval_noaug, clean_val_idx)
    train_val_dataset = _SubsetWithWeights(trainval_noaug, clean_trainval_idx)

    # Attach gdf subsets so predict_and_save_predictions can merge spatial info.
    # reset_index so positional concat with predictions_df aligns correctly.
    train_dataset.gdf = trainval_noaug.gdf.iloc[clean_train_idx].reset_index(drop=True)
    val_dataset.gdf = trainval_noaug.gdf.iloc[clean_val_idx].reset_index(drop=True)
    train_val_dataset.gdf = trainval_noaug.gdf.iloc[clean_trainval_idx].reset_index(drop=True)

else:
    # Texas / California
    clean_train_ts = tv_ts[:N_TRAIN][~train_anomaly_flags]
    clean_train_doys = tv_doys[:N_TRAIN][~train_anomaly_flags]
    clean_train_ys = tv_ys[:N_TRAIN][~train_anomaly_flags]

    clean_val_ts = tv_ts[N_TRAIN:][~val_anomaly_flags]
    clean_val_doys = tv_doys[N_TRAIN:][~val_anomaly_flags]
    clean_val_ys = tv_ys[N_TRAIN:][~val_anomaly_flags]

    train_dataset = NpzSubset(clean_train_ts, clean_train_doys, clean_train_ys, transform=aug_transforms)
    val_dataset = NpzSubset(clean_val_ts, clean_val_doys, clean_val_ys, transform=transforms)
    train_val_dataset = NpzSubset(
        np.concatenate([clean_train_ts, clean_val_ts]),
        np.concatenate([clean_train_doys, clean_val_doys]),
        np.concatenate([clean_train_ys, clean_val_ys]),
        transform=transforms,
    )

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
sample_weights = train_dataset.get_weights_for_WeightedRandomSampler()
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
)
train_val_dataloader = torch.utils.data.DataLoader(
    train_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
)

# ---------------------------------------------------------------------------
# MLflow logger (used for all 3 final phases)
# ---------------------------------------------------------------------------
mlflow_logger = MLFlowLogger(
    experiment_name=EXPERIMENT_NAME,
    tags=TAGS,
    run_name=RUN_NAME,
    tracking_uri=mlflow.get_tracking_uri(),  # explicit — uses MLFLOW_TRACKING_URI env var
)

# Log anomaly stats
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "kfold_anomaly_rate_train", train_anomaly_flags.mean())
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "kfold_anomaly_rate_val", val_anomaly_flags.mean())
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "kfold_anomaly_rate_total", anomaly_flags.mean())
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "clean_train_size", len(clean_train_idx))
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "clean_val_size", int(len(clean_val_pos)))

# ---------------------------------------------------------------------------
# Phase 1: Supervised classification on clean data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3: Supervised training on clean data — Phase 1 (Classifier)")
print("=" * 60)

model_phase1 = Phase1_Classifier(
    num_classes=NUM_CLASSES,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

checkpoint_cb_p1 = ModelCheckpoint(monitor="p1_val_loss", filename="best_classifier", mode="min")
early_stopping_cb_p1 = EarlyStopping(monitor="p1_val_loss", patience=10, mode="min")
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
    enable_progress_bar=IS_TTY,
)

trainer_p1.fit(model_phase1, train_dataloader, val_dataloader)
best_model_p1 = Phase1_Classifier.load_from_checkpoint(checkpoint_cb_p1.best_model_path)

# ---------------------------------------------------------------------------
# Phase 2: ConfidNet on clean data
# ---------------------------------------------------------------------------
print("\n--- Phase 2: ConfidNet ---")

model_phase2 = Phase2_ConfidNet(
    pretrained_model=best_model_p1,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

checkpoint_cb_p2 = ModelCheckpoint(monitor="p2_val_loss", filename="best_confidnet", mode="min")
early_stopping_cb_p2 = EarlyStopping(monitor="p2_val_loss", patience=10, mode="min")

trainer_p2 = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu",
    devices=[GPU_ID],
    precision="bf16-mixed",
    callbacks=[checkpoint_cb_p2, early_stopping_cb_p2],
    logger=mlflow_logger,
    log_every_n_steps=5,
    enable_progress_bar=IS_TTY,
)

trainer_p2.fit(model_phase2, train_dataloader, val_dataloader)
best_model_p2 = Phase2_ConfidNet.load_from_checkpoint(
    checkpoint_cb_p2.best_model_path, pretrained_model=best_model_p1
)

# ---------------------------------------------------------------------------
# Phase 3: ConfidNet fine-tuning on clean data
# ---------------------------------------------------------------------------
print("\n--- Phase 3: ConfidNet fine-tuning ---")

model_phase3 = Phase3_ConfidNetFinetuning(
    pretrained_confidnet=best_model_p2,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

checkpoint_cb_p3 = ModelCheckpoint(monitor="p3_val_loss", filename="best_confidnet_finetuned", mode="min")
early_stopping_cb_p3 = EarlyStopping(monitor="p3_val_loss", patience=10, mode="min")

trainer_p3 = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu",
    devices=[GPU_ID],
    precision="bf16-mixed",
    callbacks=[checkpoint_cb_p3, early_stopping_cb_p3],
    logger=mlflow_logger,
    log_every_n_steps=5,
    enable_progress_bar=IS_TTY,
)

trainer_p3.fit(model_phase3, train_dataloader, val_dataloader)
best_model_p3 = Phase3_ConfidNetFinetuning.load_from_checkpoint(
    checkpoint_cb_p3.best_model_path, pretrained_confidnet=best_model_p2
)

trainer_p3.validate(model=best_model_p3, dataloaders=val_dataloader)
best_model_p3.eval()

# ---------------------------------------------------------------------------
# Predictions & evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 4: Generating predictions and running GMM evaluation")
print("=" * 60)

# train_val_gdf  — clean train set WITHOUT aug  (used to fit GMM in run_gemos)
train_val_gdf = predict_and_save_predictions(
    best_model_p3, train_val_dataloader, train_val_dataset,
    mlflow_logger, "train", class_map, to_print=False,
)

# train_gdf — clean train set WITH aug (same data, augmented inference)
train_gdf = predict_and_save_predictions(
    best_model_p3, train_dataloader, train_dataset,
    mlflow_logger, "train_aug", class_map, to_print=False,
)

# val_gdf — clean val set
val_gdf = predict_and_save_predictions(
    best_model_p3, val_dataloader, val_dataset,
    mlflow_logger, "val", class_map, to_print=False,
)

# test_gdf — FULL test set (all samples, including potential anomalies)
# run_gemos will score and report separately with/without GMM-flagged test anomalies
test_gdf = predict_and_save_predictions(
    best_model_p3, test_dataloader, test_dataset,
    mlflow_logger, "test", class_map, to_print=True,
)

# GMM fitted on clean train → scores clean val + full test
# Metrics reported: test_accuracy (all), test_accuracy_wo_anomaly (GMM-filtered)
run_gemos(train_gdf, train_val_gdf, val_gdf, test_gdf, mlflow_logger)
save_pytorch_model(best_model_p3, mlflow_logger)

print("\nDone.")
