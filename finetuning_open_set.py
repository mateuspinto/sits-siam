import os
_HALF_CORES = str(max(1, os.cpu_count() // 2))
os.environ.setdefault("OMP_NUM_THREADS",      _HALF_CORES)
os.environ.setdefault("MKL_NUM_THREADS",      _HALF_CORES)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _HALF_CORES)
os.environ.setdefault("NUMEXPR_NUM_THREADS",  _HALF_CORES)
NUM_WORKERS = max(1, int(_HALF_CORES) // 2)

import copy
import math
import argparse
import sys
import logging

IS_TTY = sys.stdout.isatty()

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

import geopandas as gpd
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
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
    load_pretrain_weights,
    predict_and_save_predictions,
    setup_seed,
    run_gemos,
    save_pytorch_model,
    split_with_percent_and_class_coverage,
    string_confusion_matrix,
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

# ─── Args ─────────────────────────────────────────────────────────────────────
BATCHED_ARGS_PARSER = argparse.ArgumentParser(add_help=False)
BATCHED_ARGS_PARSER.add_argument("--train_percent", type=float, default=70.0)
BATCHED_ARGS_PARSER.add_argument(
    "--model_name", type=str,
    choices=["MAMBA", "BERT", "BERTPP", "LSTM", "CNN"], default="MAMBA",
)
BATCHED_ARGS_PARSER.add_argument(
    "--dataset", type=str,
    choices=["brazil", "california", "texas", "pastis"], default="brazil",
)
BATCHED_ARGS_PARSER.add_argument(
    "--pretrain", type=str,
    choices=["off", "reconstruct", "MoCo", "PMSN", "FastSiam"], default="off",
)
BATCHED_ARGS_PARSER.add_argument("--gpu", type=int, default=0)
_parsed_args, _ = BATCHED_ARGS_PARSER.parse_known_args()
TRAIN_PERCENT = float(_parsed_args.train_percent)
GPU_ID        = _parsed_args.gpu
DATASET       = _parsed_args.dataset
MODEL_NAME    = _parsed_args.model_name
PRETRAIN      = _parsed_args.pretrain

BATCH_SIZE         = 2 * 512
MAX_EPOCHS         = 100
if TRAIN_PERCENT <= 1:
    MAX_EPOCHS = 200
NUM_WARMUP_EPOCHS  = 10
BASE_LR            = 1e-4
OPEN_SET_THRESHOLD = 0.15

TAGS = {
    "dataset":             str(DATASET),
    "batch_size":          str(BATCH_SIZE),
    "max_epochs":          str(MAX_EPOCHS),
    "num_warmup_epochs":   str(NUM_WARMUP_EPOCHS),
    "base_lr":             str(BASE_LR),
    "train_percent":       str(TRAIN_PERCENT),
    "model_name":          str(MODEL_NAME),
    "pretrain":            str(PRETRAIN),
    "open_set_threshold":  str(OPEN_SET_THRESHOLD),
}
RUN_NAME        = f"{MODEL_NAME}-{TRAIN_PERCENT}"
EXPERIMENT_NAME = f"{DATASET}-openset-finetuning"
if PRETRAIN != "off":
    RUN_NAME += f"-{PRETRAIN}"

# ─── Transforms ───────────────────────────────────────────────────────────────
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

# ─── Open-set helpers ─────────────────────────────────────────────────────────

def get_open_classes(class_counts: dict, threshold: float = 0.15) -> set:
    """Smallest-count classes whose sum >= threshold * total. Determined globally."""
    total = sum(class_counts.values())
    target = threshold * total
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    open_classes, accumulated = set(), 0
    for cls_id, count in sorted_classes:
        if accumulated >= target:
            break
        open_classes.add(cls_id)
        accumulated += count
    return open_classes


class NpzDataset(torch.utils.data.Dataset):
    """Numpy-array-backed dataset; same interface as AgriGEELiteDataset."""
    def __init__(self, ts, doys, ys, transform=None):
        self.ts        = ts.astype(np.float16)
        self.doys      = doys.astype(np.int16)
        self.ys        = ys.astype(np.int64)
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
        return torch.from_numpy(
            np.array([weight_map[t] for t in self.ys])
        ).double()

# ─── Dataset loading ──────────────────────────────────────────────────────────

if DATASET == "brazil":
    gdf     = gpd.read_parquet("data/agl/gdf.parquet")
    sits_df = pd.read_parquet("data/agl/df_sits.parquet")

    orig_class_map = (
        gdf[["crop_class", "crop_number"]]
        .drop_duplicates()
        .set_index("crop_number")["crop_class"]
        .to_dict()
    )

    # Determine open/closed globally (full dataset before any split)
    class_counts   = gdf["crop_number"].value_counts().to_dict()
    open_class_ids = get_open_classes(class_counts, OPEN_SET_THRESHOLD)
    closed_class_ids = sorted(set(class_counts.keys()) - open_class_ids)
    N_CLOSED = len(closed_class_ids)

    closed_label_map = {orig: new for new, orig in enumerate(closed_class_ids)}
    open_label_map   = {orig: N_CLOSED + i for i, orig in enumerate(sorted(open_class_ids))}

    closed_class_map      = {closed_label_map[k]: v for k, v in orig_class_map.items() if k in closed_class_ids}
    open_class_map_display = {open_label_map[k]: v  for k, v in orig_class_map.items() if k in open_class_ids}
    full_class_map        = {**closed_class_map, **open_class_map_display}

    print(f"Open  classes ({len(open_class_ids)}): {[orig_class_map[c] for c in sorted(open_class_ids)]}")
    print(f"Closed classes ({N_CLOSED}): {list(closed_class_map.values())}")

    # Split
    if abs(TRAIN_PERCENT - 70.0) < 1e-9:
        unique_mun = gdf["CD_MUN"].unique()
        mun_train, mun_temp = train_test_split(unique_mun, test_size=0.30, random_state=13)
        mun_val, mun_test   = train_test_split(mun_temp,  test_size=0.50, random_state=13)
        gdf_train_all = gdf[gdf["CD_MUN"].isin(mun_train)].copy()
        gdf_val_all   = gdf[gdf["CD_MUN"].isin(mun_val)].copy()
        gdf_test_all  = gdf[gdf["CD_MUN"].isin(mun_test)].copy()
    else:
        gdf_train_all, gdf_val_all, gdf_test_all = split_with_percent_and_class_coverage(
            gdf, percent=TRAIN_PERCENT, max_attempts=500
        )

    # Open samples from train/val are moved to test
    is_open_tr = gdf_train_all["crop_number"].isin(open_class_ids)
    is_open_va = gdf_val_all["crop_number"].isin(open_class_ids)

    gdf_train = gdf_train_all[~is_open_tr].copy()
    gdf_val   = gdf_val_all[~is_open_va].copy()

    gdf_open_extra = pd.concat([
        gdf_train_all[is_open_tr],
        gdf_val_all[is_open_va],
    ]).copy()

    gdf_test_closed = gdf_test_all[~gdf_test_all["crop_number"].isin(open_class_ids)].copy()
    gdf_test_open   = pd.concat([
        gdf_test_all[gdf_test_all["crop_number"].isin(open_class_ids)],
        gdf_open_extra,
    ]).copy()

    # Remap labels
    gdf_train["crop_number"]        = gdf_train["crop_number"].map(closed_label_map)
    gdf_val["crop_number"]          = gdf_val["crop_number"].map(closed_label_map)
    gdf_test_closed["crop_number"]  = gdf_test_closed["crop_number"].map(closed_label_map)
    gdf_test_open["crop_number"]    = gdf_test_open["crop_number"].map(open_label_map)
    gdf_test = pd.concat([gdf_test_closed, gdf_test_open]).reset_index(drop=True)

    # Datasets
    train_dataset     = AgriGEELiteDataset(gdf_train, sits_df, transform=aug_transforms, timestamp_processing="days_after_start")
    val_dataset       = AgriGEELiteDataset(gdf_val,   sits_df, transform=transforms,     timestamp_processing="days_after_start")
    train_val_dataset = AgriGEELiteDataset(gdf_train, sits_df, transform=transforms,     timestamp_processing="days_after_start")
    test_dataset      = AgriGEELiteDataset(gdf_test,  sits_df, transform=transforms,     timestamp_processing="days_after_start")

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

    _tr  = np.load(f"data/{DATASET}_{split_string}/train.npz")
    _va  = np.load(f"data/{DATASET}_{split_string}/val.npz")
    _te  = np.load(f"data/{DATASET}_{split_string}/test.npz")

    all_ys = np.concatenate([_tr["ys"], _va["ys"], _te["ys"]])
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(all_ys, return_counts=True))}

    open_class_ids   = get_open_classes(class_counts, OPEN_SET_THRESHOLD)
    closed_class_ids = sorted(set(class_counts.keys()) - open_class_ids)
    N_CLOSED = len(closed_class_ids)

    _ref         = SitsFinetuneDatasetFromNpz(f"data/{DATASET}_{split_string}/train.npz")
    _class_names = _ref.get_class_names()
    orig_class_map = {i: _class_names[i] for i in range(len(_class_names))}

    closed_label_map = {orig: new for new, orig in enumerate(closed_class_ids)}
    open_label_map   = {orig: N_CLOSED + i for i, orig in enumerate(sorted(open_class_ids))}

    closed_class_map       = {closed_label_map[k]: v for k, v in orig_class_map.items() if k in closed_class_ids}
    open_class_map_display = {open_label_map[k]:   v for k, v in orig_class_map.items() if k in open_class_ids}
    full_class_map         = {**closed_class_map, **open_class_map_display}

    print(f"Open  classes ({len(open_class_ids)}): {[orig_class_map[c] for c in sorted(open_class_ids)]}")
    print(f"Closed classes ({N_CLOSED}): {list(closed_class_map.values())}")

    def _remap(ys, lmap):
        return np.array([lmap[int(y)] for y in ys], dtype=np.int64)

    tr_ts, tr_doys, tr_ys = _tr["ts"], _tr["doys"], _tr["ys"]
    va_ts, va_doys, va_ys = _va["ts"], _va["doys"], _va["ys"]
    te_ts, te_doys, te_ys = _te["ts"], _te["doys"], _te["ys"]

    tr_open = np.isin(tr_ys, list(open_class_ids))
    va_open = np.isin(va_ys, list(open_class_ids))
    te_open = np.isin(te_ys, list(open_class_ids))

    # Closed train / val (remapped 0..N_CLOSED-1)
    tr_ts_c = tr_ts[~tr_open]; tr_doys_c = tr_doys[~tr_open]; tr_ys_c = _remap(tr_ys[~tr_open], closed_label_map)
    va_ts_c = va_ts[~va_open]; va_doys_c = va_doys[~va_open]; va_ys_c = _remap(va_ys[~va_open], closed_label_map)

    # Closed test samples
    te_ts_c = te_ts[~te_open]; te_doys_c = te_doys[~te_open]; te_ys_c = _remap(te_ys[~te_open], closed_label_map)

    # All open samples → remapped to N_CLOSED..N_CLOSED+N_OPEN-1
    op_ts   = np.concatenate([tr_ts[tr_open],  va_ts[va_open],  te_ts[te_open]])
    op_doys = np.concatenate([tr_doys[tr_open], va_doys[va_open], te_doys[te_open]])
    op_ys   = _remap(np.concatenate([tr_ys[tr_open], va_ys[va_open], te_ys[te_open]]), open_label_map)

    full_te_ts   = np.concatenate([te_ts_c,  op_ts])
    full_te_doys = np.concatenate([te_doys_c, op_doys])
    full_te_ys   = np.concatenate([te_ys_c,  op_ys])

    train_dataset     = NpzDataset(tr_ts_c,    tr_doys_c,    tr_ys_c,    transform=aug_transforms)
    val_dataset       = NpzDataset(va_ts_c,    va_doys_c,    va_ys_c,    transform=transforms)
    train_val_dataset = NpzDataset(tr_ts_c,    tr_doys_c,    tr_ys_c,    transform=transforms)
    test_dataset      = NpzDataset(full_te_ts, full_te_doys, full_te_ys, transform=transforms)

else:
    raise ValueError(f"Dataset {DATASET} not recognized.")

print(f"Train (closed): {len(train_dataset)}  Val (closed): {len(val_dataset)}  Test (all): {len(test_dataset)}")

# ─── Model phases (identical to finetuning.py) ────────────────────────────────

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
        self.criterion  = nn.CrossEntropyLoss()
        self.train_acc  = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.val_acc    = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size; self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs; self.base_lr = base_lr

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
        self.log("p1_lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                 prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        logits, _ = self.forward(x, doy, mask)
        loss = self.criterion(logits, y)
        self.log("p1_val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("p1_val_acc", self.val_acc(logits, y), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_steps   = steps_per_epoch * self.max_epochs
        warmup_steps  = steps_per_epoch * self.num_warmup_epochs

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                alpha = float(current_step) / float(max(1, warmup_steps))
                return (1.0 / 1000) * (1 - alpha) + alpha
            progress = min(1.0, float(current_step - warmup_steps) /
                           float(max(1, total_steps - warmup_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda),
                               "interval": "step", "frequency": 1}]


class Phase2_ConfidNet(pl.LightningModule):
    def __init__(self, pretrained_model: Phase1_Classifier, train_dataset_size: int,
                 max_epochs=100, batch_size=512, num_warmup_epochs=10, base_lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])
        self.backbone = pretrained_model.backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.confid_net = nn.Sequential(
            nn.Linear(self.backbone.hidden_dim, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 1),   nn.Sigmoid(),
        )
        self.mse_loss = nn.MSELoss()
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size; self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs; self.base_lr = base_lr

    def forward(self, x, doy, mask):
        with torch.no_grad():
            pooled, logits, _ = self.backbone(x, doy, mask)
        return self.confid_net(pooled), logits, pooled, pooled

    def training_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        confidence, logits, last_emb, _ = self.forward(x, doy, mask)
        with torch.no_grad():
            self.backbone.classifier.eval()
            logits = self.backbone.classifier(last_emb.detach())
            tcp_target = F.softmax(logits, dim=1).gather(1, y.unsqueeze(1)).squeeze()
        loss = self.mse_loss(confidence.squeeze(), tcp_target)
        self.log("p2_conf_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("p2_lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                 prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        confidence, logits, last_emb, _ = self.forward(x, doy, mask)
        with torch.no_grad():
            logits = self.backbone.classifier(last_emb)
            tcp_target = F.softmax(logits, dim=1).gather(1, y.unsqueeze(1)).squeeze()
        self.log("p2_val_loss", self.mse_loss(confidence.squeeze(), tcp_target),
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_steps  = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.num_warmup_epochs

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                alpha = float(current_step) / float(max(1, warmup_steps))
                return (1.0 / 1000) * (1 - alpha) + alpha
            progress = min(1.0, float(current_step - warmup_steps) /
                           float(max(1, total_steps - warmup_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda),
                               "interval": "step", "frequency": 1}]


class Phase3_ConfidNetFinetuning(pl.LightningModule):
    def __init__(self, pretrained_confidnet: Phase2_ConfidNet, train_dataset_size: int,
                 max_epochs=100, batch_size=512, num_warmup_epochs=10, base_lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_confidnet"])
        self.backbone_frozen = copy.deepcopy(pretrained_confidnet.backbone)
        self.backbone_frozen.eval()
        for p in self.backbone_frozen.parameters():
            p.requires_grad = False
        self.backbone = copy.deepcopy(pretrained_confidnet.backbone)
        for m in self.backbone.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        self.confid_net = pretrained_confidnet.confid_net
        for p in self.backbone.parameters():
            p.requires_grad = True
        for p in self.confid_net.parameters():
            p.requires_grad = True
        self.backbone.train(); self.confid_net.train()
        self.mse_loss = nn.MSELoss()
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size; self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs; self.base_lr = base_lr

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
        self.log("p3_lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                 prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, doy, mask, y = batch["x"], batch["doy"], batch["mask"], batch["y"].squeeze()
        confidence, logits_frozen, _, _ = self.forward(x, doy, mask)
        probs = F.softmax(logits_frozen, dim=1)
        tcp_target = probs.gather(1, y.unsqueeze(1)).squeeze()
        self.log("p3_val_loss", self.mse_loss(confidence.squeeze(), tcp_target),
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr / 10)
        steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_steps  = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.num_warmup_epochs

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                alpha = float(current_step) / float(max(1, warmup_steps))
                return (1.0 / 1000) * (1 - alpha) + alpha
            progress = min(1.0, float(current_step - warmup_steps) /
                           float(max(1, total_steps - warmup_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda),
                               "interval": "step", "frequency": 1}]

# ─── DataLoaders ──────────────────────────────────────────────────────────────

sample_weights = train_dataset.get_weights_for_WeightedRandomSampler()
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)
train_val_dataloader = torch.utils.data.DataLoader(
    train_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

# ─── MLflow ───────────────────────────────────────────────────────────────────

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow_logger = MLFlowLogger(
    experiment_name=EXPERIMENT_NAME,
    tags=TAGS,
    run_name=RUN_NAME,
    tracking_uri=mlflow.get_tracking_uri(),
)

# Log open/closed class split to MLflow
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "n_closed_classes", N_CLOSED)
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "n_open_classes",   len(open_class_ids))
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "train_size_closed", len(train_dataset))
mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "test_size_all",     len(test_dataset))

# ─── Phase 1: Classifier ──────────────────────────────────────────────────────

print("--- PHASE 1: Classification (closed set) ---")
model_phase1 = Phase1_Classifier(
    num_classes=N_CLOSED,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS,
    base_lr=BASE_LR,
)

checkpoint_cb_p1 = ModelCheckpoint(monitor="p1_val_loss", filename="best_classifier", mode="min")
early_stopping_cb_p1 = EarlyStopping(monitor="p1_val_loss", patience=10, mode="min")
devicestats_monitor  = DeviceStatsMonitor(cpu_stats=False)

trainer_p1 = pl.Trainer(
    max_epochs=MAX_EPOCHS, min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu", devices=[GPU_ID], precision="bf16-mixed",
    callbacks=[checkpoint_cb_p1, early_stopping_cb_p1, devicestats_monitor],
    logger=mlflow_logger, log_every_n_steps=5, enable_progress_bar=IS_TTY,
)
trainer_p1.fit(model_phase1, train_dataloader, val_dataloader)
best_model_p1 = Phase1_Classifier.load_from_checkpoint(checkpoint_cb_p1.best_model_path)

# ─── Phase 2: ConfidNet ───────────────────────────────────────────────────────

print("--- PHASE 2: ConfidNet ---")
model_phase2 = Phase2_ConfidNet(
    pretrained_model=best_model_p1,
    max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS, base_lr=BASE_LR,
)
checkpoint_cb_p2    = ModelCheckpoint(monitor="p2_val_loss", filename="best_confidnet", mode="min")
early_stopping_cb_p2 = EarlyStopping(monitor="p2_val_loss", patience=10, mode="min")

trainer_p2 = pl.Trainer(
    max_epochs=MAX_EPOCHS, min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu", devices=[GPU_ID], precision="bf16-mixed",
    callbacks=[checkpoint_cb_p2, early_stopping_cb_p2],
    logger=mlflow_logger, log_every_n_steps=5, enable_progress_bar=IS_TTY,
)
trainer_p2.fit(model_phase2, train_dataloader, val_dataloader)
best_model_p2 = Phase2_ConfidNet.load_from_checkpoint(
    checkpoint_cb_p2.best_model_path, pretrained_model=best_model_p1
)

# ─── Phase 3: ConfidNet fine-tuning ───────────────────────────────────────────

print("--- PHASE 3: ConfidNet fine-tuning ---")
model_phase3 = Phase3_ConfidNetFinetuning(
    pretrained_confidnet=best_model_p2,
    max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=NUM_WARMUP_EPOCHS, base_lr=BASE_LR,
)
checkpoint_cb_p3    = ModelCheckpoint(monitor="p3_val_loss", filename="best_confidnet_finetuned", mode="min")
early_stopping_cb_p3 = EarlyStopping(monitor="p3_val_loss", patience=10, mode="min")

trainer_p3 = pl.Trainer(
    max_epochs=MAX_EPOCHS, min_epochs=2 * NUM_WARMUP_EPOCHS,
    accelerator="gpu", devices=[GPU_ID], precision="bf16-mixed",
    callbacks=[checkpoint_cb_p3, early_stopping_cb_p3],
    logger=mlflow_logger, log_every_n_steps=5, enable_progress_bar=IS_TTY,
)
trainer_p3.fit(model_phase3, train_dataloader, val_dataloader)
best_model_p3 = Phase3_ConfidNetFinetuning.load_from_checkpoint(
    checkpoint_cb_p3.best_model_path, pretrained_confidnet=best_model_p2
)

trainer_p3.validate(model=best_model_p3, dataloaders=val_dataloader)
best_model_p3.eval()

# ─── Predictions ──────────────────────────────────────────────────────────────
# Train/val: closed only → closed_class_map
# Test: all samples (closed + open) → full_class_map

print("--- Generating predictions ---")

train_gdf = predict_and_save_predictions(
    best_model_p3, train_dataloader, train_dataset,
    mlflow_logger, "train_aug", closed_class_map, to_print=False,
)
train_val_gdf = predict_and_save_predictions(
    best_model_p3, train_val_dataloader, train_val_dataset,
    mlflow_logger, "train", closed_class_map, to_print=False,
)
val_gdf = predict_and_save_predictions(
    best_model_p3, val_dataloader, val_dataset,
    mlflow_logger, "val", closed_class_map, to_print=False,
)
test_gdf = predict_and_save_predictions(
    best_model_p3, test_dataloader, test_dataset,
    mlflow_logger, "test", full_class_map, to_print=True,
)

# ─── GEMOS (GMM fit on closed only, scored on all test) ───────────────────────

run_gemos(train_gdf, train_val_gdf, val_gdf, test_gdf, mlflow_logger)

# ─── Open-set detection metrics ───────────────────────────────────────────────

_SEP = "=" * 60
open_class_names = set(open_class_map_display.values())
is_open = test_gdf["y_true"].isin(open_class_names)

print(f"\n{_SEP}")
print(f"OPEN-SET SUMMARY  ({is_open.sum()} open / {(~is_open).sum()} closed in test)")
print(_SEP)

if is_open.any() and (~is_open).any():
    # negate gmm_score: lower likelihood = more anomalous → higher score for open
    gmm_scores = -test_gdf["gmm_score"].fillna(test_gdf["gmm_score"].min())
    auroc_openset = roc_auc_score(is_open.astype(int), gmm_scores)
    print(f"Open-set detection AUROC (GMM): {auroc_openset:.4f}")
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "openset_auroc_gmm", auroc_openset)

    # Boolean anomaly flag as binary detector
    if test_gdf["gmm_gemos_anomaly"].nunique() > 1:
        auroc_flag = roc_auc_score(is_open.astype(int),
                                   test_gdf["gmm_gemos_anomaly"].astype(int))
        print(f"Open-set detection AUROC (flag): {auroc_flag:.4f}")
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "openset_auroc_flag", auroc_flag)
else:
    print("WARNING: test set has no open or no closed samples — skipping open-set AUROC.")

# Closed-set accuracy (OPEN samples excluded from accuracy computation)
closed_test = test_gdf[~is_open]
if len(closed_test) > 0:
    closed_acc = accuracy_score(closed_test.y_true, closed_test.y_pred)
    closed_f1w = f1_score(closed_test.y_true, closed_test.y_pred, average="weighted", zero_division=1)
    closed_f1m = f1_score(closed_test.y_true, closed_test.y_pred, average="micro",    zero_division=1)
    print(f"Closed-set test accuracy   : {closed_acc:.4f}")
    print(f"Closed-set test F1-weighted: {closed_f1w:.4f}")
    print(f"Closed-set test F1-micro   : {closed_f1m:.4f}")
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "closed_test_accuracy",    closed_acc)
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "closed_test_f1_weighted", closed_f1w)
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "closed_test_f1_micro",    closed_f1m)

# Full test report
_class_names = sorted(closed_class_map.values())
_anom_pct    = test_gdf.gmm_gemos_anomaly.mean() * 100
_clean_test  = test_gdf[~test_gdf.gmm_gemos_anomaly]

print(f"\n{_SEP}")
print("TEST METRICS — all samples (closed + open)")
print(_SEP)
print(f"N={len(test_gdf)}")
print(f"Accuracy  : {accuracy_score(test_gdf.y_true, test_gdf.y_pred):.4f}")
print(f"F1-weighted: {f1_score(test_gdf.y_true, test_gdf.y_pred, average='weighted', zero_division=1):.4f}")
print("\nClassification Report:")
print(classification_report(test_gdf.y_true, test_gdf.y_pred, zero_division=1))

print(f"\n{_SEP}")
print(f"TEST METRICS — GMM-filtered (anomaly rate {_anom_pct:.1f}%)")
print(_SEP)
print(f"N={len(_clean_test)}")
if len(_clean_test) > 0:
    print(f"Accuracy  : {accuracy_score(_clean_test.y_true, _clean_test.gmm_pred):.4f}")
    print(f"F1-weighted: {f1_score(_clean_test.y_true, _clean_test.gmm_pred, average='weighted', zero_division=1):.4f}")

save_pytorch_model(best_model_p3, mlflow_logger)
print("\nDone.")
