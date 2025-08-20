import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np

from lightly.loss import NegativeCosineSimilarity
from lightly.utils.debug import std_of_l2_normalized
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sits_siam.backbone import TransformerBackbone
from sits_siam.utils import SitsFinetuneDatasetFromNpz
from sits_siam.bottleneck import PoolingBottleneck
from sits_siam.augment import (
    RandomAddNoise,
    RandomTempSwapping,
    RandomTempShift,
    RandomTempRemoval,
    RandomCloudAugmentation,
    RandomChanSwapping,
    RandomChanRemoval,
    AddMissingMask,
    Normalize,
    Pipeline,
    IncreaseSequenceLength,
    ToPytorchTensor,
)

# disable scientific notation pytorch, keep 3 numbers after decimal
torch.set_printoptions(precision=3, sci_mode=False)

class FastSiamMultiViewTransform(object):
    def __init__(
        self,
        n_views: int = 4,
    ):
        self.n_views = n_views
        self.transform = Pipeline(
            [
                IncreaseSequenceLength(140),
                RandomAddNoise(0.02),
                RandomTempSwapping(max_distance=3),
                RandomTempShift(),
                RandomTempRemoval(),
                #RandomChanRemoval(0.1),
                #RandomChanSwapping(0.1),
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

val_dataset = SitsFinetuneDatasetFromNpz("data/texas0101998_npz/train.npz", transform=FastSiamMultiViewTransform(),)

train_dataset = SitsFinetuneDatasetFromNpz("data/texas0101998_npz/test.npz", transform=FastSiamMultiViewTransform(),)

class TransformerClassifier(pl.LightningModule):
    def __init__(self, max_seq_len=140):
        super(TransformerClassifier, self).__init__()
        self.backbone = TransformerBackbone(max_seq_len=max_seq_len)
        self.bottleneck = PoolingBottleneck()
        self.projection_head = SimSiamProjectionHead(128, 512, 1024)
        self.prediction_head = SimSiamPredictionHead(1024, 512, 1024)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, batch):
        x = batch["x"]
        doy = batch["doy"]
        mask = batch["mask"]

        f = self.backbone(x, doy, mask)
        f = self.bottleneck(f)
        z = self.projection_head(f)
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
        optim = torch.optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
        )
        return optim


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)

trainer = pl.Trainer(max_epochs=40, callbacks=[checkpoint_callback], accelerator='gpu', precision="16-mixed", devices=[0, 1, 2])
model = TransformerClassifier()


trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model = TransformerClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

# Saving pytorch model backbone state dict
torch.save(model.backbone.state_dict(), "weights/fastsiam_texas.pth")
