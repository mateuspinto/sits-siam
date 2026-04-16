# AGENTS.md — AI Agent Guide for SITS-SIAM

## Project Identity

**SITS-SIAM** is a satellite image time series (SITS) crop classification framework.
Domain: remote sensing, precision agriculture, self-supervised learning.

Core loop: **Pretrain backbone on unlabeled SITS → Fine-tune classifier on labeled parcels → Evaluate with uncertainty**.

---

## Codebase Map

```
sits-siam/
├── sits_siam/                  # Main importable package
│   ├── models.py               # ALL model architectures (SITSBert, LSTM, ConvNext, Mamba, MLP)
│   ├── augment.py              # ALL transforms/augmentations (temporal, spectral, cloud)
│   ├── auxiliar.py             # Training utilities: callbacks, metrics, split logic, GMM
│   └── utils/dataset.py        # Dataset classes for all supported data formats
├── pretrain_siam.py            # FastSiam/SimSiam pretraining (4-view)
├── pretrain_moco.py            # MoCo pretraining (momentum contrast)
├── pretrain_moco_remake.py     # Alternative MoCo variant
├── pretrain_pmsn.py            # PMSN pretraining
├── pretrain_reconstruct.py     # Reconstruction-based pretraining (MAE-style)
├── finetuning.py               # Supervised fine-tuning + GMM uncertainty (main eval script)
├── in_season.py                # In-season monitoring (vary sequence length by days)
├── shallows.py                 # Baseline: SVM/RF/LGBM on monthly means
├── svm_from_pretrain.py        # SVM on frozen pretrained features
├── mlp.py                      # MLP end-to-end baseline
├── visualize.py                # t-SNE, reconstruction plots
└── run_all.py                  # Orchestration: loops datasets × models × pretrains
```

---

## Architecture Choices

### Models (`sits_siam/models.py`)

All models accept input shape `(batch, seq_len, 10)` — 10 Sentinel-2 spectral bands, variable sequence length.

| Class | Type | Key params |
|---|---|---|
| `SITSBert` | Transformer | `hidden=256, n_layers=3, n_heads=8` |
| `SITSBertPlusPlus` | Transformer + extra attn | `hidden=252, n_layers=3, n_heads=12` |
| `SITS_LSTM` | Bidirectional GRU | Uses packed sequences + masking |
| `SITSConvNext` | 1D ConvNext | Depth-wise separable + drop-path |
| `SITSMamba` | State-space (Mamba) | Sequential MambaBlocks |
| `SITS_MLP_Backbone` | Simple MLP | Flattens (seq × features) → 3 dense layers |

**Canonical model string** used in CLI args and MLflow: `"BERT"`, `"LSTM"`, `"ConvNext"`, `"Mamba"`, `"MLP"`.

### Dataset Classes (`sits_siam/utils/dataset.py`)

| Class | Input | Use case |
|---|---|---|
| `AgriGEELiteDataset` | GeoDataFrame (parcels) + DataFrame (SITS) | Brazil dataset |
| `SitsFinetuneDatasetFromNpz` | `.npz` files | Texas/California fine-tuning |
| `SitsPretrainDatasetFromNpz` | `.npz` files | Texas/California pretraining (distributed) |
| `SitsDatasetFromDataframe` | Pandas DataFrame | Generic |

### Augmentations (`sits_siam/augment.py`)

Applied as transform pipelines in `torchvision.transforms.Compose` style.

**Temporal:** `RandomTempShift(max_shift=30)`, `RandomTempSwapping(max_distance=3)`, `RandomTempRemoval(probability=0.5)`
**Spectral:** `RandomChanSwapping`, `RandomAddNoise(max_noise=0.05)`
**Cloud masking:** `RandomCloudAugmentation(probability=0.15)`
**Masking/reconstruction:** `AddCorruptedSample`, `RandomMasking(mask_rate=0.15)`
**Normalization:** `Normalize` — hardcoded per-band mean/std for Sentinel-2

### Pretraining Strategies

| Script | Method | Loss | Views |
|---|---|---|---|
| `pretrain_siam.py` | FastSiam/SimSiam | NegativeCosineSimilarity | 4 |
| `pretrain_moco.py` | MoCo | NTXentLoss | 2 |
| `pretrain_pmsn.py` | PMSN | Custom | 2 |
| `pretrain_reconstruct.py` | Reconstruction | MSE | 1 + corrupted |

### Training Defaults (applies to fine-tuning and pretraining)

```python
BATCH_SIZE = 1024       # (2 × 512 for SimSiam multi-view)
MAX_EPOCHS = 100        # 200 if train_percent <= 1
WARMUP_EPOCHS = 10
BASE_LR = 1e-4
OPTIMIZER = AdamW
LR_SCHEDULE = CosineAnnealing (after warmup)
PRECISION = "bf16-mixed"
SEED = 42
```

### Experiment Tracking

**MLflow** is the tracking backend. Experiments named `{dataset}-finetuning`, `{dataset}-pretrain`, `brazil-in-season`.

Pretrained weights saved as MLflow artifact `weights.pth`. Fine-tuning loads them via `load_pretrain_weights()` in `auxiliar.py` — this function strips classifier head keys.

---

## Data Conventions

**Spectral bands order** (always 10 features):
`[Blue, Green, Red, RedEdge1, RedEdge2, RedEdge3, RedEdge4, NIR, SWIR1, SWIR2]`

**Normalization stats** are hardcoded in `augment.py > Normalize`.

**Sequence length:** Brazil uses up to 140 timesteps; Texas/California use 45.

**DOY encoding:** `PositionalEncoding` in `models.py` handles up to day 366. Can also use days-after-start.

**Missing data:** Zero-padding. `AddMissingMask` creates boolean mask for padded positions.

**Datasets by region:**
- Brazil: Agricultural parcels via IBGE + Google Earth Engine exports. Split by municipality (`CD_MUN`) to prevent spatial leakage.
- Texas/California: USDA NASS CDL labels. Pre-packaged `.npz`. Train splits: 70%, 10%, 1%, 0.1%.

---

## Evaluation Pipeline

1. **During pretraining:** `KNNCallback` evaluates backbone quality every 2 epochs (KNN F1-weighted on labeled subset).
2. **Fine-tuning:** Standard val accuracy + weighted F1. Early stopping on val accuracy.
3. **Post fine-tuning:** `run_gemos()` fits per-class GMM on correctly-predicted training embeddings. Used for anomaly/uncertainty detection.
4. **Uncertainty metrics:** AUROC, AUPR-Error, AUPR-Success, FPR@95TPR (via `gemos` library).
5. **In-season:** `in_season.py` crops sequences at N days intervals to measure early classification ability.

---

## Common Patterns to Follow

### Adding a new model

1. Implement in `sits_siam/models.py` following same interface: `forward(x, mask=None, doy=None)` → returns `(pooled_embedding, logits, reconstruction_or_None)`.
2. Add to model selection logic in all training scripts (grep for `if model_name == "BERT"`).
3. Register canonical string name.

### Adding a new dataset

1. Implement Dataset class in `sits_siam/utils/dataset.py` inheriting `torch.utils.data.Dataset`.
2. Return dict with keys: `features`, `label`, optionally `mask`, `doy`.
3. Add dataset loading logic in training scripts (grep for `if dataset == "brazil"`).

### Adding a new pretraining method

1. Create `pretrain_{name}.py` following structure of `pretrain_moco.py`.
2. Use `lightly` library for SSL heads/losses where possible.
3. Save weights to MLflow as `weights.pth`.
4. Expose `--pretrain {name}` flag in `finetuning.py`.

---

## Key Functions to Know

| Function | File | Purpose |
|---|---|---|
| `load_pretrain_weights(model, run_id)` | `auxiliar.py` | Load backbone from MLflow, skip head |
| `predict_and_save_predictions(...)` | `auxiliar.py` | Generate predictions + embeddings + confidence |
| `split_with_percent_and_class_coverage(...)` | `auxiliar.py` | Stratified split ensuring all classes present |
| `run_gemos(...)` | `auxiliar.py` | Fit GMM uncertainty model on embeddings |
| `get_class_weights(...)` | `auxiliar.py` | Compute inverse-frequency weights for CrossEntropy |

---

## CLI Conventions

All scripts use `argparse`. Common args across scripts:

```
--model_name    {BERT, LSTM, ConvNext, Mamba, MLP}
--dataset       {brazil, texas, california}
--gpu           GPU index (int)
--pretrain      {off, Siam, MoCo, PMSN, Reconstruct}  # finetuning.py only
--train_percent {0.1, 1, 10, 70}                       # finetuning.py only
--num_days      int                                     # in_season.py only
```

---

## What NOT to Assume

- **No single config file.** Hyperparameters are inline in each script's `main()` or argument defaults.
- **MLflow run IDs are not committed.** Pretrained weights must be referenced by MLflow run ID at runtime.
- **No automated tests.** Correctness validated via MLflow metrics and visual inspection.
- **Brazil dataset is not NPZ.** It uses GeoDataFrame + SITS DataFrame (AgriGEELiteDataset), unlike Texas/California.
- **Class lists differ per dataset.** Always check the dataset-specific label mappings in the training scripts.

---

## Priorities When Modifying Code

1. **Correctness of data splits** — spatial leakage (municipality-level splits for Brazil) is the main validity concern.
2. **Reproducibility** — seed=42 set in all scripts; keep it.
3. **MLflow logging** — all metrics and artifacts must be logged; don't skip.
4. **Memory efficiency** — float16 everywhere, pre-allocated numpy arrays in pretraining datasets.
5. **GPU compatibility** — bf16-mixed precision; check Mamba requires specific CUDA versions.
