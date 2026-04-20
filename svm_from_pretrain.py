import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearnex import patch_sklearn
from tqdm.std import tqdm
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from torch.utils.data import WeightedRandomSampler
import optuna

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
    setup_seed,
    save_other_model,
    split_with_percent_and_class_coverage,
    check_if_already_ran,
    run_gemos,
    load_pretrain_weights,
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
setup_seed()


# Parse arguments
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
    "--model_name",
    type=str,
    choices=["MAMBA", "BERT", "BERTPP", "LSTM", "CNN"],
    default="MAMBA",
)
BATCHED_ARGS_PARSER.add_argument(
    "--pretrain",
    type=str,
    choices=["reconstruct", "MoCo"],
    default="reconstruct",
)
BATCHED_ARGS_PARSER.add_argument(
    "--gpu",
    type=int,
    default=0,
)
BATCHED_ARGS_PARSER.add_argument(
    "--n_trials",
    type=int,
    default=100,
    help="Number of Optuna trials for hyperparameter optimization",
)
BATCHED_ARGS_PARSER.add_argument(
    "--n_jobs",
    type=int,
    default=20,
    help="Number of parallel jobs for Optuna SVM optimization",
)
_parsed_args, _ = BATCHED_ARGS_PARSER.parse_known_args()
TRAIN_PERCENT = float(_parsed_args.train_percent)
DATASET = _parsed_args.dataset
MODEL_NAME = _parsed_args.model_name
PRETRAIN = _parsed_args.pretrain
GPU_ID = _parsed_args.gpu
N_TRIALS = _parsed_args.n_trials
N_JOBS = _parsed_args.n_jobs
BATCH_SIZE = 2 * 512

TAGS = {
    "dataset": str(DATASET),
    "train_percent": str(TRAIN_PERCENT),
    "model_name": str(MODEL_NAME),
    "pretrain": str(PRETRAIN),
    "n_trials": str(N_TRIALS),
    "n_jobs": str(N_JOBS),
}
EXPERIMENT_NAME = f"{DATASET}-finetuning"
RUN_NAME_SUFFIX = f"SVM{MODEL_NAME}-{TRAIN_PERCENT}-{PRETRAIN}"
mlflow.set_experiment(EXPERIMENT_NAME)

# if check_if_already_ran(EXPERIMENT_NAME, RUN_NAME_SUFFIX):
#     print(f"{RUN_NAME_SUFFIX} already ran in {EXPERIMENT_NAME}. Exiting.")
#     exit(0)

# Setup transforms
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

# Load datasets
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
        gdf_train_val = gdf[gdf["CD_MUN"].isin(mun_train)].copy()
        gdf_val = gdf[gdf["CD_MUN"].isin(mun_val)].copy()
        gdf_test = gdf[gdf["CD_MUN"].isin(mun_test)].copy()
    else:
        gdf_train, gdf_val, gdf_test = split_with_percent_and_class_coverage(
            gdf, percent=TRAIN_PERCENT, max_attempts=500
        )
        gdf_train_val = gdf_train.copy()

    print(f"Municípios Treino: {len(gdf_train)} linhas")
    print(f"Municípios Validação: {len(gdf_val)} linhas")
    print(f"Municípios Teste: {len(gdf_test)} linhas")

    train_dataset = AgriGEELiteDataset(
        gdf_train,
        "data/agl/df_sits.parquet",
        transform=aug_transforms,
        timestamp_processing="days_after_start",
    )

    train_val_dataset = AgriGEELiteDataset(
        gdf_train_val,
        "data/agl/df_sits.parquet",
        transform=transforms,
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
    train_val_dataset = SitsFinetuneDatasetFromNpz(
        f"data/{DATASET}_{split_string}/train.npz",
        transform=transforms,
    )
    val_dataset = SitsFinetuneDatasetFromNpz(
        f"data/{DATASET}_{split_string}/val.npz",
        transform=transforms,
    )
    test_dataset = SitsFinetuneDatasetFromNpz(
        f"data/{DATASET}_{split_string}/test.npz",
        transform=transforms,
    )

    print(f"Treino: {len(train_dataset)} linhas")
    print(f"Validação: {len(val_dataset)} linhas")
    print(f"Teste: {len(test_dataset)} linhas")

    class_names = train_dataset.get_class_names()
    class_map = {i: name for i, name in enumerate(class_names)}

    gdf_train = pd.DataFrame()
    gdf_train_val = pd.DataFrame()
    gdf_val = pd.DataFrame()
    gdf_test = pd.DataFrame()
else:
    raise ValueError(f"Dataset {DATASET} not recognized.")

# Create DataLoaders
print("Creating DataLoaders...")
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
train_val_dataloader = torch.utils.data.DataLoader(
    train_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# Initialize model and load pretrained weights
num_classes = len(class_map)
BACKBONES = {
    "BERT": SITSBert,
    "BERTPP": SITSBertPlusPlus,
    "LSTM": SITS_LSTM,
    "CNN": SITSConvNext,
    "MAMBA": SITSMamba,
}

backbone = BACKBONES[MODEL_NAME](num_classes=num_classes)

if PRETRAIN != "off":
    backbone = load_pretrain_weights(DATASET, PRETRAIN, MODEL_NAME, backbone)

# Move model to GPU
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
backbone.to(device)
backbone.eval()

print(f"Using device: {device}")
print("----------------------------------------------")


# Extract embeddings function
def extract_embeddings(dataloader, model, device):
    """Extract embeddings from pretrained model using DataLoader"""
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            x = batch["x"].to(device)
            doy = batch["doy"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].squeeze()
            
            # Get embeddings from the model
            # Depending on model architecture, we get pooled embeddings
            pooled, logits, _ = model(x, doy, mask)
            
            embeddings_list.append(pooled.cpu().numpy())
            labels_list.append(y.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    return embeddings, labels


# Extract features from all datasets using DataLoaders
print("Extracting embeddings from train dataset...")
train_embeddings, train_labels = extract_embeddings(train_dataloader, backbone, device)

print("Extracting embeddings from train_val dataset...")
train_val_embeddings, train_val_labels = extract_embeddings(train_val_dataloader, backbone, device)

print("Extracting embeddings from val dataset...")
val_embeddings, val_labels = extract_embeddings(val_dataloader, backbone, device)

print("Extracting embeddings from test dataset...")
test_embeddings, test_labels = extract_embeddings(test_dataloader, backbone, device)

print(f"Embedding shapes: train={train_embeddings.shape}, val={val_embeddings.shape}, test={test_embeddings.shape}")
print("----------------------------------------------")

# Create dataframes with embeddings
emb_size = train_embeddings.shape[1]
X_COLUMNS = [f"emb_{i}" for i in range(emb_size)]

train_features = pd.DataFrame(train_embeddings, columns=X_COLUMNS)
train_val_features = pd.DataFrame(train_val_embeddings, columns=X_COLUMNS)
val_features = pd.DataFrame(val_embeddings, columns=X_COLUMNS)
test_features = pd.DataFrame(test_embeddings, columns=X_COLUMNS)

# Create geodataframes
if DATASET == "brazil":
    gdf_train = pd.concat([gdf_train.reset_index(drop=True), train_features], axis=1)
    gdf_train_val = pd.concat([gdf_train_val.reset_index(drop=True), train_val_features], axis=1)
    gdf_val = pd.concat([gdf_val.reset_index(drop=True), val_features], axis=1)
    gdf_test = pd.concat([gdf_test.reset_index(drop=True), test_features], axis=1)
elif DATASET in {"texas", "california"}:
    gdf_train = train_features.copy()
    gdf_train["crop_class"] = [class_map[label] for label in train_labels]

    gdf_train_val = train_val_features.copy()
    gdf_train_val["crop_class"] = [class_map[label] for label in train_val_labels]

    gdf_val = val_features.copy()
    gdf_val["crop_class"] = [class_map[label] for label in val_labels]

    gdf_test = test_features.copy()
    gdf_test["crop_class"] = [class_map[label] for label in test_labels]


# Train SVM with hyperparameter optimization
def train_svc_from_pretrain(gdf_train, gdf_train_val, gdf_val, gdf_test, run_name, n_trials=30):
    """Train SVM from pretrained embeddings with hyperparameter search"""
    
    # Limit samples for hyperparameter search (max 5000)
    max_samples_for_search = 5000
    if len(gdf_train) > max_samples_for_search:
        sampled_gdf_train = (
            gdf_train.sample(max_samples_for_search, random_state=42)
            .reset_index(drop=True)
            .copy()
        )
        print(f"Using {max_samples_for_search} samples for hyperparameter search (out of {len(gdf_train)})")
    else:
        sampled_gdf_train = gdf_train.copy()
        print(f"Using all {len(gdf_train)} samples for hyperparameter search")

    gdf_train = gdf_train.copy()
    gdf_train_val = gdf_train_val.copy()
    gdf_val = gdf_val.copy()
    gdf_test = gdf_test.copy()

    sampled_X_train = sampled_gdf_train[X_COLUMNS].to_numpy().astype(np.float32)
    sampled_y_train = sampled_gdf_train["crop_class"].values

    X_train = gdf_train[X_COLUMNS].to_numpy().astype(np.float32)
    y_train = gdf_train["crop_class"].values

    X_train_val = gdf_train_val[X_COLUMNS].to_numpy().astype(np.float32)

    X_val = gdf_val[X_COLUMNS].to_numpy().astype(np.float32)
    y_val = gdf_val["crop_class"].values

    X_test = gdf_test[X_COLUMNS].to_numpy().astype(np.float32)

    def objective(trial):
        C = trial.suggest_float("C", 0.01, 100, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])

        params = {
            "C": C,
            "kernel": kernel,
            "class_weight": "balanced",
            "random_state": 42,
            "cache_size": 2000,
        }

        if kernel == "rbf":
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            params["gamma"] = gamma
        elif kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            params["degree"] = degree
            params["gamma"] = gamma

        svc = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(random_state=42)),
                ("svc", SVC(**params)),
            ]
        )
        svc.fit(sampled_X_train, sampled_y_train)

        y_pred_val = svc.predict(X_val)
        f1 = f1_score(y_val, y_pred_val, average="weighted")

        return f1

    study = optuna.create_study(
        direction="maximize",
        study_name=f"svc_pretrain_optimization_{EXPERIMENT_NAME}_{run_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS, show_progress_bar=True)

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best validation F1: {study.best_value:.4f}")

    # Train final model with best parameters on full training set
    best_params = study.best_params.copy()
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = 42
    best_params["cache_size"] = 2000
    best_params["probability"] = True

    print("Training final SVM model on full training set...")
    svc_best = SVC(**best_params)
    svc_best.fit(X_train, y_train)

    # Make predictions
    gdf_train["y_true"] = gdf_train["crop_class"]
    gdf_train["y_pred"] = svc_best.predict(X_train)
    gdf_train["y_proba"] = np.max(svc_best.predict_proba(X_train), axis=1)

    gdf_train_val["y_true"] = gdf_train_val["crop_class"]
    gdf_train_val["y_pred"] = svc_best.predict(X_train_val)
    gdf_train_val["y_proba"] = np.max(svc_best.predict_proba(X_train_val), axis=1)

    gdf_val["y_true"] = gdf_val["crop_class"]
    gdf_val["y_pred"] = svc_best.predict(X_val)
    gdf_val["y_proba"] = np.max(svc_best.predict_proba(X_val), axis=1)

    gdf_test["y_true"] = gdf_test["crop_class"]
    gdf_test["y_pred"] = svc_best.predict(X_test)
    gdf_test["y_proba"] = np.max(svc_best.predict_proba(X_test), axis=1)

    mlflow_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tags=TAGS,
        run_name=run_name,
        tracking_uri=mlflow.get_tracking_uri(),
    )

    # Log hyperparameters
    for param_name, param_value in study.best_params.items():
        mlflow_logger.experiment.log_param(
            mlflow_logger.run_id, param_name, param_value
        )
    
    # Log additional tags
    for tag_name, tag_value in TAGS.items():
        mlflow_logger.experiment.log_param(
            mlflow_logger.run_id, tag_name, tag_value
        )

    # Run GEMOS evaluation
    run_gemos(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger, True, 100)
    
    # Save model
    save_other_model(svc_best, mlflow_logger)

    print(f"Model saved and logged to MLflow as {run_name}")


# Execute training
print("Starting SVM training with pretrained embeddings...")
train_svc_from_pretrain(
    gdf_train,
    gdf_train_val,
    gdf_val,
    gdf_test,
    RUN_NAME_SUFFIX,
    n_trials=N_TRIALS,
)

print("Done!")
