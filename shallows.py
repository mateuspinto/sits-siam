import geopandas as gpd
from lightgbm import LGBMClassifier, early_stopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearnex import patch_sklearn
from tqdm.std import tqdm
from pytorch_lightning.loggers import MLFlowLogger
import optuna

from sits_siam.augment import (
    AddMissingMask,
    IncreaseSequenceLength,
    LimitSequenceLength,
    Pipeline,
    RandomAddNoise,
    RandomTempRemoval,
    RandomTempShift,
    RandomTempSwapping,
    ReduceToMonthlyMeans,
)
from sits_siam.auxiliar import beautify_prints, setup_seed, save_other_model
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz

DATASET = "brazil"
TRAIN_SIZE = 70


TAGS = {
    "dataset": DATASET,
}
EXPERIMENT_NAME = f"{DATASET}-finetuning"

patch_sklearn()
setup_seed()
beautify_prints()


def log_results_in_mlflow(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger):
    for name, gdf in zip(
        ["train_aug", "train", "val", "test"],
        [gdf_train, gdf_train_val, gdf_val, gdf_test],
    ):
        acc = accuracy_score(gdf["y_true"], gdf["y_pred"])
        f1_weighted = f1_score(gdf["y_true"], gdf["y_pred"], average="weighted")
        f1_micro = f1_score(gdf["y_true"], gdf["y_pred"], average="micro")

        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_accuracy", acc
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_f1_weighted", f1_weighted
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_f1_micro", f1_micro
        )


transforms = Pipeline(
    [
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        ReduceToMonthlyMeans(),
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
    ]
)

gdf = gpd.read_parquet("/home/m/Downloads/gdf.parquet")

class_map = (
    gdf[["crop_class", "crop_number"]]
    .drop_duplicates()
    .set_index("crop_number")["crop_class"]
    .to_dict()
)

unique_mun = gdf["CD_MUN"].unique()
mun_train, mun_temp = train_test_split(unique_mun, test_size=0.30, random_state=13)
mun_val, mun_test = train_test_split(mun_temp, test_size=0.50, random_state=13)


gdf_train = gdf[gdf["CD_MUN"].isin(mun_train)].copy()
gdf_train_val = gdf[gdf["CD_MUN"].isin(mun_train)].copy()
gdf_val = gdf[gdf["CD_MUN"].isin(mun_val)].copy()
gdf_test = gdf[gdf["CD_MUN"].isin(mun_test)].copy()

print(f"Municípios Treino: {len(mun_train)} - {len(gdf_train)} linhas")
print(f"Municípios Validação: {len(mun_val)} - {len(gdf_val)} linhas")
print(f"Municípios Teste: {len(mun_test)} - {len(gdf_test)} linhas")

train_dataset = AgriGEELiteDataset(
    gdf_train,
    "/home/m/Downloads/df_sits.parquet",
    transform=aug_transforms,
    timestamp_processing="days_after_start",
)

train_val_dataset = AgriGEELiteDataset(
    gdf_train,
    "/home/m/Downloads/df_sits.parquet",
    transform=transforms,
    timestamp_processing="days_after_start",
)

val_dataset = AgriGEELiteDataset(
    gdf_val,
    "/home/m/Downloads/df_sits.parquet",
    transform=transforms,
    timestamp_processing="days_after_start",
)

test_dataset = AgriGEELiteDataset(
    gdf_test,
    "/home/m/Downloads/df_sits.parquet",
    transform=transforms,
    timestamp_processing="days_after_start",
)

emb_sizes = train_dataset[0]["x"].flatten().shape[0]
X_COLUMNS = [f"x_{i}" for i in range(emb_sizes)]

train_features = np.zeros((len(train_dataset), emb_sizes), dtype=np.float32)
train_val_features = np.zeros((len(train_val_dataset), emb_sizes), dtype=np.float32)
val_features = np.zeros((len(val_dataset), emb_sizes), dtype=np.float32)
test_features = np.zeros((len(test_dataset), emb_sizes), dtype=np.float32)

print(
    train_features.shape,
    train_val_features.shape,
    val_features.shape,
    test_features.shape,
)

print("----------------------------------------------")

for n, sample in enumerate(tqdm(train_dataset, desc="Extracting train features")):
    train_features[n] = sample["x"].flatten()

for n, sample in enumerate(
    tqdm(train_val_dataset, desc="Extracting train_val features")
):
    train_val_features[n] = sample["x"].flatten()

for n, sample in enumerate(tqdm(val_dataset, desc="Extracting val features")):
    val_features[n] = sample["x"].flatten()

for n, sample in enumerate(tqdm(test_dataset, desc="Extracting test features")):
    test_features[n] = sample["x"].flatten()

train_features = pd.DataFrame(train_features, columns=X_COLUMNS)
train_val_features = pd.DataFrame(train_val_features, columns=X_COLUMNS)
val_features = pd.DataFrame(val_features, columns=X_COLUMNS)
test_features = pd.DataFrame(test_features, columns=X_COLUMNS)

gdf_train = pd.concat([gdf_train.reset_index(drop=True), train_features], axis=1)

gdf_train_val = pd.concat(
    [gdf_train_val.reset_index(drop=True), train_val_features], axis=1
)
gdf_val = pd.concat([gdf_val.reset_index(drop=True), val_features], axis=1)
gdf_test = pd.concat([gdf_test.reset_index(drop=True), test_features], axis=1)


def train_lgbm(gdf_train, gdf_train_val, gdf_val, gdf_test, run_name):
    gdf_train = gdf_train.copy()
    gdf_train_val = gdf_train_val.copy()
    gdf_val = gdf_val.copy()
    gdf_test = gdf_test.copy()

    lgbm = LGBMClassifier(
        random_state=42,
        class_weight="balanced",
        objective="multiclassova",
        n_jobs=-1,
        verbose=-1,
        num_iterations=1000,
        num_estimators=500,
        max_depth=5,
    )

    lgbm.fit(
        gdf_train[X_COLUMNS],
        gdf_train["crop_class"],
        eval_set=[
            (
                gdf_val[X_COLUMNS],
                gdf_val["crop_class"],
            )
        ],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=50),
        ],
    )

    gdf_train["y_true"] = gdf_train["crop_class"]
    gdf_train["y_pred"] = lgbm.predict(gdf_train[X_COLUMNS])
    gdf_train["y_proba"] = np.max(lgbm.predict_proba(gdf_train[X_COLUMNS]), axis=1)

    gdf_train_val["y_true"] = gdf_train_val["crop_class"]
    gdf_train_val["y_pred"] = lgbm.predict(gdf_train_val[X_COLUMNS])
    gdf_train_val["y_proba"] = np.max(
        lgbm.predict_proba(gdf_train_val[X_COLUMNS]),
        axis=1,
    )

    gdf_val["y_true"] = gdf_val["crop_class"]
    gdf_val["y_pred"] = lgbm.predict(gdf_val[X_COLUMNS])
    gdf_val["y_proba"] = np.max(lgbm.predict_proba(gdf_val[X_COLUMNS]), axis=1)

    gdf_test["y_true"] = gdf_test["crop_class"]
    gdf_test["y_pred"] = lgbm.predict(gdf_test[X_COLUMNS])
    gdf_test["y_proba"] = np.max(lgbm.predict_proba(gdf_test[X_COLUMNS]), axis=1)

    mlflow_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=run_name)

    log_results_in_mlflow(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger)
    save_other_model(lgbm, mlflow_logger)


def train_svc(gdf_train, gdf_train_val, gdf_val, gdf_test, run_name, n_trials=60):
    sampled_gdf_train = gdf_train.sample(10_000, random_state=42).copy()

    gdf_train = gdf_train
    gdf_train_val = gdf_train_val
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
        study_name=f"svc_optimization_{EXPERIMENT_NAME}_{run_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = 42
    best_params["cache_size"] = 2000
    best_params["probability"] = False

    svc_best = SVC(**best_params)
    svc_best.fit(X_train, y_train)

    gdf_train["y_true"] = gdf_train["crop_class"]
    gdf_train["y_pred"] = svc_best.predict(X_train)

    gdf_train_val["y_true"] = gdf_train_val["crop_class"]
    gdf_train_val["y_pred"] = svc_best.predict(X_train_val)

    gdf_val["y_true"] = gdf_val["crop_class"]
    gdf_val["y_pred"] = svc_best.predict(X_val)

    gdf_test["y_true"] = gdf_test["crop_class"]
    gdf_test["y_pred"] = svc_best.predict(X_test)

    mlflow_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=run_name)

    for param_name, param_value in study.best_params.items():
        mlflow_logger.experiment.log_param(
            mlflow_logger.run_id, param_name, param_value
        )

    log_results_in_mlflow(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger)
    save_other_model(svc_best, mlflow_logger)


def train_rf(gdf_train, gdf_train_val, gdf_val, gdf_test, run_name, n_trials=30):
    X_train = gdf_train[X_COLUMNS].to_numpy().astype(np.float32)
    y_train = gdf_train["crop_class"].values

    X_train_val = gdf_train_val[X_COLUMNS].to_numpy().astype(np.float32)

    X_val = gdf_val[X_COLUMNS].to_numpy().astype(np.float32)
    y_val = gdf_val["crop_class"].values

    X_test = gdf_test[X_COLUMNS].to_numpy().astype(np.float32)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 10, 15, 20, 30],
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        rf = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)
        y_pred_val = rf.predict(X_val)

        return f1_score(y_val, y_pred_val, average="weighted")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"rf_optimization_{EXPERIMENT_NAME}_{run_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    rf_best = RandomForestClassifier(**best_params)
    rf_best.fit(X_train, y_train)

    gdf_train = gdf_train.copy()
    gdf_train_val = gdf_train_val.copy()
    gdf_val = gdf_val.copy()
    gdf_test = gdf_test.copy()

    gdf_train["y_true"] = gdf_train["crop_class"]
    gdf_train["y_pred"] = rf_best.predict(X_train)

    gdf_train_val["y_true"] = gdf_train_val["crop_class"]
    gdf_train_val["y_pred"] = rf_best.predict(X_train_val)

    gdf_val["y_true"] = gdf_val["crop_class"]
    gdf_val["y_pred"] = rf_best.predict(X_val)

    gdf_test["y_true"] = gdf_test["crop_class"]
    gdf_test["y_pred"] = rf_best.predict(X_test)

    mlflow_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=run_name)

    for param_name, param_value in study.best_params.items():
        mlflow_logger.experiment.log_param(
            mlflow_logger.run_id, param_name, param_value
        )

    log_results_in_mlflow(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger)
    save_other_model(rf_best, mlflow_logger)


train_rf(gdf_train, gdf_train_val, gdf_val, gdf_test, f"RF-{TRAIN_SIZE}", n_trials=20)
train_lgbm(gdf_train, gdf_train_val, gdf_val, gdf_test, f"LGBM-{TRAIN_SIZE}")
train_svc(
    gdf_train,
    gdf_train_val,
    gdf_val,
    gdf_test,
    f"SVM-{TRAIN_SIZE}",
    n_trials=20,
)
