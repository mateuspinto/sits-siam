import argparse
import geopandas as gpd
from lightgbm import LGBMClassifier, early_stopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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
from sits_siam.auxiliar import (
    beautify_prints,
    setup_seed,
    save_other_model,
    split_with_percent_and_class_coverage,
    log_results_in_mlflow,
    check_if_already_ran,
    run_gemos,
)
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz

patch_sklearn()
setup_seed()
beautify_prints()

BATCHED_ARGS_PARSER = argparse.ArgumentParser(add_help=False)
BATCHED_ARGS_PARSER.add_argument(
    "--train_percent",
    type=float,
    default=70.0,
)
BATCHED_ARGS_PARSER.add_argument(
    "--dataset",
    type=str,
    choices=["brazil", "california", "texas", "pastis"],
    default="brazil",
)
BATCHED_ARGS_PARSER.add_argument(
    "--n_trials",
    type=int,
    default=30,
    help="Number of Optuna trials for hyperparameter optimization",
)
BATCHED_ARGS_PARSER.add_argument(
    "--n_jobs",
    type=int,
    default=4,
    help="Number of parallel jobs for Optuna SVM optimization",
)
_parsed_args, _ = BATCHED_ARGS_PARSER.parse_known_args()
TRAIN_PERCENT = float(_parsed_args.train_percent)
DATASET = _parsed_args.dataset
N_TRIALS = _parsed_args.n_trials
N_JOBS = _parsed_args.n_jobs

TAGS = {
    "dataset": DATASET,
    "train_percent": TRAIN_PERCENT,
    "n_trials": N_TRIALS,
    "n_jobs": N_JOBS,
}
EXPERIMENT_NAME = f"{DATASET}-finetuning"
RUN_NAME_SUFFIX = f"{TRAIN_PERCENT}"

if (
    check_if_already_ran(
        EXPERIMENT_NAME,
        f"RF-{RUN_NAME_SUFFIX}",
    )
    and check_if_already_ran(
        EXPERIMENT_NAME,
        f"LGBM-{RUN_NAME_SUFFIX}",
    )
    and check_if_already_ran(
        EXPERIMENT_NAME,
        f"SVM-{RUN_NAME_SUFFIX}",
    )
):
    print("All models already ran. Exiting.")
    exit(0)

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
        # New path for small percent modes: 10, 1, 0.1
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

    # For texas/california, get class names from dataset
    class_names = train_dataset.get_class_names()
    class_map = {i: name for i, name in enumerate(class_names)}

    gdf_train = pd.DataFrame()
    gdf_train_val = pd.DataFrame()
    gdf_val = pd.DataFrame()
    gdf_test = pd.DataFrame()
else:
    raise ValueError(f"Dataset {DATASET} not recognized.")

emb_sizes = train_dataset[0]["x"].flatten().shape[0]
X_COLUMNS = [f"emb_{i}" for i in range(emb_sizes)]

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

# Extract features and labels
train_labels = []
for n, sample in enumerate(tqdm(train_dataset, desc="Extracting train features")):
    train_features[n] = sample["x"].flatten()
    train_labels.append(sample["y"].item())

train_val_labels = []
for n, sample in enumerate(
    tqdm(train_val_dataset, desc="Extracting train_val features")
):
    train_val_features[n] = sample["x"].flatten()
    train_val_labels.append(sample["y"].item())

val_labels = []
for n, sample in enumerate(tqdm(val_dataset, desc="Extracting val features")):
    val_features[n] = sample["x"].flatten()
    val_labels.append(sample["y"].item())

test_labels = []
for n, sample in enumerate(tqdm(test_dataset, desc="Extracting test features")):
    test_features[n] = sample["x"].flatten()
    test_labels.append(sample["y"].item())

train_features = pd.DataFrame(train_features, columns=X_COLUMNS)
train_val_features = pd.DataFrame(train_val_features, columns=X_COLUMNS)
val_features = pd.DataFrame(val_features, columns=X_COLUMNS)
test_features = pd.DataFrame(test_features, columns=X_COLUMNS)

# For brazil, use existing gdf; for texas/california, create from labels
if DATASET == "brazil":
    gdf_train = pd.concat([gdf_train.reset_index(drop=True), train_features], axis=1)
    gdf_train_val = pd.concat(
        [gdf_train_val.reset_index(drop=True), train_val_features], axis=1
    )
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


def train_lgbm(gdf_train, gdf_train_val, gdf_val, gdf_test, run_name):
    if check_if_already_ran(EXPERIMENT_NAME, run_name):
        print(run_name, "already ran in", EXPERIMENT_NAME)
        return

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

    run_gemos(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger, True, 100)
    save_other_model(lgbm, mlflow_logger)


def train_svc(gdf_train, gdf_train_val, gdf_val, gdf_test, run_name, n_trials=60):
    if check_if_already_ran(EXPERIMENT_NAME, run_name):
        print(run_name, "already ran in", EXPERIMENT_NAME)
        return

    if len(gdf_train) > 10_000:
        sampled_gdf_train = (
            gdf_train.sample(10_000, random_state=42).reset_index(drop=True).copy()
        )
    else:
        sampled_gdf_train = gdf_train

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

    study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS, show_progress_bar=True)

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

    run_gemos(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger, True, 100)
    save_other_model(svc_best, mlflow_logger)


def train_rf(gdf_train, gdf_train_val, gdf_val, gdf_test, run_name, n_trials=30):
    if check_if_already_ran(EXPERIMENT_NAME, run_name):
        print(run_name, "already ran in", EXPERIMENT_NAME)
        return

    X_train = gdf_train[X_COLUMNS].to_numpy().astype(np.float32)
    y_train = gdf_train["crop_class"].values

    X_train_val = gdf_train_val[X_COLUMNS].to_numpy().astype(np.float32)

    X_val = gdf_val[X_COLUMNS].to_numpy().astype(np.float32)
    y_val = gdf_val["crop_class"].values

    X_test = gdf_test[X_COLUMNS].to_numpy().astype(np.float32)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 5, 10],
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

    run_gemos(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger, True, 100)
    save_other_model(rf_best, mlflow_logger)


train_rf(
    gdf_train,
    gdf_train_val,
    gdf_val,
    gdf_test,
    f"RF-{RUN_NAME_SUFFIX}",
    n_trials=N_TRIALS,
)
train_lgbm(gdf_train, gdf_train_val, gdf_val, gdf_test, f"LGBM-{RUN_NAME_SUFFIX}")
train_svc(
    gdf_train,
    gdf_train_val,
    gdf_val,
    gdf_test,
    f"SVM-{RUN_NAME_SUFFIX}",
    n_trials=N_TRIALS,
)
