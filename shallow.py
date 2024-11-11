import os
import tsfresh
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest
from pprint import pprint
import numpy as np
import torch
import random


def setup_seed():
    seed = 123

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def beautify_prints():
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)


setup_seed()
beautify_prints()

RECALCULATE_FEATURE_SELECTION = False

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


df = pd.read_parquet("data/california_sits_bert_original.parquet")

if RECALCULATE_FEATURE_SELECTION:
    train_val_df = df[df.use_bert != 2].reset_index(drop=True)
    train_val_labels = train_val_df[["label", "use_bert", "id"]].groupby("id").first()

    train_val_df.drop(columns=["label", "use_bert"], inplace=True)
    train_val_features = tsfresh.extract_features(
        train_val_df,
        column_id="id",
        column_sort="time",
    )

    train_val_dataset = train_val_labels.merge(
        train_val_features, left_index=True, right_index=True
    )

    all_column_features = train_val_dataset.columns[2:]

    model = LGBMClassifier(verbosity=-1, n_jobs=-1)
    model.fit(
        train_val_dataset[train_val_dataset.use_bert == 0][
            all_column_features
        ].to_numpy(),
        train_val_dataset[train_val_dataset.use_bert == 0].label.to_numpy(),
    )

    y_pred = model.predict(
        train_val_dataset[train_val_dataset.use_bert == 1][
            all_column_features
        ].to_numpy()
    )
    y_true = train_val_dataset[train_val_dataset.use_bert == 1].label.to_numpy()

    print(
        f"LGBM on validation dataset with all features. OA={accuracy_score(y_true, y_pred):.2%}, Kappa={cohen_kappa_score(y_true, y_pred):.2f}"
    )

    train_val_dataset = train_val_dataset.dropna(axis=1)
    train_val_dataset = train_val_dataset.loc[
        :, (train_val_dataset != train_val_dataset.iloc[0]).any()
    ]
    selector = SelectKBest(k=30)
    selector.fit(
        train_val_dataset[train_val_dataset.use_bert == 0]
        .drop(columns=["label", "use_bert"])
        .to_numpy(),
        train_val_dataset[train_val_dataset.use_bert == 0].label.to_numpy(),
    )

    selected_features = (
        train_val_dataset.drop(columns=["label", "use_bert"])
        .columns.to_numpy()[selector.get_support()]
        .tolist()
    )
    print("Number of selected features: ", len(selected_features))

    model = LGBMClassifier(verbosity=-1, n_jobs=-1)
    model.fit(
        train_val_dataset[train_val_dataset.use_bert == 0][
            selected_features
        ].to_numpy(),
        train_val_dataset[train_val_dataset.use_bert == 0].label.to_numpy(),
    )

    y_pred = model.predict(
        train_val_dataset[train_val_dataset.use_bert == 1][selected_features].to_numpy()
    )
    y_true = train_val_dataset[train_val_dataset.use_bert == 1].label.to_numpy()

    print(
        f"LGBM on validation dataset with selected features. OA={accuracy_score(y_true, y_pred):.2%}, Kappa={cohen_kappa_score(y_true, y_pred):.2f}"
    )

    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(
        selected_features
    )

    pprint(kind_to_fc_parameters)

else:
    kind_to_fc_parameters = {
        "nir": {
            "absolute_sum_of_changes": None,
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0}
            ],
            "cid_ce": [{"normalize": False}],
            "fft_coefficient": [{"attr": "real", "coeff": 1}],
            "mean_abs_change": None,
        },
        "red": {"absolute_sum_of_changes": None, "standard_deviation": None},
        "red_edge_2": {
            "absolute_sum_of_changes": None,
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0}
            ],
            "cid_ce": [{"normalize": False}],
            "mean_abs_change": None,
        },
        "red_edge_3": {
            "absolute_sum_of_changes": None,
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.0},
                {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.0},
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0},
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2},
            ],
            "cid_ce": [{"normalize": False}],
            "fft_coefficient": [{"attr": "real", "coeff": 1}],
            "mean_abs_change": None,
        },
        "red_edge_4": {
            "absolute_sum_of_changes": None,
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0},
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2},
            ],
            "cid_ce": [{"normalize": False}],
            "fft_coefficient": [{"attr": "real", "coeff": 1}],
            "mean_abs_change": None,
            "variation_coefficient": None,
        },
        "swir_1": {"absolute_sum_of_changes": None, "cid_ce": [{"normalize": False}]},
        "swir_2": {
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2}
            ],
            "standard_deviation": None,
        },
    }

    df = df[df.use_bert != 2].reset_index(drop=True)
    labels = df[["label", "use_bert", "id"]].groupby("id").first()
    features = tsfresh.extract_features(
        df,
        column_id="id",
        column_sort="time",
        kind_to_fc_parameters=kind_to_fc_parameters,
    )

    df = labels.merge(features, left_index=True, right_index=True)
    model = LGBMClassifier(verbosity=-1, n_jobs=-1)
    model.fit(
        df[df.use_bert == 0].drop(columns=["label", "use_bert"]).to_numpy(),
        df[df.use_bert == 0].label.to_numpy(),
    )
    y_true = df[df.use_bert == 1].label.to_numpy()
    y_pred = model.predict(
        df[df.use_bert == 1].drop(columns=["label", "use_bert"]).to_numpy()
    )

    print(
        f"LGBM on validation dataset with selected features. OA={accuracy_score(y_true, y_pred):.2%}, Kappa={cohen_kappa_score(y_true, y_pred):.2%}"
    )

df = pd.read_parquet("data/california_sits_bert_original.parquet")
labels = df[["label", "use_bert", "id"]].groupby("id").first()
features = tsfresh.extract_features(
    df,
    column_id="id",
    column_sort="time",
    kind_to_fc_parameters=kind_to_fc_parameters,
)
df = labels.merge(features, left_index=True, right_index=True)

model = LGBMClassifier(verbosity=-1, n_jobs=-1)
model.fit(
    df[df.use_bert != 2].drop(columns=["label", "use_bert"]).to_numpy(),
    df[df.use_bert != 2].label.to_numpy(),
)

df = df[df.use_bert == 2].reset_index(drop=True)

y_true = df.label.to_numpy()
y_pred = model.predict(df.drop(columns=["label", "use_bert"]).to_numpy())
print(
    f"LGBM on test dataset with selected features. OA={accuracy_score(y_true, y_pred):.2%}, Kappa={cohen_kappa_score(y_true, y_pred):.2%}"
)

pprint(confusion_matrix(y_true, y_pred, normalize="true") * 100)
