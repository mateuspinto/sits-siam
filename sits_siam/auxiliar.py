import random
import os
import tempfile
import joblib
import json

import geopandas as gpd
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearnex import patch_sklearn
from tqdm.std import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import optuna
from imblearn.metrics import specificity_score
from imblearn.metrics import geometric_mean_score
import mlflow


patch_sklearn()


def check_if_already_ran(experiment_name, run_name):
    runs_df = mlflow.search_runs(experiment_names=[experiment_name])
    return len(runs_df[runs_df["tags.mlflow.runName"] == run_name]) > 0


def log_results_in_mlflow(gdf_train, gdf_train_val, gdf_val, gdf_test, mlflow_logger):
    for name, gdf in zip(
        ["train_aug", "train", "val", "test"],
        [gdf_train, gdf_train_val, gdf_val, gdf_test],
    ):
        acc = accuracy_score(gdf["y_true"], gdf["y_pred"])
        f1_weighted = f1_score(gdf["y_true"], gdf["y_pred"], average="weighted")
        f1_micro = f1_score(gdf["y_true"], gdf["y_pred"], average="micro")
        f1_macro = f1_score(gdf["y_true"], gdf["y_pred"], average="macro")
        gms = geometric_mean_score(gdf["y_true"], gdf["y_pred"], average="macro")
        spec = specificity_score(gdf["y_true"], gdf["y_pred"], average="macro")
        matt = matthews_corrcoef(gdf["y_true"], gdf["y_pred"])

        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_accuracy", acc
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_f1_weighted", f1_weighted
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_f1_micro", f1_micro
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_f1_macro", f1_macro
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_geometric_mean_score", gms
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_specificity", spec
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_matthews_corrcoef", matt
        )


def setup_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def beautify_prints():
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)


def string_confusion_matrix(y_true, y_pred, class_names=None) -> str:
    if class_names is None:
        class_names = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    max_cell_width = max(len(str(np.max(cm))), 1)
    max_label_width = max(len(name) for name in class_names)
    max_header_height = max(len(name) for name in class_names)
    output = []

    true_label_pad = max(max_label_width, 4)
    left_pad_width = true_label_pad + 3
    left_padding_str = " " * left_pad_width

    matrix_content_width = len(class_names) * (max_cell_width + 2) - 2

    output.append(f"{left_padding_str}{'Predicted':^{matrix_content_width}}")

    for i in range(max_header_height):
        row_str = left_padding_str
        cells = []
        for name in class_names:
            letter = name[i] if i < len(name) else " "
            cells.append(f"{letter:^{max_cell_width}}")
        row_str += "  ".join(cells)
        output.append(row_str)

    separator = "-" * matrix_content_width
    output.append(f"{'True':>{true_label_pad}} | {separator}")

    for i, row in enumerate(cm):
        row_label = class_names[i]
        row_str = f"{row_label:>{true_label_pad}} |"
        cells = []
        for val in row:
            cells.append(f"{val:>{max_cell_width}}")
        row_str += "  ".join(cells)
        output.append(row_str)

    return "\n".join(output)


def predict_and_save_predictions(
    model,
    dataloader,
    dataset,
    mlflow_logger,
    name,
    class_map,
    to_print=False,
    device="cuda",
) -> gpd.GeoDataFrame:
    display_string = ""

    model.eval()
    model.to(device)

    all_true = []
    all_preds = []
    all_proba_max = []
    all_conf = []
    all_pooled = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Predicting {name}"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            x, doy, mask = batch["x"], batch["doy"], batch["mask"]

            confidence, logits, last_emb, all_embs = model(x, doy, mask)

            preds = torch.argmax(logits, dim=1)
            proba = F.softmax(logits, dim=1)
            proba_max = proba.max(dim=1).values

            all_true.append(batch["y"].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_proba_max.append(proba_max.cpu().numpy())
            all_conf.append(confidence.cpu().numpy())
            all_pooled.append(all_embs.cpu().numpy())

    y_true_int = np.concatenate(all_true, axis=0).squeeze()
    y_pred_int = np.concatenate(all_preds, axis=0)
    y_proba_max = np.concatenate(all_proba_max, axis=0)
    y_conf = np.concatenate(all_conf, axis=0).squeeze()
    y_pooled = np.concatenate(all_pooled, axis=0).astype(np.float32)

    y_true_names = np.array([class_map[num] for num in y_true_int])
    y_pred_names = np.array([class_map[num] for num in y_pred_int])

    acc = accuracy_score(y_true_names, y_pred_names)
    f1_weighted = f1_score(y_true_names, y_pred_names, average="weighted")
    f1_micro = f1_score(y_true_names, y_pred_names, average="micro")
    gms = geometric_mean_score(y_true_names, y_pred_names, average="macro")
    spec = specificity_score(y_true_names, y_pred_names, average="macro")
    matt = matthews_corrcoef(y_true_names, y_pred_names)

    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"{name}_accuracy", acc)
    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, f"{name}_f1_weighted", f1_weighted
    )
    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, f"{name}_f1_micro", f1_micro
    )
    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, f"{name}_geometric_mean_score", gms
    )
    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, f"{name}_specificity", spec
    )
    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, f"{name}_matthews_corrcoef", matt
    )

    def compute_failure_metrics(y_true_correct, y_confidence, prefix) -> str:
        display_string = ""

        y_true_error = (~y_true_correct).astype(int)
        y_true_success = y_true_correct.astype(int)

        y_score_error = 1.0 - y_confidence

        auroc = roc_auc_score(y_true_error, y_score_error)

        precision_err, recall_err, _ = precision_recall_curve(
            y_true_error, y_score_error
        )
        aupr_error = auc(recall_err, precision_err)

        precision_succ, recall_succ, _ = precision_recall_curve(
            y_true_success, y_confidence
        )
        aupr_success = auc(recall_succ, precision_succ)

        fpr, tpr, _ = roc_curve(y_true_error, y_score_error)
        idx = np.argmax(tpr >= 0.95)
        fpr_at_95_tpr = fpr[idx] if idx < len(fpr) else fpr[-1]

        display_string += f"\n--- Failure Prediction Metrics ({prefix}) ---\n"
        display_string += f"{name}_AUC: {auroc:.4f}\n"
        display_string += f"{name}_AUPR-Error: {aupr_error:.4f}\n"
        display_string += f"{name}_AUPR-Success: {aupr_success:.4f}\n"
        display_string += f"{name}_FPR-95%-TPR: {fpr_at_95_tpr:.4f}\n"

        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_{prefix}_AUC", auroc
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_{prefix}_AUPR_Error", aupr_error
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_{prefix}_AUPR_Success", aupr_success
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_{prefix}_FPR_95_TPR", fpr_at_95_tpr
        )

        return display_string

    is_correct = y_true_int == y_pred_int

    display_string += compute_failure_metrics(is_correct, y_conf, prefix="conf")

    display_string += compute_failure_metrics(is_correct, y_proba_max, prefix="mcp")

    display_string += "\n--- Confusion Matrix ---\n"
    class_names = [class_map[i] for i in sorted(class_map)]
    display_string += (
        string_confusion_matrix(y_true_names, y_pred_names, class_names) + "\n"
    )

    display_string += "\n--- Classification Report ---\n"
    display_string += (
        str(classification_report(y_true_names, y_pred_names, zero_division=1)) + "\n"
    )

    # Criar DataFrame com todas as predições e embeddings
    n_emb_dims = y_pooled.shape[1]
    emb_dict = {f"emb_{i}": y_pooled[:, i] for i in range(n_emb_dims)}

    pred_dict = {
        "y_true": y_true_names,
        "y_pred": y_pred_names,
        "y_proba": y_proba_max,
        "y_conf": y_conf,
    }
    pred_dict.update(emb_dict)
    predictions_df = pd.DataFrame(pred_dict)

    if hasattr(dataset, "gdf"):
        display_string += "Merging predictions with original GeoDataFrame...\n"
        try:
            final_gdf = dataset.gdf.copy()
        except:  # noqa: E722
            final_gdf = pd.DataFrame()

        if len(final_gdf) != len(y_pred_names):
            display_string += f"WARNING: Dataset length ({len(final_gdf)}) differs from predictions ({len(y_pred_names)}).\n"
            display_string += "Falling back to saving only predictions DataFrame.\n"
            final_df = predictions_df
        else:
            final_df = pd.concat([final_gdf, predictions_df], axis=1)

    else:
        display_string += "Dataset does not have 'gdf' attribute. Saving simple predictions DataFrame.\n"
        final_df = predictions_df
    if to_print:
        print(display_string)

    return final_df


class KNNCallback(pl.Callback):
    def __init__(
        self, train_dataloader, val_dataloader, every_n_epochs=10, num_classes=13, k=5
    ):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.every_n_epochs = every_n_epochs
        self.num_classes = num_classes
        self.k = k

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        device = pl_module.device

        train_feats = []
        train_labels = []
        with torch.no_grad():
            for batch in self.train_dataloader:
                x = batch["x"].to(device)
                doy = batch["doy"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y"].to(device)

                pooled, _, _ = pl_module.backbone(x, doy, mask)

                train_feats.append(pooled.cpu().numpy())
                train_labels.append(y.cpu().numpy())
        train_feats = np.concatenate(train_feats, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        val_feats = []
        val_labels = []
        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch["x"].to(device)
                doy = batch["doy"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y"].to(device)

                pooled, _, _ = pl_module.backbone(x, doy, mask)
                val_feats.append(pooled.cpu().numpy())
                val_labels.append(y.cpu().numpy())
        val_feats = np.concatenate(val_feats, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(train_feats, train_labels)
        preds = knn.predict(val_feats)

        acc = accuracy_score(val_labels, preds)
        f1_weighted = f1_score(val_labels, preds, average="weighted", zero_division=0)
        f1_micro = f1_score(val_labels, preds, average="micro", zero_division=0)

        pl_module.log("knn_acc", acc, prog_bar=True, logger=True, sync_dist=True)
        pl_module.log(
            "knn_f1_weighted", f1_weighted, prog_bar=True, logger=True, sync_dist=True
        )
        pl_module.log(
            "knn_f1_micro", f1_micro, prog_bar=True, logger=True, sync_dist=True
        )

        pl_module.train()


def run_gemos(
    train_gdf, train_val_gdf, val_gdf, test_gdf, mlflow_logger, preprocess=False
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_gdf["right_pred"] = train_gdf.y_pred == train_gdf.y_true
    train_val_gdf["right_pred"] = train_val_gdf.y_pred == train_val_gdf.y_true
    val_gdf["right_pred"] = val_gdf.y_pred == val_gdf.y_true
    test_gdf["right_pred"] = test_gdf.y_pred == test_gdf.y_true

    train_val_gdf["gmm_score"] = np.nan
    val_gdf["gmm_score"] = np.nan
    test_gdf["gmm_score"] = np.nan

    X_columns = [
        col_name for col_name in train_gdf.columns if col_name.startswith("emb_")
    ]

    results = {}
    gmms = {}

    classes = train_gdf["y_true"].unique()

    for cls in classes:
        try:
            X_train = (
                train_gdf[(train_gdf["y_true"] == cls) & (train_gdf["right_pred"])][
                    X_columns
                ]
                .to_numpy()
                .astype(np.float32)
            )

            val_cls = val_gdf[val_gdf["y_true"] == cls]
            X_val_pos = (
                val_cls[val_cls["right_pred"]][X_columns].to_numpy().astype(np.float32)
            )
            X_val_neg = (
                val_cls[~val_cls["right_pred"]][X_columns].to_numpy().astype(np.float32)
            )

            def objective(trial):
                params = {
                    "n_components": trial.suggest_int("n_components", 1, 10),
                    "covariance_type": trial.suggest_categorical(
                        "covariance_type", ["full", "tied", "diag", "spherical"]
                    ),
                    "reg_covar": trial.suggest_float("reg_covar", 1e-6, 1e-3, log=True),
                }

                if preprocess:
                    gmm = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(random_state=42)),
                            (
                                "gmm",
                                GaussianMixture(
                                    **params, random_state=42, init_params="k-means++"
                                ),
                            ),
                        ]
                    )
                else:
                    gmm = GaussianMixture(
                        **params, random_state=42, init_params="k-means++"
                    )

                try:
                    gmm.fit(X_train)
                    pos_scores = gmm.score_samples(X_val_pos)
                    neg_scores = gmm.score_samples(X_val_neg)

                    y_scores = np.concatenate([pos_scores, neg_scores])
                    y_true_binary = np.concatenate(
                        [np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
                    )

                    return roc_auc_score(y_true_binary, y_scores)
                except Exception:
                    return 0.0

            if len(X_train) < 10 or len(X_val_pos) < 2 or len(X_val_neg) < 2:
                if preprocess:
                    best_gmm = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(random_state=42)),
                            (
                                "gmm",
                                GaussianMixture(
                                    n_components=3,
                                    random_state=42,
                                    init_params="k-means++",
                                ),
                            ),
                        ]
                    )
                else:
                    best_gmm = GaussianMixture(
                        n_components=1, random_state=42, init_params="k-means++"
                    )
                best_gmm.fit(X_train)

                # Calculate threshold as the percentile 2.5 of the scores on the train set
                best_threshold = np.percentile(best_gmm.score_samples(X_train), 2.5)
                results[cls] = {
                    "threshold": float(best_threshold),
                    "auc": float("nan"),
                    "params": {
                        "random_state": 42,
                        "n_components": 3,
                        "init_params": "k-means++",
                    },
                }
            else:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                    study_name="gemmos",
                )
                study.optimize(objective, n_trials=30, n_jobs=4, show_progress_bar=True)

                best_gmm = GaussianMixture(
                    **study.best_params, random_state=42, init_params="k-means++"
                )
                best_gmm.fit(X_train)

                pos_scores = best_gmm.score_samples(X_val_pos)
                neg_scores = best_gmm.score_samples(X_val_neg)
                y_scores = np.concatenate([pos_scores, neg_scores])
                y_true_binary = np.concatenate(
                    [np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
                )

                precision, recall, thresholds = precision_recall_curve(
                    y_true_binary, y_scores
                )
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                best_threshold = thresholds[np.argmax(f1_scores[:-1])]
                results[cls] = {
                    "threshold": float(best_threshold),
                    "auc": float(study.best_value),
                    "params": study.best_params,
                }

            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, f"{cls}.joblib")
                joblib.dump(best_gmm, file_path)
                mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

            gmms[cls] = best_gmm

            print(
                f"Class {cls}: AUC {results[cls]['auc']:.4f} | Threshold {results[cls]['threshold']:.4f}, best params: {results[cls]['params']}"
            )

            mask = train_val_gdf["y_pred"] == cls
            if mask.any():
                X_subset = train_val_gdf.loc[mask, X_columns]
                scores = best_gmm.score_samples(X_subset)

                train_val_gdf.loc[mask, "gmm_score"] = scores

            mask = val_gdf["y_pred"] == cls
            if mask.any():
                X_subset = val_gdf.loc[mask, X_columns]
                scores = best_gmm.score_samples(X_subset)

                val_gdf.loc[mask, "gmm_score"] = scores

            mask = test_gdf["y_pred"] == cls
            if mask.any():
                X_subset = test_gdf.loc[mask, X_columns]
                scores = best_gmm.score_samples(X_subset)

                test_gdf.loc[mask, "gmm_score"] = scores

        except Exception as e:
            print(f"Erro ao processar classe {cls}: {e}")
            print(f"Atribuindo gmm_score=0 e gmm_gemos_anomaly=False para classe {cls}")

            # Set gmm_score = 0 for all samples of this class
            mask_train_val = train_val_gdf["y_pred"] == cls
            if mask_train_val.any():
                train_val_gdf.loc[mask_train_val, "gmm_score"] = 0.0

            mask_val = val_gdf["y_pred"] == cls
            if mask_val.any():
                val_gdf.loc[mask_val, "gmm_score"] = 0.0

            mask_test = test_gdf["y_pred"] == cls
            if mask_test.any():
                test_gdf.loc[mask_test, "gmm_score"] = 0.0

            # Store a dummy result indicating failure
            results[cls] = {
                "threshold": float("-inf"),  # Will make all samples non-anomalous
                "auc": float("nan"),
                "params": {},
                "error": str(e),
            }

            continue

    print("GMM-GEMOS Thresholds:")
    gmm_gemos_threshold = {cls: data["threshold"] for cls, data in results.items()}
    print(gmm_gemos_threshold)

    train_val_gdf["gmm_gemos_anomaly"] = train_val_gdf.apply(
        lambda row: gmm_gemos_threshold[row.y_pred] >= row.gmm_score, axis=1
    )
    val_gdf["gmm_gemos_anomaly"] = val_gdf.apply(
        lambda row: gmm_gemos_threshold[row.y_pred] >= row.gmm_score, axis=1
    )
    test_gdf["gmm_gemos_anomaly"] = test_gdf.apply(
        lambda row: gmm_gemos_threshold[row.y_pred] >= row.gmm_score, axis=1
    )

    train_val_gdf["gmm_pred"] = train_val_gdf.apply(
        lambda row: row.y_pred if not row.gmm_gemos_anomaly else "ZUnknown", axis=1
    )
    val_gdf["gmm_pred"] = val_gdf.apply(
        lambda row: row.y_pred if not row.gmm_gemos_anomaly else "ZUnknown", axis=1
    )
    test_gdf["gmm_pred"] = test_gdf.apply(
        lambda row: row.y_pred if not row.gmm_gemos_anomaly else "ZUnknown", axis=1
    )

    train_val_gdf["gmm_pred"] = train_val_gdf.apply(
        lambda row: "ZUnknow" if row.gmm_gemos_anomaly else row.y_pred, axis=1
    )
    val_gdf["gmm_pred"] = val_gdf.apply(
        lambda row: "ZUnknow" if row.gmm_gemos_anomaly else row.y_pred, axis=1
    )
    test_gdf["gmm_pred"] = test_gdf.apply(
        lambda row: "ZUnknow" if row.gmm_gemos_anomaly else row.y_pred, axis=1
    )

    for name, current_gdf in zip(
        ["train_val", "val", "test"], [train_val_gdf, val_gdf, test_gdf]
    ):
        acc = accuracy_score(
            current_gdf[~current_gdf.gmm_gemos_anomaly].y_true,
            current_gdf[~current_gdf.gmm_gemos_anomaly].gmm_pred,
        )
        f1_weighted = f1_score(
            current_gdf[~current_gdf.gmm_gemos_anomaly].y_true,
            current_gdf[~current_gdf.gmm_gemos_anomaly].gmm_pred,
            average="weighted",
            zero_division=1,
        )
        f1_micro = f1_score(
            current_gdf[~current_gdf.gmm_gemos_anomaly].y_true,
            current_gdf[~current_gdf.gmm_gemos_anomaly].gmm_pred,
            average="micro",
            zero_division=1,
        )

        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_accuracy_wo_anomaly", acc
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_f1_weighted_wo_anomaly", f1_weighted
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id, f"{name}_f1_micro_wo_anomaly", f1_micro
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id,
            f"{name}_anomaly_percentage",
            current_gdf.gmm_gemos_anomaly.mean(),
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "train.parquet")
        train_val_gdf[X_columns] = train_val_gdf[X_columns].astype(np.float16)
        train_val_gdf.to_parquet(file_path, compression="brotli")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

        file_path = os.path.join(tmpdir, "val.parquet")
        val_gdf[X_columns] = val_gdf[X_columns].astype(np.float16)
        val_gdf.to_parquet(file_path, compression="brotli")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

        file_path = os.path.join(tmpdir, "test.parquet")
        test_gdf[X_columns] = test_gdf[X_columns].astype(np.float16)
        test_gdf.to_parquet(file_path, compression="brotli")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

        thresholds_path = os.path.join(tmpdir, "gmm_infos.json")
        with open(thresholds_path, "w") as f:
            json.dump(results, f)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, thresholds_path)


def save_pytorch_model(model, mlflow_logger):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "weights.pth")
        torch.save(model.state_dict(), file_path)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)


def save_other_model(model, mlflow_logger):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "model.joblib")
        joblib.dump(model, file_path)
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)


def split_with_percent_and_class_coverage(
    gdf_all: gpd.GeoDataFrame, percent: float, max_attempts: int = 200
):
    p = percent / 100.0
    if p <= 0 or p >= 0.5:
        raise ValueError("Percent must be in (0, 50); received %s" % percent)

    unique_mun_local = gdf_all["CD_MUN"].unique()
    classes_full = set(gdf_all["crop_class"].unique())

    val_on_temp = p / (1.0 - p)
    test_on_temp = 1.0 - val_on_temp

    for attempt_seed in range(0, max_attempts):
        mun_train_local, mun_temp_local = train_test_split(
            unique_mun_local, test_size=(1.0 - p), random_state=attempt_seed
        )

        mun_val_local, mun_test_local = train_test_split(
            mun_temp_local, test_size=test_on_temp, random_state=attempt_seed
        )

        gdf_train_local = gdf_all[gdf_all["CD_MUN"].isin(mun_train_local)].copy()
        gdf_val_local = gdf_all[gdf_all["CD_MUN"].isin(mun_val_local)].copy()
        gdf_test_local = gdf_all[gdf_all["CD_MUN"].isin(mun_test_local)].copy()

        train_classes = set(pd.unique(gdf_train_local["crop_class"]))
        val_classes = set(pd.unique(gdf_val_local["crop_class"]))
        test_classes = set(pd.unique(gdf_test_local["crop_class"]))

        if (
            classes_full.issubset(train_classes)
            and classes_full.issubset(val_classes)
            and classes_full.issubset(test_classes)
        ):
            print("SEED USADA PARA DIVISÃO:", attempt_seed)
            return gdf_train_local, gdf_val_local, gdf_test_local

    raise RuntimeError(
        (
            "Não foi possível obter divisão com cobertura de classes em todos os conjuntos "
            f"após {max_attempts} tentativas variando a seed. Ajuste o percentual ({percent}%) ou verifique o dataset."
        )
    )
