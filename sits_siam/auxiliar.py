import random
import os
import tempfile
import joblib

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
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearnex import patch_sklearn
from tqdm.std import tqdm

patch_sklearn()


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

            confidence, pooled = model(x, doy, mask)

            logits = model.backbone.classifier(pooled)

            preds = torch.argmax(logits, dim=1)
            proba = F.softmax(logits, dim=1)
            proba_max = proba.max(dim=1).values

            all_true.append(batch["y"].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_proba_max.append(proba_max.cpu().numpy())
            all_conf.append(confidence.cpu().numpy())
            all_pooled.append(pooled.cpu().numpy())

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

    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"{name}_accuracy", acc)
    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, f"{name}_f1_weighted", f1_weighted
    )
    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, f"{name}_f1_micro", f1_micro
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
        final_gdf = dataset.gdf.copy()

        if len(final_gdf) != len(y_pred_names):
            display_string += f"WARNING: Dataset length ({len(final_gdf)}) differs from predictions ({len(y_pred_names)}).\n"
            display_string += "Falling back to saving only predictions DataFrame.\n"
            final_df = predictions_df
            filename = "predictions.parquet"
        else:
            final_df = pd.concat([final_gdf, predictions_df], axis=1)
            filename = "predictions_geo.parquet"

    else:
        display_string += "Dataset does not have 'gdf' attribute. Saving simple predictions DataFrame.\n"
        final_df = predictions_df
        filename = f"{name}.parquet"

    if to_print:
        print(display_string)

    # with tempfile.TemporaryDirectory() as tmpdir:
    #     file_path = os.path.join(tmpdir, filename)
    #     final_df.to_parquet(file_path)
    #     mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)
    # print(f"Artifact saved: {filename}")

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


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# import PCA
from sklearn.decomposition import PCA


def run_gemos(train_gdf, val_gdf, test_gdf, mlflow_logger, preprocess=False):
    train_gdf["right_pred"] = train_gdf.y_pred == train_gdf.y_true
    val_gdf["right_pred"] = val_gdf.y_pred == val_gdf.y_true
    test_gdf["right_pred"] = test_gdf.y_pred == test_gdf.y_true

    X_columns = [
        col_name for col_name in train_gdf.columns if col_name.startswith("emb_")
    ]

    train_gdf["gmm_score"] = np.nan
    val_gdf["gmm_score"] = np.nan
    test_gdf["gmm_score"] = np.nan

    gmms = {}
    for crop_class in tqdm(train_gdf.y_true.unique().tolist()):
        try:
            if preprocess:
                gmms[crop_class] = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(random_state=42)),
                        ("gmm", GaussianMixture(random_state=42, n_components=6)),
                    ]
                )
            else:
                gmms[crop_class] = GaussianMixture(random_state=42, n_components=6)

            gmms[crop_class].fit(
                train_gdf.loc[
                    (train_gdf.y_true == crop_class) & (train_gdf.right_pred),
                    X_columns,
                ]
            )
        except:  # noqa: E722
            if preprocess:
                gmms[crop_class] = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(random_state=42)),
                        ("gmm", GaussianMixture(random_state=42, n_components=2)),
                    ]
                )
            else:
                gmms[crop_class] = GaussianMixture(random_state=42, n_components=2)

            gmms[crop_class].fit(
                train_gdf.loc[
                    (train_gdf.y_true == crop_class) & (train_gdf.right_pred),
                    X_columns,
                ]
            )

        gmm_model = gmms[crop_class]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, f"{crop_class}.joblib")
            joblib.dump(gmm_model, file_path)
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

        mask = train_gdf["y_pred"] == crop_class
        if mask.any():
            X_subset = train_gdf.loc[mask, X_columns]
            scores = gmm_model.score_samples(X_subset)
            # aics = gmm_model.aic(X_subset)
            # bics = gmm_model.bic(X_subset)

            train_gdf.loc[mask, "gmm_score"] = scores
            # train_gdf.loc[mask, "aic"] = aics
            # train_gdf.loc[mask, "bic"] = bics

        mask = val_gdf["y_pred"] == crop_class
        if mask.any():
            X_subset = val_gdf.loc[mask, X_columns]
            scores = gmm_model.score_samples(X_subset)
            # aics = gmm_model.aic(X_subset)
            # bics = gmm_model.bic(X_subset)

            val_gdf.loc[mask, "gmm_score"] = scores
            # val_gdf.loc[mask, "aic"] = aics
            # val_gdf.loc[mask, "bic"] = bics

        mask = test_gdf["y_pred"] == crop_class
        if mask.any():
            X_subset = test_gdf.loc[mask, X_columns]
            scores = gmm_model.score_samples(X_subset)
            # aics = gmm_model.aic(X_subset)
            # bics = gmm_model.bic(X_subset)

            test_gdf.loc[mask, "gmm_score"] = scores
            # test_gdf.loc[mask, "aic"] = aics
            # test_gdf.loc[mask, "bic"] = bics

    gmm_gemos_threshold = (
        train_gdf[train_gdf.right_pred]
        .groupby("y_pred")["gmm_score"]
        .quantile(0.025)
        .to_dict()
    )

    print("GMM-GEMOS Thresholds:")
    print(gmm_gemos_threshold)
    print()

    train_gdf["gmm_gemos_anomaly"] = train_gdf.apply(
        lambda row: gmm_gemos_threshold[row.y_pred] >= row.gmm_score, axis=1
    )
    val_gdf["gmm_gemos_anomaly"] = val_gdf.apply(
        lambda row: gmm_gemos_threshold[row.y_pred] >= row.gmm_score, axis=1
    )
    test_gdf["gmm_gemos_anomaly"] = test_gdf.apply(
        lambda row: gmm_gemos_threshold[row.y_pred] >= row.gmm_score, axis=1
    )

    train_gdf["gmm_pred"] = train_gdf.apply(
        lambda row: row.y_pred if not row.gmm_gemos_anomaly else "ZUnknown", axis=1
    )
    val_gdf["gmm_pred"] = val_gdf.apply(
        lambda row: row.y_pred if not row.gmm_gemos_anomaly else "ZUnknown", axis=1
    )
    test_gdf["gmm_pred"] = test_gdf.apply(
        lambda row: row.y_pred if not row.gmm_gemos_anomaly else "ZUnknown", axis=1
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "train.parquet")
        train_gdf.to_parquet(file_path, compression="brotli")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

        file_path = os.path.join(tmpdir, "val.parquet")
        val_gdf.to_parquet(file_path, compression="brotli")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

        file_path = os.path.join(tmpdir, "test.parquet")
        test_gdf.to_parquet(file_path, compression="brotli")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)


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
