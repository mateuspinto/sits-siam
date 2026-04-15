import os
import json
import mlflow
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearnex import patch_sklearn
from tqdm.std import tqdm
from imblearn.metrics import classification_report_imbalanced
from joblib import Parallel, delayed

ESCALA_FONTE = 1.6

def setup_plot_params():
    params = {
        'font.size': 10 * ESCALA_FONTE,
        'axes.titlesize': 12 * ESCALA_FONTE,
        'axes.labelsize': 10 * ESCALA_FONTE,
        'xtick.labelsize': 8 * ESCALA_FONTE,
        'ytick.labelsize': 8 * ESCALA_FONTE,
        'legend.fontsize': 9 * ESCALA_FONTE,
        'figure.titlesize': 16 * ESCALA_FONTE,
        'figure.dpi': 100
    }
    plt.rcParams.update(params)

setup_plot_params()
patch_sklearn()

FIXED_CROP_COLORS = {
    # --- Classes Originais (Ajustadas para maior contraste) ---
    "Other Temporary Crops": "#8B4513", # SaddleBrown (Marrom forte)
    "Pasture": "#2E8B57",               # SeaGreen (Verde médio, diferente de floresta)
    "Sugar Cane": "#DAA520",            # Goldenrod (Amarelo queimado/Ouro)
    "Coffee": "#0000CD",                # MediumBlue (Azul forte, mantendo a matiz original)
    "Forest Plantation": "#006400",     # DarkGreen (Verde escuro profundo)
    "Other Perennial Crops": "#D35400", # Pumpkin (Laranja escuro/Terra)
    "Soybean": "#FFD700",               # Gold (Amarelo Ouro, mais visível que o amarelo pálido)
    "Citrus": "#FF8C00",                # DarkOrange (Laranja vivo)
    "Palm Oil": "#800000",              # Maroon (Vinho/Castanho avermelhado)
    "Rice": "#000080",                  # Navy (Azul marinho bem escuro)
    "ZUnknow": "#000000",               # PRETO (Conforme solicitado)

    # --- Novas Classes (Removidos os tons pastéis/brancos) ---
    "Corn": "#CC0000",           # Vermelho Escuro (Forte, sangue)
    "Cotton": "#00BFFF",         # DeepSkyBlue (Ciano forte/Azul piscina)
    "Oats": "#9400D3",           # DarkViolet (Roxo vibrante, substitui o lavanda claro)
    "Sorghum": "#556B2F",        # DarkOliveGreen (Verde oliva escuro)
    "Wheat": "#CD853F",          # Peru (Marrom claro/Bronze - substitui o bege invisível)
    "Alfalfa": "#32CD32",        # LimeGreen (Verde limão forte, diferente de Pasture)
    "Almonds": "#A0522D",        # Sienna (Marrom madeira, substitui o salmão claro)
    "Grapes": "#4B0082",         # Indigo (Roxo muito escuro/Anil)
    "Pistachios": "#6B8E23",     # OliveDrab (Verde amarelado escuro)
    "Tomatoes": "#FF4500",       # OrangeRed (Vermelho alaranjado vibrante)
    "Walnuts": "#555555",        # DarkGrey (Cinza escuro, mais visível que o cinza claro)
    "Wildflowers": "#C71585",    # MediumVioletRed (Rosa choque escuro, substitui rosa pálido)
    "Wheat and Corn": "#2F4F4F", # DarkSlateGray (Mantido, já é escuro e bom)
}

CROP_NAME_MAPPING = {
    # --- Classes Originais ---
    "Other Temporary Crops": "Temp. Crops",
    "Pasture": "Pasture",
    "Sugar Cane": "Sugar Cane",
    "Coffee": "Coffee",
    "Forest Plantation": "Forest Plant.",
    "Other Perennial Crops": "Peren. Crops",
    "Soybean": "Soybean",
    "Citrus": "Citrus",
    "Palm Oil": "Palm Oil",
    "Rice": "Rice",
    "ZUnknow": "Unknown",

    # --- Novas Classes Adicionadas ---
    "Corn": "Corn",
    "Cotton": "Cotton",
    "Oats": "Oats",
    "Sorghum": "Sorghum",
    "Wheat": "Wheat",
    "Alfalfa": "Alfalfa",
    "Almonds": "Almonds",
    "Grapes": "Grapes",
    "Pistachios": "Pistachios",
    "Tomatoes": "Tomatoes",
    "Walnuts": "Walnuts",
    "Wildflowers": "Wildflw.",
    "Wheat and Corn": "Wheat/Corn",
}

CROP_MARKER_MAPPING = {
    # --- Classes Originais ---
    "Other Temporary Crops": "s",      # Marrom (SaddleBrown) -> Quadrado
    "Pasture": "s",                    # Verde (SeaGreen) -> Quadrado
    "Sugar Cane": "^",                 # Amarelo (Goldenrod) -> Triângulo
    "Coffee": "o",                     # Azul (MediumBlue)
    "Forest Plantation": "o",          # Verde Escuro (DarkGreen)
    "Other Perennial Crops": "o",      # Laranja/Terra (Pumpkin)
    "Soybean": "o",                    # Amarelo Ouro (Gold)
    "Citrus": "o",                     # Laranja (DarkOrange)
    "Palm Oil": "o",                   # Vinho (Maroon)
    "Rice": "o",                       # Azul Marinho (Navy)
    "ZUnknow": "x",                    # Preto -> X (Conforme solicitado)

    # --- Novas Classes Adicionadas ---
    "Corn": "o",                       # Vermelho
    "Cotton": "o",                     # Ciano
    "Oats": "o",                       # Roxo vibrante
    "Sorghum": "^",                    # Verde Oliva (Diferencia de Pasture/Forest)
    "Wheat": "d",                      # Bronze (Diferencia de Other Temp. Crops)
    "Alfalfa": "d",                    # Verde Limão (Diferencia de Pasture)
    "Almonds": "o",                    # Marrom Madeira
    "Grapes": "d",                     # Indigo (Diferencia de Oats/Rice)
    "Pistachios": "o",                 # Verde Amarelado
    "Tomatoes": "o",                   # Vermelho Alaranjado
    "Walnuts": "s",                    # Cinza Escuro (Diferencia do Unknown/Preto)
    "Wildflowers": "o",                # Rosa Choque
    "Wheat and Corn": "o",             # DarkSlateGray
}


def get_x_columns(gdf):
    return [col for col in gdf.columns if col.startswith("emb_")]


def remap_crop_names(gdf):
    for col in ['y_pred', 'y_true', 'gmm_pred']:
        if col in gdf.columns:
            gdf[col] = gdf[col].map(lambda x: CROP_NAME_MAPPING.get(x, x))
    return gdf


def plot_tsne_from_gdf(
    gdf, color_column, x_columns, n_samples=1000, ax=None, color_map=None, alpha=0.9
):
    n_samples = min(n_samples, len(gdf))
    sample_gdf = gdf.sample(n=n_samples, random_state=42)

    X_sample = sample_gdf[x_columns]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sample)

    sample_gdf["tsne1"] = X_tsne[:, 0]
    sample_gdf["tsne2"] = X_tsne[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    palette_to_use = color_map if color_map is not None else "deep"

    unique_classes = sample_gdf[color_column].unique()
    for class_name in unique_classes:
        class_data = sample_gdf[sample_gdf[color_column] == class_name]
        marker = CROP_MARKER_MAPPING.get(class_name, 'o')
        color = palette_to_use.get(class_name) if isinstance(palette_to_use, dict) else None
        
        ax.scatter(
            class_data["tsne1"],
            class_data["tsne2"],
            label=class_name,
            marker=marker,
            c=[color] if color else None,
            alpha=alpha,
            s=50,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax.grid(True, linestyle="--", alpha=0.6)

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_mcp_histograms(train_gdf, val_gdf, test_gdf, class_names, run_name, experiment_name):
    setup_plot_params()
    fig, ax = plt.subplots(3, len(class_names), figsize=(25, 7), sharex=True)

    for n, crop_class in enumerate(class_names):
        subpart_train_gdf = train_gdf[train_gdf.y_pred == crop_class].reset_index(
            drop=True
        )
        subpart_val_gdf = val_gdf[val_gdf.y_pred == crop_class].reset_index(drop=True)
        subpart_test_gdf = test_gdf[test_gdf.y_pred == crop_class].reset_index(
            drop=True
        )

        subpart_train_gdf[subpart_train_gdf.right_pred].y_proba.plot.hist(
            color="green", ax=ax[0][n]
        )
        subpart_train_gdf[~subpart_train_gdf.right_pred].y_proba.plot.hist(
            color="red", ax=ax[0][n], alpha=0.5
        )

        subpart_val_gdf[subpart_val_gdf.right_pred].y_proba.plot.hist(
            color="green", ax=ax[1][n]
        )
        subpart_val_gdf[~subpart_val_gdf.right_pred].y_proba.plot.hist(
            color="red", ax=ax[1][n], alpha=0.5
        )

        subpart_test_gdf[subpart_test_gdf.right_pred].y_proba.plot.hist(
            color="green", ax=ax[2][n]
        )
        subpart_test_gdf[~subpart_test_gdf.right_pred].y_proba.plot.hist(
            color="red", ax=ax[2][n], alpha=0.5
        )

        ax[0][n].set_title(crop_class)
        ax[1][n].set_title("")
        ax[2][n].set_title("")

        if n == 0:
            ax[0][n].set_ylabel("Train")
            ax[1][n].set_ylabel("Validation")
            ax[2][n].set_ylabel("Test")
        else:
            ax[0][n].set_ylabel("")
            ax[1][n].set_ylabel("")
            ax[2][n].set_ylabel("")

        ax[0][n].set_xlim(0, 1)
        ax[1][n].set_xlim(0, 1)

        for i in range(3):
            ax[i][n].set_xticks([])
            ax[i][n].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"figs/{experiment_name}/{run_name}/hist_mcp.pdf", bbox_inches="tight")
    plt.close()


def plot_confidnet_histograms(train_gdf, val_gdf, test_gdf, class_names, run_name, experiment_name):
    setup_plot_params()
    fig, ax = plt.subplots(3, len(class_names), figsize=(25, 7), sharex=True)

    for n, crop_class in enumerate(class_names):
        subpart_train_gdf = train_gdf[train_gdf.y_pred == crop_class].reset_index(
            drop=True
        )
        subpart_val_gdf = val_gdf[val_gdf.y_pred == crop_class].reset_index(drop=True)
        subpart_test_gdf = test_gdf[test_gdf.y_pred == crop_class].reset_index(
            drop=True
        )

        subpart_train_gdf[subpart_train_gdf.right_pred].y_conf.plot.hist(
            color="green", ax=ax[0][n]
        )
        subpart_train_gdf[~subpart_train_gdf.right_pred].y_conf.plot.hist(
            color="red", ax=ax[0][n], alpha=0.5
        )

        subpart_val_gdf[subpart_val_gdf.right_pred].y_conf.plot.hist(
            color="green", ax=ax[1][n]
        )
        subpart_val_gdf[~subpart_val_gdf.right_pred].y_conf.plot.hist(
            color="red", ax=ax[1][n], alpha=0.5
        )

        subpart_test_gdf[subpart_test_gdf.right_pred].y_conf.plot.hist(
            color="green", ax=ax[2][n]
        )
        subpart_test_gdf[~subpart_test_gdf.right_pred].y_conf.plot.hist(
            color="red", ax=ax[2][n], alpha=0.5
        )

        ax[0][n].set_title(crop_class)
        ax[1][n].set_title("")
        ax[2][n].set_title("")

        if n == 0:
            ax[0][n].set_ylabel("Train")
            ax[1][n].set_ylabel("Validation")
            ax[2][n].set_ylabel("Test")
        else:
            ax[0][n].set_ylabel("")
            ax[1][n].set_ylabel("")
            ax[2][n].set_ylabel("")

        ax[0][n].set_xlim(0, 1)
        ax[1][n].set_xlim(0, 1)

        for i in range(3):
            ax[i][n].set_xticks([])
            ax[i][n].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"figs/{experiment_name}/{run_name}/hist_confidnet.pdf", bbox_inches="tight")
    plt.close()


def plot_gmm_score_histograms(
    train_gdf, val_gdf, test_gdf, gmm_thresholds, class_names, run_name, experiment_name
):
    setup_plot_params()
    fig, ax = plt.subplots(3, len(class_names), figsize=(25, 7), sharex="col")

    for n, crop_class in enumerate(class_names):
        subpart_train_gdf = train_gdf[train_gdf.y_pred == crop_class].reset_index(
            drop=True
        )
        subpart_val_gdf = val_gdf[val_gdf.y_pred == crop_class].reset_index(drop=True)
        subpart_test_gdf = test_gdf[test_gdf.y_pred == crop_class].reset_index(
            drop=True
        )

        subpart_train_gdf[subpart_train_gdf.right_pred].gmm_score.plot.hist(
            color="green", ax=ax[0][n]
        )
        subpart_train_gdf[~subpart_train_gdf.right_pred].gmm_score.plot.hist(
            color="red", ax=ax[0][n], alpha=0.5
        )

        subpart_val_gdf[subpart_val_gdf.right_pred].gmm_score.plot.hist(
            color="green", ax=ax[1][n]
        )
        subpart_val_gdf[~subpart_val_gdf.right_pred].gmm_score.plot.hist(
            color="red", ax=ax[1][n], alpha=0.5
        )

        subpart_test_gdf[subpart_test_gdf.right_pred].gmm_score.plot.hist(
            color="green", ax=ax[2][n]
        )
        subpart_test_gdf[~subpart_test_gdf.right_pred].gmm_score.plot.hist(
            color="red", ax=ax[2][n], alpha=0.5
        )

        ax[0][n].set_title(crop_class)
        ax[1][n].set_title("")
        ax[2][n].set_title("")

        if n == 0:
            ax[0][n].set_ylabel("Train")
            ax[1][n].set_ylabel("Validation")
            ax[2][n].set_ylabel("Test")
        else:
            ax[0][n].set_ylabel("")
            ax[1][n].set_ylabel("")
            ax[2][n].set_ylabel("")

        for i in range(3):
            ax[i][n].set_xticks([])
            ax[i][n].set_yticks([])

            ax[i][n].axvline(
                x=gmm_thresholds[crop_class],
                color="blue",
                linestyle="--",
                label="GMM Threshold",
            )

    plt.tight_layout()
    plt.savefig(f"figs/{experiment_name}/{run_name}/hist_gmm.pdf", bbox_inches="tight")
    plt.close()


def plot_classification_results(y_true, y_pred, filename):
    setup_plot_params()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    class_names = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 6 * ESCALA_FONTE},
        cbar_kws={"shrink": 0.8}
    )

    ax1.set_xlabel("Pred", fontsize=12 * ESCALA_FONTE)
    ax1.set_ylabel("True", fontsize=12 * ESCALA_FONTE)
    ax1.tick_params(axis='both', labelsize=9 * ESCALA_FONTE)

    report_dict = classification_report_imbalanced(
        y_true, 
        y_pred, 
        output_dict=True, 
        zero_division=1
    )
    report_df = pd.DataFrame(report_dict).T

    report_df = report_df[~report_df.index.astype(str).str.contains('avg|total')]

    metrics = ["pre", "rec", "spe", "f1", "geo", "iba", "sup"]
    report_df = report_df[metrics]
    report_df[["pre", "rec", "spe", "f1", "geo", "iba"]] = (
        report_df[["pre", "rec", "spe", "f1", "geo", "iba"]] * 100
    ).round()
    report_df = report_df.astype(int)

    report_df = report_df.rename(columns={
        "pre": "Precision",
        "rec": "Recall",
        "spe": "Specificity",
        "f1": "F1-Score",
        "geo": "G-Mean",
        "iba": "IBA",
        "sup": "Support"
    })

    sns.heatmap(
        report_df, 
        annot=True, 
        cmap="RdYlGn", 
        fmt="d", 
        vmin=0, 
        vmax=100, 
        ax=ax2,
        annot_kws={"size": 8 * ESCALA_FONTE},
        cbar_kws={"shrink": 0.8}
    )
    ax2.tick_params(axis='both', labelsize=9 * ESCALA_FONTE)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_classification_results_gemmos(y_true, y_pred, filename):
    setup_plot_params()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    true_classes = sorted(list(set(y_true)))
    all_classes = sorted(list(set(y_true) | set(y_pred)))
    
    cm_manual = []
    for true_class in true_classes:
        row = []
        for pred_class in all_classes:
            count = ((y_true == true_class) & (y_pred == pred_class)).sum()
            row.append(count)
        cm_manual.append(row)
    
    cm_manual = np.array(cm_manual)
    
    sns.heatmap(
        cm_manual,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=all_classes,
        yticklabels=true_classes,
        annot_kws={"size": 6 * ESCALA_FONTE},
        cbar_kws={"shrink": 0.8}
    )

    ax1.set_xlabel("Pred", fontsize=12 * ESCALA_FONTE)
    ax1.set_ylabel("True", fontsize=12 * ESCALA_FONTE)
    ax1.tick_params(axis='both', labelsize=9 * ESCALA_FONTE)

    mask_no_anomaly = y_pred != "Unknown"
    y_true_filtered = y_true[mask_no_anomaly]
    y_pred_filtered = y_pred[mask_no_anomaly]

    report_dict = classification_report_imbalanced(
        y_true_filtered, 
        y_pred_filtered, 
        output_dict=True, 
        zero_division=1
    )
    report_df = pd.DataFrame(report_dict).T

    report_df = report_df[~report_df.index.astype(str).str.contains('avg|total')]

    metrics = ["pre", "rec", "spe", "f1", "geo", "iba", "sup"]
    report_df = report_df[metrics]
    report_df[["pre", "rec", "spe", "f1", "geo", "iba"]] = (
        report_df[["pre", "rec", "spe", "f1", "geo", "iba"]] * 100
    ).round()
    report_df = report_df.astype(int)

    report_df = report_df.rename(columns={
        "pre": "Precision",
        "rec": "Recall",
        "spe": "Specificity",
        "f1": "F1-Score",
        "geo": "G-Mean",
        "iba": "IBA",
        "sup": "Support"
    })

    sns.heatmap(
        report_df, 
        annot=True, 
        cmap="RdYlGn", 
        fmt="d", 
        vmin=0, 
        vmax=100, 
        ax=ax2,
        annot_kws={"size": 8 * ESCALA_FONTE},
        cbar_kws={"shrink": 0.8}
    )
    ax2.tick_params(axis='both', labelsize=9 * ESCALA_FONTE)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix_pdf(y_true, y_pred, filename):
    setup_plot_params()
    # Ajuste o figsize conforme necessário para um único plot
    fig, ax = plt.subplots(figsize=(8, 6))

    true_classes = sorted(list(set(y_true)))
    all_classes = sorted(list(set(y_true) | set(y_pred)))
    
    cm_manual = []
    for true_class in true_classes:
        row = []
        for pred_class in all_classes:
            count = ((y_true == true_class) & (y_pred == pred_class)).sum()
            row.append(count)
        cm_manual.append(row)
    
    cm_manual = np.array(cm_manual)
    
    sns.heatmap(
        cm_manual,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=all_classes,
        yticklabels=true_classes,
        annot_kws={"size": 6 * ESCALA_FONTE},
        cbar_kws={"shrink": 0.8}
    )

    #ax.set_title("Confusion Matrix", fontsize=14 * ESCALA_FONTE)
    ax.set_xlabel("Pred", fontsize=12 * ESCALA_FONTE)
    ax.set_ylabel("True", fontsize=12 * ESCALA_FONTE)
    ax.tick_params(axis='both', labelsize=9 * ESCALA_FONTE)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Matriz de confusão salva em: {filename}")

def plot_metrics_report_pdf(y_true, y_pred, filename):
    setup_plot_params()
    # Ajuste o figsize conforme necessário para um único plot
    fig, ax = plt.subplots(figsize=(8, 6))

    mask_no_anomaly = y_pred != "Unknown"
    y_true_filtered = y_true[mask_no_anomaly]
    y_pred_filtered = y_pred[mask_no_anomaly]

    report_dict = classification_report_imbalanced(
        y_true_filtered, 
        y_pred_filtered, 
        output_dict=True, 
        zero_division=1
    )
    report_df = pd.DataFrame(report_dict).T

    report_df = report_df[~report_df.index.astype(str).str.contains('avg|total')]

    metrics = ["pre", "rec", "spe", "f1", "geo", "iba", "sup"]
    report_df = report_df[metrics]
    report_df[["pre", "rec", "spe", "f1", "geo", "iba"]] = (
        report_df[["pre", "rec", "spe", "f1", "geo", "iba"]] * 100
    ).round()
    report_df = report_df.astype(int)

    report_df = report_df.rename(columns={
        "pre": "Precision",
        "rec": "Recall",
        "spe": "Specificity",
        "f1": "F1-Score",
        "geo": "G-Mean",
        "iba": "IBA",
        "sup": "Support"
    })

    sns.heatmap(
        report_df, 
        annot=True, 
        cmap="RdYlGn", 
        fmt="d", 
        vmin=0, 
        vmax=100, 
        ax=ax,
        annot_kws={"size": 8 * ESCALA_FONTE},
        cbar_kws={"shrink": 0.8}
    )
    #ax.set_title("Classification Metrics", fontsize=14 * ESCALA_FONTE)
    ax.tick_params(axis='both', labelsize=9 * ESCALA_FONTE)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Relatório de métricas salvo em: {filename}")

def plot_classification_results_gemmos(y_true, y_pred, base_filename):
    """
    Função wrapper que gera os dois PDFs separadamente.
    Se base_filename for 'resultado.pdf', ele vai gerar:
    - resultado_matrix.pdf
    - resultado_report.pdf
    """
    base, ext = os.path.splitext(base_filename)
    if not ext:
        ext = ".pdf"
    
    file_matrix = f"{base}_matrix{ext}"
    file_report = f"{base}_report{ext}"

    plot_confusion_matrix_pdf(y_true, y_pred, file_matrix)
    plot_metrics_report_pdf(y_true, y_pred, file_report)

def plot_embeddings_gemmos(test_gdf, x_columns, run_name, experiment_name):
    if "gmm_pred" not in test_gdf.columns:
        return

    setup_plot_params()
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(32, 7))

    plot_tsne_from_gdf(
        test_gdf,
        color_column="y_true",
        x_columns=x_columns,
        color_map=FIXED_CROP_COLORS,
        n_samples=3000,
        ax=ax[0],
        alpha=0.7,
    )
    plot_tsne_from_gdf(
        test_gdf,
        color_column="gmm_pred",
        x_columns=x_columns,
        color_map=FIXED_CROP_COLORS,
        n_samples=3000,
        ax=ax[1],
        alpha=0.7,
    )
    plot_tsne_from_gdf(
        test_gdf[~test_gdf.gmm_gemos_anomaly],
        color_column="y_true",
        x_columns=x_columns,
        color_map=FIXED_CROP_COLORS,
        n_samples=3000,
        ax=ax[2],
        alpha=0.7,
    )
    plot_tsne_from_gdf(
        test_gdf[~test_gdf.gmm_gemos_anomaly],
        color_column="gmm_pred",
        x_columns=x_columns,
        color_map=FIXED_CROP_COLORS,
        n_samples=3000,
        ax=ax[3],
        alpha=0.7,
    )

    ax[0].set_title("True Classes", fontsize=16 * ESCALA_FONTE)
    ax[1].set_title("Pred Classes", fontsize=16 * ESCALA_FONTE)
    ax[2].set_title("True Classes (Wo Anom.)", fontsize=16 * ESCALA_FONTE)
    ax[3].set_title("Pred Classes (Wo Anom.)", fontsize=16 * ESCALA_FONTE)

    ax[0].legend_.remove()
    ax[1].legend_.remove()
    ax[2].legend_.remove()

    for i in range(4):
        ax[i].set_ylabel("")
        ax[i].set_xlabel("")
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    plt.tight_layout()
    plt.savefig(f"figs/{experiment_name}/{run_name}/tsne_embeddings.pdf", bbox_inches="tight")
    plt.close()


def load_data_for_run(runs_df, run_name):
    artifact_uri = runs_df.loc[run_name].artifact_uri
    test_gdf = pd.read_parquet(f"{artifact_uri}/test.parquet")
    train_gdf = pd.read_parquet(f"{artifact_uri}/train.parquet")
    val_gdf = pd.read_parquet(f"{artifact_uri}/val.parquet")

    try:
        with open(f"{artifact_uri[5:]}/gmm_infos.json", "r") as f:
            gmm_infos = json.load(f)
            gmm_thresholds = {k: v["threshold"] for k, v in gmm_infos.items()}
            gmm_thresholds = {CROP_NAME_MAPPING.get(k, k): v for k, v in gmm_thresholds.items()}
    except FileNotFoundError:
        gmm_thresholds = None

    return train_gdf, val_gdf, test_gdf, gmm_thresholds


def add_gmm_pred_column(gdf):
    if "gmm_gemos_anomaly" in gdf.columns:
        gdf["gmm_pred"] = gdf.apply(
            lambda row: "ZUnknow" if row.gmm_gemos_anomaly else row.y_pred, axis=1
        )
    return gdf

def plot_combined_histograms_test(test_gdf, gmm_thresholds, class_names, run_name, experiment_name):
    """
    Plota histogramas combinados (ORDER, ConfidNet, MCP) apenas para o dataset de Teste.
    Layout: N linhas (Classes) x 3 Colunas (Métricas).
    """
    setup_plot_params()
    nrows = len(class_names)
    ncols = 3
    
    # Ajusta o tamanho da figura baseado no número de classes
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 2 * nrows), sharey=False)
    
    # Garante que ax seja indexável mesmo se houver apenas 1 classe
    if nrows == 1:
        ax = ax.reshape(1, -1)

    for n, crop_class in enumerate(class_names):
        # Filtra predições para a classe atual no dataset de teste
        sub_gdf = test_gdf[test_gdf.y_pred == crop_class].reset_index(drop=True)
        has_data = len(sub_gdf) > 0

        # ---------------------------------------------------------
        # COLUNA 1: ORDER (GMM Scores)
        # ---------------------------------------------------------
        current_ax = ax[n, 0]
        if has_data and "gmm_score" in sub_gdf.columns:
            # Predições Corretas (Verde)
            right = sub_gdf[sub_gdf.right_pred]
            if not right.empty:
                right.gmm_score.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            
            # Predições Incorretas (Vermelho)
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not wrong.empty:
                wrong.gmm_score.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)

            # Threshold do GMM
            if gmm_thresholds and crop_class in gmm_thresholds:
                current_ax.axvline(
                    x=gmm_thresholds[crop_class], 
                    color="blue", 
                    linestyle="--", 
                    label="Threshold"
                )
        
        # Formatação Coluna 1
        if n == 0:
            current_ax.set_title("ORDER", fontsize=14 * ESCALA_FONTE)
        
        # Label Y com o nome da classe
        current_ax.set_ylabel(crop_class, fontsize=14 * ESCALA_FONTE, rotation=90)
        current_ax.set_yscale('log')
        current_ax.set_yticks([]) # Remove ticks numéricos do Y para limpeza
        current_ax.set_xticks([]) # Remove ticks numéricos do Y para limpeza

        # ---------------------------------------------------------
        # COLUNA 2: ConfidNet
        # ---------------------------------------------------------
        current_ax = ax[n, 1]
        if has_data and "y_conf" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            if not right.empty:
                right.y_conf.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not wrong.empty:
                wrong.y_conf.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)
        
        # Formatação Coluna 2
        current_ax.set_xlim(0, 1)
        if n == 0:
            current_ax.set_title("ConfidNet", fontsize=14 * ESCALA_FONTE)

        current_ax.set_ylabel("")
        current_ax.set_yscale('log')
        current_ax.set_yticks([]) # Remove ticks numéricos do Y para limpeza
        current_ax.set_xticks([]) # Remove ticks numéricos do Y para limpeza

        # ---------------------------------------------------------
        # COLUNA 3: MCP (Probabilidade)
        # ---------------------------------------------------------
        current_ax = ax[n, 2]
        if has_data and "y_proba" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            if not right.empty:
                right.y_proba.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not wrong.empty:
                wrong.y_proba.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)
        
        # Formatação Coluna 3
        current_ax.set_xlim(0, 1)
        if n == 0:
            current_ax.set_title("MCP", fontsize=14 * ESCALA_FONTE)

        current_ax.set_ylabel("")
        current_ax.set_yscale('log')
        current_ax.set_yticks([]) # Remove ticks numéricos do Y para limpeza
        current_ax.set_xticks([]) # Remove ticks numéricos do Y para limpeza


    plt.tight_layout()
    plt.savefig(f"figs/{experiment_name}/{run_name}/hist_combined_test.pdf", bbox_inches="tight")
    plt.close()

def plot_combined_histograms_test(test_gdf, gmm_thresholds, class_names, run_name, experiment_name):
    """
    Plota histogramas combinados (ORDER, ConfidNet, MCP) apenas para o dataset de Teste.
    Layout: N linhas (Classes) x 3 Colunas (Métricas).
    """
    setup_plot_params()
    nrows = len(class_names)
    ncols = 3
    
    # Ajusta o tamanho da figura baseado no número de classes
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 2 * nrows), sharey=False)
    
    # Garante que ax seja indexável mesmo se houver apenas 1 classe
    if nrows == 1:
        ax = ax.reshape(1, -1)

    for n, crop_class in enumerate(class_names):
        # Filtra predições para a classe atual no dataset de teste
        sub_gdf = test_gdf[test_gdf.y_pred == crop_class].reset_index(drop=True)
        has_data = len(sub_gdf) > 0

        # ---------------------------------------------------------
        # COLUNA 1: ORDER (GMM Scores)
        # ---------------------------------------------------------
        current_ax = ax[n, 0]
        if has_data and "gmm_score" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            if not right.empty:
                right.gmm_score.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not wrong.empty:
                wrong.gmm_score.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)

            if gmm_thresholds and crop_class in gmm_thresholds:
                current_ax.axvline(
                    x=gmm_thresholds[crop_class], 
                    color="blue", 
                    linestyle="--", 
                    label="Threshold"
                )
        
        # Formatação Coluna 1
        if n == 0:
            current_ax.set_title("ORDER", fontsize=14 * ESCALA_FONTE)
        
        current_ax.set_ylabel(crop_class, fontsize=14 * ESCALA_FONTE, rotation=90)
        current_ax.set_yscale('log')
        
        # --- CORREÇÃO AQUI ---
        current_ax.minorticks_off()      # Desliga os ticks menores do log
        current_ax.set_yticks([])        # Remove os ticks principais
        current_ax.set_yticklabels([])   # Remove os números (texto)
        current_ax.set_xticks([])        
        # ---------------------

        # ---------------------------------------------------------
        # COLUNA 2: ConfidNet
        # ---------------------------------------------------------
        current_ax = ax[n, 1]
        if has_data and "y_conf" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            if not right.empty:
                right.y_conf.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not wrong.empty:
                wrong.y_conf.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)
        
        # Formatação Coluna 2
        current_ax.set_xlim(0, 1)
        if n == 0:
            current_ax.set_title("ConfidNet", fontsize=14 * ESCALA_FONTE)

        current_ax.set_ylabel("")
        current_ax.set_yscale('log')
        
        # --- CORREÇÃO AQUI ---
        current_ax.minorticks_off()
        current_ax.set_yticks([])
        current_ax.set_yticklabels([])
        current_ax.set_xticks([])
        # ---------------------

        # ---------------------------------------------------------
        # COLUNA 3: MCP (Probabilidade)
        # ---------------------------------------------------------
        current_ax = ax[n, 2]
        if has_data and "y_proba" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            if not right.empty:
                right.y_proba.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not wrong.empty:
                wrong.y_proba.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)
        
        # Formatação Coluna 3
        current_ax.set_xlim(0, 1)
        if n == 0:
            current_ax.set_title("MCP", fontsize=14 * ESCALA_FONTE)

        current_ax.set_ylabel("")
        current_ax.set_yscale('log')
        
        # --- CORREÇÃO AQUI ---
        current_ax.minorticks_off()
        current_ax.set_yticks([])
        current_ax.set_yticklabels([])
        current_ax.set_xticks([])
        # ---------------------


    plt.tight_layout()
    plt.savefig(f"figs/{experiment_name}/{run_name}/hist_combined_test.pdf", bbox_inches="tight")
    plt.close()

def plot_combined_histograms_test(test_gdf, gmm_thresholds, class_names, run_name, experiment_name):
    """
    Plota histogramas combinados (ORDER, ConfidNet, MCP) apenas para o dataset de Teste.
    Layout: N linhas (Classes) x 3 Colunas (Métricas).
    """
    setup_plot_params()
    nrows = len(class_names)
    ncols = 3
    
    # Ajusta o tamanho da figura baseado no número de classes
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 2 * nrows), sharey=False)
    
    # Garante que ax seja sempre uma matriz 2D [linhas, colunas]
    if nrows == 1:
        ax = ax.reshape(1, -1)
    elif nrows > 1 and ncols == 1: # Apenas segurança extra
        ax = ax.reshape(-1, 1)

    # --- 1. Lógica de Plotagem ---
    for n, crop_class in enumerate(class_names):
        sub_gdf = test_gdf[test_gdf.y_pred == crop_class].reset_index(drop=True)
        has_data = len(sub_gdf) > 0

        # Coluna 0: ORDER
        current_ax = ax[n, 0]
        if has_data and "gmm_score" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not right.empty:
                right.gmm_score.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            if not wrong.empty:
                wrong.gmm_score.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)
            if gmm_thresholds and crop_class in gmm_thresholds:
                current_ax.axvline(x=gmm_thresholds[crop_class], color="blue", linestyle="--")
        
        current_ax.set_title("ORDER", fontsize=14 * ESCALA_FONTE) if n == 0 else None
        current_ax.set_ylabel(crop_class, fontsize=14 * ESCALA_FONTE, rotation=90)
        current_ax.set_yscale('log')

        # Coluna 1: ConfidNet
        current_ax = ax[n, 1]
        if has_data and "y_conf" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not right.empty:
                right.y_conf.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            if not wrong.empty:
                wrong.y_conf.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)
        
        current_ax.set_title("ConfidNet", fontsize=14 * ESCALA_FONTE) if n == 0 else None
        current_ax.set_yscale('log')
        current_ax.set_xlim(0, 1)

        # Coluna 2: MCP
        current_ax = ax[n, 2]
        if has_data and "y_proba" in sub_gdf.columns:
            right = sub_gdf[sub_gdf.right_pred]
            wrong = sub_gdf[~sub_gdf.right_pred]
            if not right.empty:
                right.y_proba.plot.hist(color="green", ax=current_ax, alpha=1.0, bins=30)
            if not wrong.empty:
                wrong.y_proba.plot.hist(color="red", ax=current_ax, alpha=0.5, bins=30)
        
        current_ax.set_title("MCP", fontsize=14 * ESCALA_FONTE) if n == 0 else None
        current_ax.set_yscale('log')
        current_ax.set_xlim(0, 1)

    # --- 2. Limpeza Brutal (O Loop Duplo Solicitado) ---
    # Isso roda DEPOIS de todos os plots, garantindo que o Pandas não reative nada.
    for i in range(nrows):
        for j in range(ncols):
            current_ax = ax[i, j]
            
            # Remove Label do Eixo X (Texto "gmm_score", "y_conf", etc)
            current_ax.set_xlabel("")
            
            # Se não for a primeira coluna, remove o Label do Eixo Y também
            # (Mantemos a primeira coluna pois é o nome da Classe/Safra)
            if j != 0:
                current_ax.set_ylabel("")
            
            # A "Opção Nuclear" para remover ticks e números dos eixos
            current_ax.tick_params(
                axis='both',          # Aplica para X e Y
                which='both',         # Aplica para Major e Minor ticks (CRUCIAL para log)
                bottom=False,         # Remove tracinhos de baixo
                top=False,            # Remove tracinhos de cima
                left=False,           # Remove tracinhos da esquerda
                right=False,          # Remove tracinhos da direita
                labelbottom=False,    # Remove números de baixo
                labelleft=False       # Remove números da esquerda
            )
            
            # Redundância para garantir
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            current_ax.minorticks_off()

    plt.tight_layout()
    plt.savefig(f"figs/{experiment_name}/{run_name}/hist_combined_test.pdf", bbox_inches="tight")
    plt.close()

def process_run(runs_df, run_name, experiment_name):
    run_dir = f"figs/{experiment_name}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    train_gdf, val_gdf, test_gdf, gmm_thresholds = load_data_for_run(runs_df, run_name)

    train_gdf = add_gmm_pred_column(train_gdf)
    val_gdf = add_gmm_pred_column(val_gdf)
    test_gdf = add_gmm_pred_column(test_gdf)

    train_gdf = remap_crop_names(train_gdf)
    val_gdf = remap_crop_names(val_gdf)
    test_gdf = remap_crop_names(test_gdf)

    class_names = sorted(test_gdf.y_true.unique().tolist())
    x_columns = get_x_columns(test_gdf)

    tasks = [
        delayed(plot_classification_results)(
            train_gdf.y_true,
            train_gdf.y_pred,
            f"{run_dir}/class_train.pdf",
        ),
        delayed(plot_classification_results)(
            val_gdf.y_true,
            val_gdf.y_pred,
            f"{run_dir}/class_val.pdf",
        ),
        delayed(plot_classification_results)(
            test_gdf.y_true,
            test_gdf.y_pred,
            f"{run_dir}/class_test.pdf",
        ),
    ]

    if "gmm_pred" in train_gdf.columns:
        tasks.extend([
            delayed(plot_classification_results_gemmos)(
                train_gdf.y_true,
                train_gdf.gmm_pred,
                f"{run_dir}/class_train_gemmos.pdf",
            ),
            delayed(plot_classification_results_gemmos)(
                val_gdf.y_true,
                val_gdf.gmm_pred,
                f"{run_dir}/class_val_gemmos.pdf",
            ),
            delayed(plot_classification_results_gemmos)(
                test_gdf.y_true,
                test_gdf.gmm_pred,
                f"{run_dir}/class_test_gemmos.pdf",
            )
        ])

    # --- Nova chamada combinada substituindo as chamadas individuais anteriores ---
    tasks.append(
        delayed(plot_combined_histograms_test)(
            test_gdf, gmm_thresholds, class_names, run_name, experiment_name
        )
    )
    # ----------------------------------------------------------------------------

    #tasks.append(delayed(plot_embeddings_gemmos)(test_gdf, x_columns, run_name, experiment_name))

    Parallel(n_jobs=-1)(tasks)

    print(f"Completed run: {run_name}")


def main():
    os.makedirs("figs", exist_ok=True)

    for experiment_name, runn_name in [("brazil-finetuning", "BERTPP-70.0-MoCo"), ("california-finetuning", "MAMBA-70.0-reconstruct"), ("texas-finetuning", "LSTM-70.0-FastSiam")]: #, "texas-finetuning", "california-finetuning"]:
        runs_df = mlflow.search_runs(experiment_names=[experiment_name])
        runs_df = (
            runs_df.sort_values("start_time", ascending=False)
            .drop_duplicates(subset=["tags.mlflow.runName"])
            .reset_index(drop=True)
        )
        runs_df = runs_df.set_index("tags.mlflow.runName")

        run_names = sorted(runs_df.index.tolist())
        
        
        # Check for existing figs to skip
        run_names = [rn for rn in run_names if not os.path.exists(f"figs/{experiment_name}/{rn}/tsne_embeddings.pdf")]
        run_names = [runn_name]
        for run_name in tqdm(run_names):
            try:
                process_run(runs_df, run_name, experiment_name)
            except Exception as e:
                print(f"Error processing {run_name}: {e}")


if __name__ == "__main__":
    main()