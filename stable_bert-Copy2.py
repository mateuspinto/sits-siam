import math
import os
import random
import sys
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report, brier_score_loss
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)
from tqdm.std import tqdm

from sits_siam.augment import (
    AddMissingMask,
    IncreaseSequenceLength,
    LimitSequenceLength,
    Pipeline,
    ToPytorchTensor,
    Normalize,
)
from sits_siam.utils import AgriGEELiteDataset, SitsFinetuneDatasetFromNpz
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, positions):
        return self.pe[positions]

class SITSBertClassifier(nn.Module):
    def __init__(
        self,
        num_features=10,
        num_classes=13, # not used in pre-training
        seq_len=64,
        hidden=256,
        n_layers=3,
        n_heads=8,
        dropout=0.1,
        max_len=366
    ):
        super().__init__()
        self.embed_dim = hidden // 2
        self.input_proj = nn.Linear(num_features, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=n_layers
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, doy, mask):
        feat_emb = self.input_proj(x)
        pos_emb = self.pos_encoder(doy)
        embed = torch.cat([feat_emb, pos_emb], dim=-1)
        src_key_padding_mask = mask
        x_enc = self.transformer(embed, src_key_padding_mask=src_key_padding_mask)
        # Pooling foi movido para fora para que o KNN callback possa usar o backbone diretamente
        return x_enc
        
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


setup_seed()
beautify_prints()

transforms = Pipeline(
    [
        LimitSequenceLength(140),
        IncreaseSequenceLength(140),
        AddMissingMask(),
        Normalize(),
        ToPytorchTensor(),
    ]
)

#train_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/train.npz", transform=transforms)
#val_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/val.npz", transform=transforms)
#test_dataset = SitsFinetuneDatasetFromNpz("data/texas_70_15_15/test.npz", transform=transforms)

import pandas as pd
# Não é estritamente necessário importar geopandas ou numpy para esta lógica,
# mas é bom ter numpy se for usar manipulação numérica ou semente aleatória.
# import numpy as np 
from sklearn.model_selection import train_test_split

gdf = gpd.read_parquet("data/agl/gdf.parquet")

# 1. Obter a lista de códigos de municípios únicos
unique_mun = gdf['CD_MUN'].unique()

# 2. Dividir os municípios em Treino (70%) e Temporário (30%)
# Use random_state para garantir que o resultado da divisão seja o mesmo toda vez.
mun_train, mun_temp = train_test_split(
    unique_mun,
    test_size=0.30,  # 30% restante para Validação + Teste
    random_state=42
)

# 3. Dividir os municípios Temporários (30%) em Validação (15%) e Teste (15%)
# 0.50 dos 30% restantes é igual a 15% do total.
mun_val, mun_test = train_test_split(
    mun_temp,
    test_size=0.50,  # 50% dos 30% restantes
    random_state=42
)

# 4. Criar os GeoDataFrames finais usando as listas de municípios
# GeoDataFrame de Treino
gdf_train = gdf[gdf['CD_MUN'].isin(mun_train)].copy()
# GeoDataFrame de Validação
gdf_val = gdf[gdf['CD_MUN'].isin(mun_val)].copy()
# GeoDataFrame de Teste
gdf_test = gdf[gdf['CD_MUN'].isin(mun_test)].copy()

# Opcional: imprimir a verificação
print(f"Municípios Treino: {len(mun_train)} - {len(gdf_train)} linhas")
print(f"Municípios Validação: {len(mun_val)} - {len(gdf_val)} linhas")
print(f"Municípios Teste: {len(mun_test)} - {len(gdf_test)} linhas")

train_dataset = AgriGEELiteDataset(
    gdf_train, "data/agl/df_sits.parquet", transform=transforms
)

val_dataset = AgriGEELiteDataset(
    gdf_val, "data/agl/df_sits.parquet", transform=transforms
)

test_dataset = AgriGEELiteDataset(
    gdf_test, "data/agl/df_sits.parquet", transform=transforms
)


class OriginalSITSBertTrainer(pl.LightningModule):
    def __init__(self, max_seq_len=127, num_classes=int(train_dataset.num_classes), 
                 max_epochs=100, batch_size=8*512, train_dataset_size=None,
                 num_warmup_epochs=10, base_lr=1e-6, max_lr=1e-3, k=4, gamma=0.5, lambda_conf=0.7):
        super(OriginalSITSBertTrainer, self).__init__()
        self.save_hyperparameters() # Saves hyperparameters for easy loading

        # --- Main Model Components ---
        self.backbone = SITSBertClassifier()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier_layer = nn.Linear(256, num_classes)

        # --- Confidence Branch (ConfidNet) ---
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid() # confidence values in [0, 1]
        )

        # --- Loss Functions ---
        self.classification_criterion = nn.CrossEntropyLoss(weight=train_dataset.get_class_weights())
        self.confidence_criterion = nn.MSELoss()

        # --- Metrics ---
        self.train_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        
        # Parâmetros de treinamento
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size
        self.num_warmup_epochs = num_warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.k = k  # Número de divisões do treinamento para redução de LR
        self.gamma = gamma  # Fator de redução do LR

    def forward(self, input_batch):
        x, doy, mask = input_batch["x"], input_batch["doy"], input_batch["mask"]

        # Get pooled features from the backbone
        backbone_features = self.backbone(x, doy, mask)
        pooled = self.pool(backbone_features.permute(0, 2, 1)).squeeze(-1) # [B, hidden]

        # Main classification task
        logits = self.classifier_layer(pooled)

        # Confidence prediction task
        confidence = self.confidence_head(pooled)

        return logits, confidence

    def _calculate_losses(self, logits, confidence, targets):
        # 1. Classification Loss (Cross Entropy)
        class_loss = self.classification_criterion(logits, targets)

        # 2. Confidence Loss (MSE)
        # Detach logits so that confidence loss doesn't flow back to the main classifier
        probs = F.softmax(logits.detach(), dim=1)
        # Get the probability assigned to the true class (TCP)
        tcp_target = probs.gather(1, targets.unsqueeze(1))
        conf_loss = self.confidence_criterion(confidence.squeeze(), tcp_target.squeeze())

        return class_loss, conf_loss
        
    def training_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        logits, confidence = self(batch)

        class_loss, conf_loss = self._calculate_losses(logits, confidence, targets)
        total_loss = class_loss + self.hparams.lambda_conf * conf_loss

        # --- Logging ---
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_class_loss", class_loss, prog_bar=True)
        self.log("train_conf_loss", conf_loss, prog_bar=True)
        self.log("train_oa", self.train_oa(logits, targets), prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        logits, confidence = self(batch)

        class_loss, conf_loss = self._calculate_losses(logits, confidence, targets)
        total_loss = class_loss + self.hparams.lambda_conf * conf_loss

        # --- Logging ---
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_class_loss", class_loss, prog_bar=True, sync_dist=True)
        self.log("val_conf_loss", conf_loss, prog_bar=True, sync_dist=True)
        self.log("val_oa", self.val_oa(logits, targets), prog_bar=True, sync_dist=True)

        return total_loss

    def on_train_epoch_end(self):
        # Pega o valor da LR do primeiro param group
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        logits, confidence = self(batch)

        test_oa_score = self.test_oa(logits, targets)
        self.test_cm(logits, targets)
        self.log("test_oa", test_oa_score, sync_dist=True)

    def on_test_epoch_end(self):
        self.confusion_matrix = self.test_cm.compute()
        self.test_cm.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        train_steps_per_epoch = math.ceil(self.train_dataset_size / self.batch_size)
        total_training_steps = train_steps_per_epoch * self.max_epochs
        num_warmup_steps = train_steps_per_epoch * self.num_warmup_epochs
        
        # define warmup parameters equivalentes ao código anterior
        warmup_factor = 1.0 / 1000        # começa com 1/1000 do LR
        warmup_iters = min(1000, num_warmup_steps - 1)  # igual ao n_iters - 1
    
        def lr_lambda(current_step):
            if current_step < warmup_iters:
                # Interpola linearmente de warmup_factor até 1.0
                alpha = current_step / float(max(1, warmup_iters))
                return warmup_factor * (1 - alpha) + alpha * 1.0
            # Após o warmup, mantém LR constante (1.0 × max_lr)
            return 1.0
    
        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }]

batch_size = 8 * 512
max_epochs = 200 #200
num_warmup_epochs = 20 #20

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True
)

mlflow_logger = MLFlowLogger(experiment_name="texas")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)

  
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu",
    precision="16-mixed",
    # devices=[0, 1],
    callbacks=[checkpoint_callback],
    logger=mlflow_logger
)

model = OriginalSITSBertTrainer(
    max_epochs=max_epochs,
    batch_size=batch_size,
    train_dataset_size=len(train_dataset),
    num_warmup_epochs=num_warmup_epochs,
    k=4,  # Divide o treinamento em 4 partes
    gamma=0.5  # Reduz LR pela metade a cada 25%
)

model.backbone.load_state_dict(torch.load("siam_texas_new_bert.pth"))
    
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)


from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os

def print_pretty_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Imprime uma matriz de confusão formatada para um terminal monoespaçado.
    
    Inclui cabeçalhos verticais para as classes previstas, garantindo
    alinhamento com base na largura máxima dos números na matriz.
    
    Argumentos:
        y_true (list/array): Lista de rótulos verdadeiros.
        y_pred (list/array): Lista de rótulos previstos.
        class_names (list, opcional): Lista ordenada de nomes de classes. 
                                     Se None, será inferida dos dados.
    """
    
    # Se os nomes das classes não forem fornecidos, infira-os dos dados
    if class_names is None:
        class_names = sorted(list(set(y_true) | set(y_pred)))
    
    # Calcule a matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # --- Calcular larguras de formatação ---
    
    # 1. Largura das células de números (baseada no maior número)
    #    Garante que a coluna tenha pelo menos 1 de largura.
    max_cell_width = max(len(str(np.max(cm))), 1)
    
    # 2. Largura dos rótulos das linhas (Classes Verdadeiras)
    max_label_width = max(len(name) for name in class_names)
    
    # 3. Altura dos cabeçalhos das colunas (Classes Previstas)
    max_header_height = max(len(name) for name in class_names)

    # --- Comece a construir a string de saída ---
    output = []
    
    # Largura do preenchimento esquerdo (para rótulos + separador " | ")
    # O preenchimento deve acomodar o rótulo da linha mais longo ou "True"
    true_label_pad = max(max_label_width, 4) # 4 para "True"
    left_pad_width = true_label_pad + 3 # "Label | "
    left_padding_str = " " * left_pad_width
    
    # --- 1. Criar Cabeçalhos Verticais "Predicted" ---
    
    # Calcula a largura total do conteúdo da matriz (números + espaços)
    matrix_content_width = len(class_names) * (max_cell_width + 2) - 2 # N*largura + (N-1)*espaços
    
    # Rótulo "Predicted" centralizado acima da matriz
    output.append(f"{left_padding_str}{'Predicted':^{matrix_content_width}}")
    
    # Cada linha de letras verticais
    for i in range(max_header_height):
        row_str = left_padding_str
        cells = []
        for name in class_names:
            # Pega a letra ou um espaço se o nome for mais curto
            letter = name[i] if i < len(name) else " "
            # Centraliza cada letra dentro da largura da coluna
            cells.append(f"{letter:^{max_cell_width}}")
        
        # Junta as células com um separador de 2 espaços
        row_str += "  ".join(cells)
        output.append(row_str)

    # --- 2. Criar Separador ---
    separator = "-" * matrix_content_width
    # Adiciona o rótulo "True" à esquerda do separador
    output.append(f"{'True':>{true_label_pad}} | {separator}")

    # --- 3. Criar Linhas da Matriz (Rótulos Verdadeiros + Números) ---
    for i, row in enumerate(cm):
        row_label = class_names[i]
        # Alinha o rótulo da classe verdadeira à direita
        row_str = f"{row_label:>{true_label_pad}} |"
        
        cells = []
        for val in row:
            # Alinha o número à direita para corresponder à formatação
            cells.append(f"{val:>{max_cell_width}}")
        
        # Junta os números com um separador de 2 espaços
        row_str += "  ".join(cells)
        output.append(row_str)

    # Imprime o resultado final
    print("\n".join(output))
    
def predict_and_save_predictions(
    model, dataloader, device="cuda"
):
    """
    Runs inference and calculates classification metrics, including Brier scores
    for both the main model and the confidence head.
    """
    model.eval()
    model.to(device)

    all_true = []
    all_preds = []
    all_proba = []
    all_conf = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Predicting"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            logits, confidence = model(batch)

            preds = torch.argmax(logits, dim=1)
            proba = F.softmax(logits, dim=1)

            all_true.append(batch["y"].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_proba.append(proba.cpu().numpy())
            all_conf.append(confidence.cpu().numpy())

    # Concatenate all batch results into single numpy arrays
    y_true_int = np.concatenate(all_true, axis=0).squeeze()
    y_pred_int = np.concatenate(all_preds, axis=0)
    y_proba = np.concatenate(all_proba, axis=0)
    y_conf = np.concatenate(all_conf, axis=0).squeeze()

    # --- Brier Score Calculation ---

    # 1. Brier Score for the standard softmax output
    # One-hot encode the true labels to compare with softmax probabilities
    classes = np.unique(y_true_int)
    y_true_one_hot = label_binarize(y_true_int, classes=classes)
    
    # For multi-class, calculate the mean of the squared error across all classes
    brier_softmax = np.mean(np.sum((y_proba - y_true_one_hot)**2, axis=1))

    # 2. Brier Score for the Confidence Head
    # The "ground truth" for confidence is whether the prediction was correct (1) or not (0)
    is_prediction_correct = (y_pred_int == y_true_int).astype(int)
    brier_confidence = brier_score_loss(is_prediction_correct, y_conf)

    print("\n--- Calibration Metrics ---")
    print(f"Brier Score (Softmax Output): {brier_softmax:.4f}")
    print(f"Brier Score (Confidence Head): {brier_confidence:.4f}")


    # --- Classification Report ---
    class_names = train_dataset.get_class_names()
    class_map = {i: name for i, name in enumerate(class_names)}
    y_true_names = np.array([class_map[num] for num in y_true_int])
    y_pred_names = np.array([class_map[num] for num in y_pred_int])

    print("\n--- Classification Report ---")
    report = classification_report(y_true_names, y_pred_names)
    print(report)

    print("\n--- Confusion Matrix ---")
    print_pretty_confusion_matrix(y_true_names, y_pred_names)

    # Save results including confidence to a file
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "results_with_conf.npz")
        np.savez(
            file_path,
            y_true=y_true_names,
            y_pred=y_pred_names,
            y_proba=y_proba,
            y_conf=y_conf
        )
        print(f"\nResults saved and logged to MLFlow artifact: results_with_conf.npz")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, file_path)

trainer.validate(model=model, dataloaders=val_dataloader)
model = OriginalSITSBertTrainer.load_from_checkpoint(checkpoint_callback.best_model_path)
model = model.eval()

predict_and_save_predictions(model, test_dataloader)
