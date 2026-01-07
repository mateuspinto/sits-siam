import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions):
        return self.pe[positions]


class SITSBertPlusPlus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features=10,
        hidden=768,
        n_layers=3,
        n_heads=12,
        dropout=0.1,
        max_len=366,
        extra_attn_heads=12,
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
                activation=nn.SiLU(),
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_layers,
        )

        self.pool_layer = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        self.rec_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=extra_attn_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.rec_norm = nn.LayerNorm(hidden)

        self.rec_features = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.rec_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_features),
        )

        self.example_input_array = (
            torch.randn(1, max_len, num_features),
            torch.randint(0, 366, (1, max_len)),
            torch.zeros(1, max_len).bool(),
        )

    def forward(self, x, doy, mask):
        feat_emb = self.input_proj(x)
        pos_emb = self.pos_encoder(doy)
        embed = torch.cat([feat_emb, pos_emb], dim=-1)

        x_enc = self.transformer(embed, src_key_padding_mask=mask)
        pooled = self.pool_layer(x_enc.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier(pooled)
        attn_out, _ = self.rec_attn(x_enc, x_enc, x_enc, key_padding_mask=mask)
        x_rec = self.rec_norm(x_enc + attn_out)

        x_rec = self.rec_features(x_rec)
        reconstruction = self.rec_head(x_rec)

        return pooled, logits, reconstruction


class SITSBert(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features=10,
        hidden=256,
        n_layers=3,
        n_heads=8,
        dropout=0.1,
        max_len=366,
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
                activation="gelu",
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.pool_layer = nn.AdaptiveMaxPool1d(1)
        self.reconstruction_layer = nn.Linear(hidden, num_features)
        self.classifier = nn.Linear(hidden, num_classes)
        self.example_input_array = (
            torch.randn(1, 120, num_features),  # x: (Batch, Time, Feats)
            torch.randint(0, 366, (1, 120)),  # doy: (Batch, Time)
            torch.zeros(1, 120).bool(),  # mask: (Batch, Time)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, doy, mask):
        feat_emb = self.input_proj(x)
        pos_emb = self.pos_encoder(doy)
        embed = torch.cat([feat_emb, pos_emb], dim=-1)
        src_key_padding_mask = mask
        x_enc = self.transformer(embed, src_key_padding_mask=src_key_padding_mask)

        pooled = self.pool_layer(x_enc.permute(0, 2, 1)).squeeze(-1)

        logits = self.classifier(pooled)

        reconstructed = self.reconstruction_layer(x_enc)

        return pooled, logits, reconstructed


class SITS_MLP_Backbone(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        time_steps=12,
        hidden_dim=256,
        dropout_rate=0.3,
    ):
        super().__init__()

        input_dim = input_channels * time_steps

        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, doy=None, mask=None):
        x = self.flatten(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        last_emb = self.layer3(out2)

        logits = self.classifier(last_emb)

        # all_embs = torch.cat([out1, out2, last_emb], dim=1)

        return logits, last_emb, last_emb


class SITS_LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features=10,
        hidden=256,
        n_layers=3,
        dropout=0.1,
        max_len=366,
    ):
        super().__init__()

        self.embed_dim = hidden // 2
        self.input_proj = nn.Linear(num_features, self.embed_dim)

        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len)

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden, num_classes)
        self.reconstruction_layer = nn.Linear(hidden, num_features)

        self.example_input_array = (
            torch.randn(1, 120, num_features),
            torch.randint(0, 366, (1, 120)),
            torch.zeros(1, 120).bool(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "lstm" in name and "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0)
            elif isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)

    def forward(self, x, doy, mask):
        feat_emb = self.input_proj(x)
        pos_emb = self.pos_encoder(doy)

        embed = torch.cat([feat_emb, pos_emb], dim=-1)

        lengths = (~mask).sum(dim=1).cpu().int()
        lengths = torch.clamp(lengths, min=1)

        packed_input = pack_padded_sequence(
            embed, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        lstm_out, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )

        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        reconstructed = self.reconstruction_layer(lstm_out)

        mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
        lstm_out_masked = lstm_out.clone()
        lstm_out_masked[mask_expanded] = -float("inf")

        pooled, _ = torch.max(lstm_out_masked, dim=1)

        logits = self.classifier(pooled)

        return pooled, logits, reconstructed


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class SITSConvNext(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features=10,
        hidden=256,
        n_layers=3,
        dropout=0.1,
        max_len=366,
    ):
        super().__init__()

        self.embed_dim = hidden // 2
        self.input_proj = nn.Linear(num_features, self.embed_dim)

        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len)

        self.stem = nn.Sequential(
            nn.Conv2d(
                hidden, hidden, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)
            ),
            LayerNorm2d(hidden, eps=1e-6),
        )

        dp_rates = [x.item() for x in torch.linspace(0, dropout, n_layers)]
        self.blocks = nn.Sequential(
            *[Block(dim=hidden, drop_path=dp_rates[i]) for i in range(n_layers)]
        )

        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.classifier = nn.Linear(hidden, num_classes)
        self.reconstruction_layer = nn.Linear(hidden, num_features)

        self.example_input_array = (
            torch.randn(1, 120, num_features),
            torch.randint(0, 366, (1, 120)),
            torch.zeros(1, 120).bool(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, doy, mask):
        feat_emb = self.input_proj(x)
        pos_emb = self.pos_encoder(doy)

        x_emb = torch.cat([feat_emb, pos_emb], dim=-1)

        mask_expanded = mask.unsqueeze(-1).expand_as(x_emb)
        x_emb[mask_expanded] = 0.0

        x_2d = x_emb.permute(0, 2, 1).unsqueeze(-1)

        x_2d = self.stem(x_2d)
        x_2d = self.blocks(x_2d)

        x_out = x_2d.squeeze(-1).permute(0, 2, 1)
        x_out = self.norm(x_out)

        reconstructed = self.reconstruction_layer(x_out)

        mask_expanded_pool = mask.unsqueeze(-1).expand_as(x_out)
        x_pool = x_out.clone()
        x_pool[mask_expanded_pool] = -float("inf")
        pooled, _ = torch.max(x_pool, dim=1)

        logits = self.classifier(pooled)

        return pooled, logits, reconstructed


if __name__ == "__main__":
    model = SITS_LSTM(num_classes=10)
    pooled, logits, reconstructed = model(*model.example_input_array)
    print("Pooled shape:", pooled.shape)
    print("Logits shape:", logits.shape)
    print("Reconstruction shape:", reconstructed.shape)
