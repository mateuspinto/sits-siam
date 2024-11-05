import torch.nn.functional as F
import torch.nn as nn
import torch
import copy


class Attention(nn.Module):
    def forward(self, X, mask=None):
        d_k = X.size(-1)
        scores = torch.matmul(X, X.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=X.dtype)
        )
        scores = scores.masked_fill(mask, float("-inf"))
        p_attn = F.softmax(scores, dim=-1)
        output = torch.matmul(p_attn, X)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, batch_first=True):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, X, src_key_padding_mask):
        batch_size, seq_len, _ = X.size()
        X = self.linear(X)
        X = X.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        mask = (
            src_key_padding_mask.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, self.num_heads, -1, -1)
        )
        X = self.attention(X, mask)
        X = (
            X.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.d_k)
        )
        X = self.output_linear(X)

        return X


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, X, src_mask=None, src_key_padding_mask=None):
        X2 = self.self_attn(X, src_key_padding_mask=src_key_padding_mask)
        X = X + self.dropout(X2)
        X = self.norm1(X)
        X2 = self.linear2(self.dropout(self.activation(self.linear1(X))))
        X = X + self.dropout(X2)
        X = self.norm2(X)
        return X


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, X, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            X = layer(X, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        X = self.norm(X)
        return X
