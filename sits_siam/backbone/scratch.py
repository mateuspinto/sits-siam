import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention' without using the math library.
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # Compute scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=query.dtype, device=query.device)
        )

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        output = torch.matmul(p_attn, value)
        return output, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Headed Attention module without using built-in attention modules.
    Supports parameters: d_model, num_heads, dropout, batch_first.
    """

    def __init__(self, d_model, num_heads, dropout=0.1, batch_first=True):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = query.size()

        # Linear projections
        query, key, value = [
            linear(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]  # Each tensor is of shape (batch_size, num_heads, seq_len, d_k)

        # Prepare masks
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len)
            # Expand to (batch_size, 1, 1, seq_len)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # (batch_size, 1, 1, seq_len)
            # Expand to match the number of heads
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)
        if attn_mask is not None:
            # attn_mask: (seq_len, seq_len)
            attn_mask = attn_mask.unsqueeze(0)  # (1, seq_len, seq_len)
            attn_mask = attn_mask.expand(batch_size * self.num_heads, -1, -1).view(
                batch_size, self.num_heads, seq_len, seq_len
            )
        # Combine masks
        if key_padding_mask is not None and attn_mask is not None:
            combined_mask = key_padding_mask | attn_mask
        elif key_padding_mask is not None:
            combined_mask = key_padding_mask
        elif attn_mask is not None:
            combined_mask = attn_mask
        else:
            combined_mask = None

        # Apply attention
        x, attn = self.attention(
            query, key, value, mask=combined_mask, dropout=self.dropout
        )

        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        x = self.output_linear(x)

        return x


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
        self.self_attn = MultiHeadedAttention(
            d_model,
            num_heads,
            dropout=dropout,
        )
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation_fn = F.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        attn_output = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward network
        ff_output = self.linear2(self.dropout(self.activation_fn(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        output = self.norm(output)

        return output
