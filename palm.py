import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PaLM(nn.Module):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, eps: float, dropout_rate: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.mask_generator = MaskGenerator()
        self.decoder = Decoder(n=n, d_model=d_model, heads=heads, eps=eps, dropout_rate=dropout_rate, bias=bias)
        self.classifier = nn.Linear(in_features=d_model, out_features=token_size, bias=bias)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor):
        padding_mask = None
        mask = None

        if self.training:
            mask, padding_mask = self.mask_generator(x)
        
        x = self.embedding(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.decoder(x, mask)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.classifier(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.training:
            return x, padding_mask
        
        return x

class Decoder(nn.Module):
    def __init__(self, n: int, d_model: int, heads: int, eps: float, dropout_rate: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, heads=heads, eps=eps, dropout_rate=dropout_rate, bias=bias) for _ in range(n)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, eps: float, dropout_rate: float, bias: bool = False) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(heads=heads, d_model=d_model, dropout_rate=dropout_rate, bias=bias)
        self.mlp = PositionWiseFeedForwardNetworks(d_ff=4 * d_model, d_model=d_model, dropout_rate=dropout_rate, bias=bias)

        self.pre_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x_norm = self.pre_norm(x)
        
        # sub - layer 1
        attention_output = self.attention(x_norm, x_norm, x_norm, mask)
        attention_output = F.dropout(attention_output, p=self.dropout_rate, training=self.training)
        # sub - layer 2
        mlp_output = self.mlp(x_norm)
        mlp_output = F.dropout(mlp_output, p=self.dropout_rate, training=self.training)
        
        x = self.layer_norm(x + attention_output + mlp_output)
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_rate: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.heads = heads
        self.d_model = d_model

        self.head_samples = self.d_model // self.heads

        self.rotary_embedding = RotaryPostionEmbedding(dim=self.head_samples)

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

        self.dropout_rate = dropout_rate

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        dk = torch.tensor(k.size(-1))

        
        q, k = self.rotary_embedding(q, k)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores/(torch.sqrt(dk))

        if mask is not None:
            attention_scores += mask*(-1e15)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_context = torch.matmul(attention_weights, v)

        return attention_context
    
    def get_rotary_embedding(self, n: int):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_embedding(n)

        self.pos_emb = pos_emb

        return pos_emb

    def split_head(self, x: torch.Tensor):
        batch_size, n_ctx, _ = x.size()

        x = x.reshape((batch_size, n_ctx, self.heads, self.head_samples))
        x = x.permute((0, 2, 1, 3))
        
        return x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, n_ctx, _ = q.size()
        
        qw = F.dropout(self.linear_q(q), p=self.dropout_rate, training=self.training)
        kw = F.dropout(self.linear_k(k), p=self.dropout_rate, training=self.training)
        vw = F.dropout(self.linear_v(v), p=self.dropout_rate, training=self.training)

        q_heads = self.split_head(qw)
        k_heads = self.split_head(kw)
        v_heads = self.split_head(vw)

        attention_context = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention_context = attention_context.permute((0, 2, 1, 3))
        attention_context = attention_context.reshape((batch_size, n_ctx, self.d_model))

        attention_context = self.linear_output(attention_context)
        attention_context = F.dropout(attention_context, p=self.dropout_rate, training=self.training)
        
        return attention_context

class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self, d_ff: int, d_model: int, dropout_rate: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=d_model, out_features=d_ff, bias=bias)
        self.activation = F.silu
        self.output_layer = nn.Linear(in_features=d_ff, out_features=d_model, bias=bias)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

class RotaryPostionEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

        self.seq_len_cached = None
        self.cos_embedding = None
        self.sin_embedding = None

    def get_rotary(self, length: int, device: str):
        if length != self.seq_len_cached:
            self.seq_len_cached = length

            inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).to(device).float() / self.dim))
            t = torch.arange(length, device=device, dtype=inv_freq.dtype)

            freqs = torch.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            self.cos_embedding = emb.cos()
            self.sin_embedding = emb.sin()
    
    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (self.rotate_half(k) * sin)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        assert q.size(2) == k.size(2)

        self.get_rotary(q.size(2), device=q.device)

        return self.apply_rotary_pos_emb(q, k, self.cos_embedding, self.sin_embedding)

class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(chunks=2, dim=-1)
        return F.silu(gate) * x
    
class MaskGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def generate_padding_mask(self, tensor: torch.Tensor)-> torch.Tensor:
        return torch.Tensor(tensor != 0)

    def __generate_look_ahead_mask(self, length: int) -> torch.Tensor:
        return torch.triu(torch.ones((length, length)), diagonal=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        padding_mask = self.generate_padding_mask(tensor)

        look_ahead_mask = self.__generate_look_ahead_mask(tensor.size(1)).to(tensor.device)

        look_ahead_mask = torch.maximum(look_ahead_mask, ~(padding_mask[:, None, None, :]).type(torch.int8))

        return look_ahead_mask, padding_mask