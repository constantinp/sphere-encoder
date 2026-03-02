# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
basic functions
"""


def vector_rms_norm(z, zero_mean=False, eps=1e-6):
    assert z.ndim in [3, 4]  # [B, C, H, W] or [B, N, D]
    dim = tuple(range(1, z.ndim))  # w/o batch dimension
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    m = z.square().mean(dim=dim, keepdim=True)
    m = torch.rsqrt(m + eps).to(z.dtype)
    return z * m


@torch.no_grad()
def stratified_unit_radii(
    size, shuffle=True, including_zero=True, device="cuda", dtype=torch.float32
):
    N = size[0]
    M = N - 2 if including_zero else N - 1
    i = torch.arange(M, device=device, dtype=torch.float32)
    v = torch.rand(M, device=device)
    v = torch.clamp(v, min=1e-3)
    v = (i + v) / M
    w = [1.0, 0.0] if including_zero else [1.0]
    v = torch.cat([v, torch.tensor(w, device=device)])
    if shuffle:
        v = v[torch.randperm(N)]
    v = v.reshape(-1, *[1] * len(size[1:])).to(dtype=dtype)
    return v


@torch.no_grad()
def beta_radii(size, loc=0.0, scale=1.0, shuffle=True, device="cuda"):
    N = size[0]
    v = torch.normal(mean=loc, std=scale, size=(N - 2,), device=device)
    v = torch.sigmoid(v).clamp(min=0.0, max=1.0)
    v = torch.cat([v, torch.tensor([1.0, 0.0], device=device)])
    if shuffle:
        v = v[torch.randperm(N)]
    v = v.reshape(-1, *[1] * len(size[1:]))
    return v


@torch.no_grad()
def shift_range(x, a, b):
    assert a < b
    return a + x * (b - a)


"""
functions and modules for models
"""


def modulate(x, shift=None, scale=None):
    if shift is None and scale is None:
        return x
    if x.ndim == shift.ndim:
        return x * (1 + scale) + shift
    elif x.ndim == shift.ndim + 1:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        raise ValueError(
            f"shift shape {shift.shape} and x shape {x.shape} are not compatible"
        )


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size [M,]
    out: [M, D]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [D/2,]

    pos = pos.reshape(-1)  # [M,]
    out = np.einsum("m,d->md", pos, omega)  # [M, D/2], outer product

    embed_sin = np.sin(out)  # [M, D/2]
    embed_cos = np.cos(out)  # [M, D/2]

    embed = np.concatenate([embed_sin, embed_cos], axis=1)  # [M, D]
    return embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    embed_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]
    embed_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]
    return np.concatenate([embed_h, embed_w], axis=1)  # [H*W, D]


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width

    returns:
        pos_embed: [grid_size*grid_size, embed_dim]
                or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_rope_tensor(dim, seq_h, seq_w, pad_size=0, theta=10000.0):
    """
    2D axial rope tensor: [pad_size + (seq_h * seq_w), dim * 2] (cos, sin)
    """
    assert (
        dim % 4 == 0
    ), "dim must be divisible by 4 for 2D RoPE (div 2 for axial, div 2 for cos/sin)"
    half_dim = dim // 2

    # create freqs for half the dim
    # 1 / (theta ^ ( 2i / (D/2) ))
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))

    # repeat freqs to interleave rotations
    # [f1, f2] -> [f1, f1, f2, f2] for rotate_half
    freqs = einops.repeat(freqs, "n -> (n r)", r=2)  # [D/2]

    # create 2D grid from seq_len
    # t_h: [0, 1, ... H-1]
    # t_w: [0, 1, ... W-1]
    t_h = torch.arange(seq_h).float()
    t_w = torch.arange(seq_w).float()

    # freqs x coords
    freqs_h = torch.outer(t_h, freqs)  # [H, D/2]
    freqs_w = torch.outer(t_w, freqs)  # [W, D/2]

    # broadcast and concat for 2D axial
    freqs_h = einops.repeat(freqs_h, "h d -> h w d", w=seq_w)
    freqs_w = einops.repeat(freqs_w, "w d -> h w d", h=seq_h)

    freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1)  # [H, W, D]
    freqs_2d = freqs_2d.view(-1, dim)  # [H*W, D]

    cos_img = freqs_2d.cos()
    sin_img = freqs_2d.sin()

    if pad_size > 0:
        ones = torch.ones(pad_size, dim, device=cos_img.device, dtype=cos_img.dtype)
        zeros = torch.zeros(pad_size, dim, device=sin_img.device, dtype=sin_img.dtype)

        cos_img = torch.cat([ones, cos_img], dim=0)
        sin_img = torch.cat([zeros, sin_img], dim=0)

    return torch.cat([cos_img, sin_img], dim=-1)


def rotate_half(x):
    x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return einops.rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(x, freqs_cis):
    """
    apply rotary position embedding to input tensor
    """
    freqs_cos, freqs_sin = freqs_cis.unsqueeze(1).chunk(2, dim=-1)
    return x * freqs_cos + rotate_half(x) * freqs_sin


class Contiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()


class SwiGLUFFN(nn.Module):
    """
    Swish-Gated Linear Unit Feed-Forward Network
    """

    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    """
    from flash_attn import flash_attn_func
    """
    # def forward(self, x, rope):
    #     bsz, n_ctx, ch = x.shape
    #     qkv = self.qkv(x)
    #     q, k, v = einops.rearrange(
    #         qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads
    #     ).unbind(0)
    #     q = apply_rotary_emb(q, rope).to(v.dtype)
    #     k = apply_rotary_emb(k, rope).to(v.dtype)
    #     q = q.transpose(1, 2)  # [B, N, H, D]
    #     k = k.transpose(1, 2)  # [B, N, H, D]
    #     v = v.transpose(1, 2)  # [B, N, H, D]
    #     x = flash_attn_func(q, k, v, causal=False)  # [B, N, H, D]
    #     return self.proj(x.reshape(bsz, n_ctx, ch))

    """
    sdpa impl.
    """

    def forward(self, x: torch.Tensor, rope: torch.Tensor):
        """
        with sdpa
        """
        bsz, n_ctx, ch = x.shape
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads
        ).unbind(0)
        q = apply_rotary_emb(q, rope).to(v.dtype)
        k = apply_rotary_emb(k, rope).to(v.dtype)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(bsz, n_ctx, ch))


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings.unsqueeze(1)


class ModulatedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_modulation: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_modulation:
            self.norm = nn.RMSNorm(in_features, eps=1e-6)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(in_features, 2 * in_features, bias=bias)
            )
        self.use_modulation = use_modulation

    def forward(self, x, cond=None):
        if self.use_modulation:
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
            x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_modulation: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)

        self.attn = Attention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, expansion_factor=2 / 3 * mlp_ratio)

        self.use_modulation = use_modulation
        self.adaLN_modulation = None

        if self.use_modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, cond=None, rope=None):
        if cond is not None:
            out = self.adaLN_modulation(cond).chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa = out[:3]
            shift_mlp, scale_mlp, gate_mlp = out[3:]
        else:
            shift_msa, scale_msa, gate_msa = None, None, 1.0
            shift_mlp, scale_mlp, gate_mlp = None, None, 1.0

        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=rope,
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp),
        )
        return x


class SyncBN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.bn = nn.SyncBatchNorm(
            num_features=dim, affine=False, track_running_stats=True
        )

    def forward(self, x):
        assert x.ndim in [3, 4]
        shape = x.shape
        x = x.reshape(x.shape[0], self.dim, 1)  # [B, D, +], which is expected by bn
        x = self.bn(x)
        x = x.reshape(shape)
        return x

    @torch.no_grad()
    def return_stats(self):
        return self.bn.running_mean, self.bn.running_var

    @torch.no_grad()
    def _forward(self, x):
        assert x.ndim in [3, 4]
        shape = x.shape
        x = x.reshape(x.shape[0], self.dim, 1)
        x = F.batch_norm(x, running_mean=None, running_var=None, training=True)
        x = x.reshape(shape)
        return x
