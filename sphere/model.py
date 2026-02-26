# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from sphere.layers import (
    Block,
    Contiguous,
    get_2d_sincos_pos_embed,
    get_rope_tensor,
    LabelEmbedder,
    ModulatedLinear,
    shift_range,
    stratified_unit_radii,
    vector_rms_norm,
)
from sphere.mixer import MLPMixer
from sphere.utils import (
    vector_compute_angle,
    vector_compute_magnitude,
)


SIZE_DICT = {
    "small": {"width": 512, "layers": 8, "heads": 8, "in_context_start": 2},
    "base": {"width": 768, "layers": 12, "heads": 12, "in_context_start": 4},
    "large": {"width": 1024, "layers": 24, "heads": 16, "in_context_start": 8},
    "xlarge": {"width": 1152, "layers": 28, "heads": 16, "in_context_start": 8},
    "huge": {"width": 1280, "layers": 32, "heads": 16, "in_context_start": 10},
    "giant": {"width": 1664, "layers": 40, "heads": 16, "in_context_start": 12},
}


class Transformer(nn.Module):

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base",
        model_type: str = "encoder",
        token_chns: int = 16,
        num_classes: int = 0,
        in_context_size: int = 0,
        pixel_head_type: str = "linear",
        latent_mlp_mixer_depth: int = 0,
        halve_model_size: bool = False,
        spherify_model: bool = False,
        affine_latent_mlp_mixer: bool = True,
    ):
        super().__init__()
        assert model_type in ["encoder", "decoder"]
        assert model_size in SIZE_DICT

        self.model_type = model_type
        self.model_size = model_size
        self.input_size = input_size
        self.patch_size = patch_size
        self.token_chns = token_chns
        self.grid_size = self.input_size // self.patch_size

        # make true to spherify the output of each transformer block
        self.spherify_model = spherify_model

        self.num_tokens = self.grid_size**2

        params = SIZE_DICT[self.model_size]
        self.hidden_size = params["width"]
        self.num_layers = params["layers"]
        if halve_model_size:
            self.num_layers = self.num_layers // 2
        self.num_heads = params["heads"]
        self.inctx_start = params["in_context_start"]
        self.inctx_size = in_context_size

        self.latent_shape = (1, self.num_tokens, self.token_chns)
        self.num_classes = num_classes

        # input embed
        if model_type == "encoder":
            self.x_embedder = nn.Sequential(
                nn.Conv2d(
                    3, self.hidden_size, self.patch_size, self.patch_size, bias=False
                ),
                nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1),
                Rearrange("b c h w -> b (h w) c", h=self.grid_size, w=self.grid_size),
                Contiguous(),
            )
        else:
            # decoder input embed
            if latent_mlp_mixer_depth > 0:
                self.x_embedder = nn.Sequential(
                    nn.Linear(self.token_chns, self.hidden_size),
                    MLPMixer(
                        self.num_tokens,
                        self.hidden_size,
                        depth=latent_mlp_mixer_depth,
                        use_affine=affine_latent_mlp_mixer,
                    ),
                )
            else:
                self.x_embedder = nn.Linear(self.token_chns, self.hidden_size)

        # pos embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.hidden_size), requires_grad=False
        )

        # rope
        rope = get_rope_tensor(
            dim=self.hidden_size // self.num_heads,
            seq_h=self.grid_size,
            seq_w=self.grid_size,
        ).unsqueeze(0)
        self.register_buffer("rope", rope, persistent=False)

        # class embed
        self.y_embedder = (
            LabelEmbedder(self.num_classes, self.hidden_size, 0.1)
            if self.num_classes > 0
            else None
        )
        self.use_modulation = self.y_embedder is not None

        # in-context embed (can be used for unconditional generation)
        self.use_inctx = self.inctx_size > 0
        if self.use_inctx:
            inctx_rope = get_rope_tensor(
                dim=self.hidden_size // self.num_heads,
                seq_h=self.grid_size,
                seq_w=self.grid_size,
                pad_size=self.inctx_size,
            ).unsqueeze(0)
            self.register_buffer("inctx_rope", inctx_rope, persistent=False)

            self.inctx_pos_embed = nn.Parameter(
                torch.zeros(1, self.inctx_size, self.hidden_size), requires_grad=True
            )

            if not self.use_modulation:
                # for unconditional generation like register tokens
                self.inctx_embed = nn.Parameter(
                    torch.zeros(1, self.inctx_size, self.hidden_size),
                    requires_grad=True,
                )

        # transformer
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    use_modulation=self.use_modulation,
                )
                for _ in range(self.num_layers)
            ]
        )

        # pred head
        if model_type == "encoder":
            # encoder latent head
            self.ffn = ModulatedLinear(
                self.hidden_size, self.token_chns, use_modulation=self.use_modulation
            )
            if latent_mlp_mixer_depth > 0:
                self.out = MLPMixer(
                    self.num_tokens,
                    self.token_chns,
                    depth=latent_mlp_mixer_depth,
                    use_affine=affine_latent_mlp_mixer,
                )
            else:
                self.out = nn.Identity()

        elif model_type == "decoder":
            # decoder pixel head
            self.pixel_head_type = pixel_head_type.lower()
            assert self.pixel_head_type in ["linear", "conv"]

            intermediate_chns = (
                32 if pixel_head_type == "conv" else self.patch_size**2 * 3
            )
            self.ffn = ModulatedLinear(
                self.hidden_size, intermediate_chns, use_modulation=self.use_modulation
            )

            if self.pixel_head_type == "linear":
                self.out = nn.Sequential(
                    Rearrange(
                        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                        h=self.grid_size,
                        w=self.grid_size,
                        p1=self.patch_size,
                        p2=self.patch_size,
                    ),
                    Contiguous(),
                    nn.Conv2d(
                        3, 3, stride=1, kernel_size=3, padding=1, padding_mode="reflect"
                    ),
                    nn.Tanh(),
                )
            elif self.pixel_head_type == "conv":
                self.out = nn.Sequential(
                    Rearrange(
                        "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
                    ),
                    Contiguous(),
                    nn.Conv2d(
                        intermediate_chns,
                        intermediate_chns,
                        kernel_size=3,
                        padding=1,
                        padding_mode="reflect",
                    ),
                    nn.SiLU(),
                    nn.ConvTranspose2d(
                        in_channels=intermediate_chns,
                        out_channels=3,
                        kernel_size=self.patch_size + 2,
                        stride=self.patch_size,
                        padding=1,
                    ),
                    nn.Tanh(),
                )

        self.initialize_weights()

    def return_last_layer_params(self):
        if self.model_type == "decoder":
            return self.out[-1].weight

    def initialize_weights(self):
        # init transformer layers and mlp-mixer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # init pos embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_tokens**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # init input embed
        if self.model_type == "encoder":
            w1 = self.x_embedder[0].weight.data
            nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
            w2 = self.x_embedder[1].weight.data
            nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
            nn.init.constant_(self.x_embedder[1].bias, 0)

        if self.model_type == "decoder":
            if isinstance(self.x_embedder, nn.Linear):
                nn.init.normal_(self.x_embedder.weight)
                nn.init.constant_(self.x_embedder.bias, 0)

            else:
                for m in self.x_embedder.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight)
                        nn.init.constant_(m.bias, 0)

        # init class embed
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # zero-out adaLN modulation layers in transformer
        for block in self.blocks:
            if not block.use_modulation:
                continue
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out adaLN modulation layers in ffn
        if self.use_modulation:
            nn.init.constant_(self.ffn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.ffn.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.ffn.linear.weight, 0)
        nn.init.constant_(self.ffn.linear.bias, 0)

        if self.model_type == "decoder":
            if self.pixel_head_type == "conv":
                w = self.out[-4].weight.data
                nn.init.xavier_uniform_(w.view(w.shape[0], -1), gain=0.01)
                nn.init.constant_(self.out[-4].bias, 0)

            w = self.out[-2].weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1), gain=0.01)
            nn.init.constant_(self.out[-2].bias, 0)

        # init in-context embed
        if self.use_inctx:
            nn.init.normal_(self.inctx_pos_embed, std=0.02)
            if not self.use_modulation:
                nn.init.normal_(self.inctx_embed, std=0.02)

    def forward(self, x, y=None, log_dict=None, cond_embed=None):
        """
        x: input images [B, C, H, W] or latents [B, L, D]
        y: class tokens [B] for condition
        """
        B = x.shape[0]
        c = self.y_embedder(y, self.training) if self.y_embedder else None

        if cond_embed is not None:
            c = cond_embed

        x = self.x_embedder(x)
        _dtype = x.dtype  # get autocast dtype
        x = x + self.pos_embed
        x = x.to(_dtype)

        if log_dict is not None:
            log_dict[f"x_std_{self.model_type}"] = x.std(dim=(1, 2)).mean().item()

        rope = self.rope.repeat(B, 1, 1)
        if self.use_inctx:
            inctx_rope = self.inctx_rope.repeat(B, 1, 1)
        inctx_size = 0

        for i, block in enumerate(self.blocks):
            if self.use_inctx and i == self.inctx_start:
                if c is not None:
                    inctx_embed = c.repeat(1, self.inctx_size, 1)
                else:
                    inctx_embed = self.inctx_embed.repeat(B, 1, 1)

                inctx_embed = inctx_embed + self.inctx_pos_embed
                x = torch.cat([inctx_embed, x], dim=1)
                inctx_size = self.inctx_size

            if self.use_inctx and i >= self.inctx_start:
                rope = inctx_rope

            x = block(x, cond=c, rope=rope)

            if self.spherify_model and i != self.num_layers - 1:
                xl = x[:, :inctx_size, :]
                xr = x[:, inctx_size:, :]
                xr = vector_rms_norm(xr)
                x = torch.cat([xl, xr], dim=1)

        x = x[:, inctx_size:, :]
        x = self.ffn(x, cond=c)
        x = self.out(x)
        return x


class G(nn.Module):
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        vit_enc_model_size: str = "base",
        vit_dec_model_size: str = "base",
        token_channels: int = 16,
        num_classes: int = 0,
        in_context_size: int = 0,
        pixel_head_type: str = "linear",
        halve_model_size: bool = False,
        spherify_model: bool = False,
        use_pixel_consistency: bool = False,
        use_latent_consistency: bool = False,
        noise_sigma_max_angle: float = 85.0,
        vit_enc_latent_mlp_mixer_depth: int = 0,
        vit_dec_latent_mlp_mixer_depth: int = 0,
        affine_latent_mlp_mixer: bool = True,
        mix_hard_cases: bool = False,
        mix_hard_cases_prob: float = 0.1,
        mix_hard_cases_max_angle: float = 89.0,
    ):
        super().__init__()

        self.encoder = Transformer(
            input_size=input_size,
            patch_size=patch_size,
            model_size=vit_enc_model_size,
            model_type="encoder",
            token_chns=token_channels,
            num_classes=num_classes,
            in_context_size=in_context_size,
            halve_model_size=halve_model_size,
            latent_mlp_mixer_depth=vit_enc_latent_mlp_mixer_depth,
            affine_latent_mlp_mixer=affine_latent_mlp_mixer,
        )

        self.decoder = Transformer(
            input_size=input_size,
            patch_size=patch_size,
            model_size=vit_dec_model_size,
            model_type="decoder",
            token_chns=token_channels,
            num_classes=num_classes,
            in_context_size=in_context_size,
            halve_model_size=halve_model_size,
            latent_mlp_mixer_depth=vit_dec_latent_mlp_mixer_depth,
            affine_latent_mlp_mixer=affine_latent_mlp_mixer,
            spherify_model=spherify_model,
            pixel_head_type=pixel_head_type,
        )

        self.num_classes = num_classes
        self.latent_shape = self.decoder.latent_shape  # [1, N, D]
        self.use_modulation = self.encoder.use_modulation or self.decoder.use_modulation

        # spherify fn
        self.f = partial(vector_rms_norm, zero_mean=False)

        self.noise_sigma_max_angle = noise_sigma_max_angle
        self.use_pix_con = use_pixel_consistency
        self.use_lat_con = use_latent_consistency

        self.mix_hard_cases = mix_hard_cases
        self.mix_hard_cases_prob = mix_hard_cases_prob
        self.mix_hard_cases_max_angle = mix_hard_cases_max_angle
        assert noise_sigma_max_angle <= mix_hard_cases_max_angle

        self.log_dict = {}

    """
    functions for training
    """

    def forward(self, x, y=None):
        extra_loss = {}

        # encode
        z = self.encoder(x, y)  # [B, N, D]
        z_clean = z.clone().detach()  # target for latent consistency loss

        # log stats
        mag_z = vector_compute_magnitude(z)
        self.log_dict["avg_mag_z"] = mag_z.mean().detach().item()
        self.log_dict["std_mag_z"] = mag_z.std().detach().item()

        # f(z)
        z = self.f(z)

        # get noise
        e = torch.randn_like(z)

        # get sigma
        sigma = np.tan(np.deg2rad(self.noise_sigma_max_angle))

        # get r in [0, 1]
        r = stratified_unit_radii(z.shape, device=z.device, dtype=z.dtype)  # [B, ...]

        # mix with hard cases
        if self.mix_hard_cases:
            scalers = torch.rand_like(r)  # [0, 1)
            r_mixed = shift_range(
                scalers,
                1.0,
                np.tan(np.deg2rad(self.mix_hard_cases_max_angle)) / sigma,
            )
            m = (scalers > self.mix_hard_cases_prob).to(r.dtype)
            r_NOISY = m * r + (1 - m) * r_mixed
        else:
            r_NOISY = r

        self.log_dict["avg_r_NOISY"] = r_NOISY.mean().detach().item()
        self.log_dict["std_r_NOISY"] = r_NOISY.std().detach().item()

        z_NOISY = self.f(z + r_NOISY * sigma * e)
        z_NOISY = torch.where(r_NOISY == 0, z, z_NOISY)

        # # get angle between f(z) and z_NOISY
        # alpha = vector_compute_angle(z, z + r_NOISY * sigma * e)

        # pixel consistency part
        if self.use_pix_con:
            # get r_noisy
            r = r * stratified_unit_radii(z.shape, device=z.device, dtype=z.dtype) * 0.5

            # f(f(z) + r_noisy * sigma * e)
            z_noisy = self.f(z + r * sigma * e)
            z_noisy = torch.where(r == 0, z, z_noisy)

            # # override alpha
            # alpha = vector_compute_angle(z, z + r * sigma * e)

            # concat
            z = torch.cat([z_NOISY, z_noisy], dim=0)  # [B x 2, ...]
            y = torch.cat([y, y], dim=0)  # [B x 2, ...]

        else:
            z = z_NOISY

        # decode
        x = self.decoder(z, y, log_dict=self.log_dict)  # [B, 3, H, W]

        # latent consistency part
        v_NOISY = None
        if self.training and self.use_lat_con:
            B = z_clean.shape[0]

            x_NOISY = x[:B]  # fake images from z_NOISY
            v_NOISY = self.encoder(x_NOISY, y[:B])

        return x, extra_loss, v_NOISY, z_clean

    """
    functions for inference
    """

    def spherify(self, z, noise_scaler=1.0, sampling=False, cache_noise=False):
        z = self.f(z)

        if sampling and noise_scaler > 0.0:  # with adding noise
            _dtype = z.dtype
            r = noise_scaler
            if cache_noise and self.cached_noise is not None:
                e = self.cached_noise
            else:
                e = torch.randn_like(z)
                if cache_noise:
                    self.cached_noise = e
            sigma = np.tan(np.deg2rad(self.noise_sigma_max_angle))
            z = self.f(z + r * sigma * e)
            z = z.to(_dtype)

        return z

    @torch.no_grad()
    def reconstruct(self, x, y=None, noise_scaler=1.0, sampling=False):
        if self.num_classes > 0 and y is None:
            y = torch.full((x.shape[0],), self.num_classes)
            y = y.to(device=x.device, dtype=torch.long)  # null class embedding
        z = self.encoder(x, y)
        z = self.spherify(z, noise_scaler=noise_scaler, sampling=sampling)
        x = self.decoder(z, y)
        x = torch.clamp(x * 0.5 + 0.5, 0, 1)
        return x

    @torch.no_grad()
    def generate(
        self,
        batch_size,
        y=None,
        cfg=1.0,
        cfg_position="combo",
        forward_steps=1,
        use_sampling_scheduler=False,
        cache_sampling_noise=False,
        device="cuda",
    ):
        assert cfg_position in ["enc", "dec", "combo"]
        self.cached_noise = None  # clean the cache

        use_cfg = cfg > 1.0
        do_enc_cfg = use_cfg and cfg_position in ["enc", "combo"]
        do_dec_cfg = use_cfg and cfg_position in ["dec", "combo"]

        if cfg_position == "combo":
            cfg = cfg**0.5  # avoid double dipping on cfg / over exposure

        e = torch.randn(batch_size, *self.latent_shape[1:]).to(device)

        if self.num_classes > 0:
            if y is None:
                y = torch.randint(
                    low=0,
                    high=self.decoder.num_classes,
                    size=(batch_size,),
                    device=device,
                )
            if use_cfg:
                y_uncond = torch.full_like(y, self.num_classes)
        else:
            y, y_uncond = None, None

        z = self.spherify(e, sampling=False)
        x = self.decoder(z, y)

        if do_dec_cfg:
            x_uncond = self.decoder(z, y_uncond)
            x = torch.lerp(x_uncond, x, cfg).clamp_(-1, 1)

        h = x.clone()
        h = torch.clamp(h * 0.5 + 0.5, 0, 1)

        if forward_steps == 1:
            return h, h

        for step in range(forward_steps - 1):
            z = self.encoder(x, y)
            if do_enc_cfg:
                z_uncond = self.encoder(x, y_uncond)
                z = torch.lerp(z_uncond, z, cfg)

            if use_sampling_scheduler:
                T = forward_steps
                t = step + 1
                r = 1 - t / T
            else:
                r = 1.0

            z = self.spherify(
                z, sampling=True, noise_scaler=r, cache_noise=cache_sampling_noise
            )
            x = self.decoder(z, y)

            if do_dec_cfg:
                x_uncond = self.decoder(z, y_uncond)
                x = torch.lerp(x_uncond, x, cfg).clamp_(-1, 1)

        x = torch.clamp(x * 0.5 + 0.5, 0, 1)
        return h, x

    @torch.no_grad()
    def edit(
        self,
        batch_size,
        y=None,
        cfg=1.0,
        cfg_position="enc",
        forward_steps=1,
        use_sampling_scheduler=False,
        noise_strength_scaler=1.0,
        cache_sampling_noise=False,
        spherify_input_noise=True,
        y_enc_embed=None,
        y_dec_embed=None,
        x_enc_image=None,
        x_dir_image=None,
        input_noise=None,
        return_step_images=False,
        device="cuda",
    ):
        """
        generate images with more control
        """
        assert cfg_position in ["enc", "dec", "combo"]
        if y_enc_embed is not None or y_dec_embed is not None:
            assert y_dec_embed is not None
            assert y_enc_embed is not None

        self.cached_noise = None

        use_cfg = cfg > 1.0
        do_enc_cfg = use_cfg and cfg_position in ["enc", "combo"]
        do_dec_cfg = use_cfg and cfg_position in ["dec", "combo"]

        if cfg_position == "combo":
            cfg = cfg**0.5  # avoid double dipping on cfg / over exposure

        e = torch.randn(batch_size, *self.latent_shape[1:]).to(device)

        if input_noise is not None:
            assert input_noise.shape == e.shape
            e = input_noise

        if self.num_classes > 0:
            if y is None:
                y = torch.randint(
                    low=0,
                    high=self.decoder.num_classes,
                    size=(batch_size,),
                    device=device,
                )
            if use_cfg:
                y_uncond = torch.full_like(y, self.num_classes)
        else:
            y, y_uncond = None, None

        if spherify_input_noise:
            z = self.spherify(e, sampling=False)
        else:
            z = e
        x = self.decoder(z, y, cond_embed=y_dec_embed)

        if do_dec_cfg:
            x_uncond = self.decoder(z, y_uncond)
            x = torch.lerp(x_uncond, x, cfg).clamp_(-1, 1)

        h = x.clone()
        h = torch.clamp(h * 0.5 + 0.5, 0, 1)

        if forward_steps == 1:
            return h, h

        if return_step_images:
            step_images = [h]

        if x_enc_image is not None:
            assert x_enc_image.ndim == 4
            assert x_enc_image.shape[1:] == x.shape[1:]
            x = x_enc_image

            # reset variables in this case
            forward_steps += 1
            if return_step_images:
                step_images = []

        if x_dir_image is not None:
            assert x_dir_image.ndim == 4
            assert x_dir_image.shape[1:] == x.shape[1:]
            x_dir = self.encoder(x_dir_image, y_uncond)
            self.cached_noise = x_dir

        for step in range(forward_steps - 1):
            z = self.encoder(x, y, cond_embed=y_enc_embed)
            if do_enc_cfg:
                z_uncond = self.encoder(x, y_uncond)
                z = torch.lerp(z_uncond, z, cfg)

            if use_sampling_scheduler:
                T = forward_steps
                t = step + 1
                r = 1 - t / T
            else:
                r = 1.0

            r *= noise_strength_scaler

            z = self.spherify(
                z, sampling=True, noise_scaler=r, cache_noise=cache_sampling_noise
            )
            x = self.decoder(z, y, cond_embed=y_dec_embed)

            if do_dec_cfg:
                x_uncond = self.decoder(z, y_uncond)
                x = torch.lerp(x_uncond, x, cfg).clamp_(-1, 1)

            if return_step_images:
                step_image = torch.clamp(x * 0.5 + 0.5, 0, 1)
                step_images.append(step_image)

        if return_step_images:
            return h, step_images

        x = torch.clamp(x * 0.5 + 0.5, 0, 1)
        return h, x
