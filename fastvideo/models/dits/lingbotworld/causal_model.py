# SPDX-License-Identifier: Apache-2.0
"""LingBot-World-Fast: block-causal, KV-cached, DMD-distilled world model.

This is the streaming variant of LingBot-World. It shares the Cam transformer's
parameter structure and camera conditioner, but replaces full bidirectional
attention with the Causal-Wan block-causal self-attention (rolling KV cache,
attention sink, chunk-by-chunk autoregressive denoising). The camera Plucker
conditioner is optional and applied per block exactly as in the Cam model.
"""

from typing import Any

import torch

from fastvideo.configs.models.dits.lingbotworld import (
    CausalLingBotWorldVideoConfig)
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import (PatchEmbed,
                                               WanCamControlPatchEmbedding)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.causal_wanvideo import (CausalWanTransformer3DModel,
                                                   CausalWanTransformerBlock)
from fastvideo.models.dits.lingbotworld.model import LingBotWorldCamConditioner
from fastvideo.models.dits.wanvideo import WanTimeTextImageEmbedding
from fastvideo.layers.layernorm import LayerNormScaleShift
from fastvideo.platforms import current_platform

import math
import torch.nn as nn

logger = init_logger(__name__)


class CausalLingBotWorldTransformerBlock(CausalWanTransformerBlock):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cam_conditioner = LingBotWorldCamConditioner(self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        frame_seqlen: int | None = None,
        c2ws_plucker_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        temb_seq_len = temb.shape[1]
        tokens_per_temb = hidden_states.shape[1] // temb_seq_len
        if frame_seqlen is None:
            frame_seqlen = tokens_per_temb
        else:
            frame_seqlen = int(frame_seqlen)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        e = self.scale_shift_table + temb
        assert e.shape == (bs, temb_seq_len, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states).unflatten(
                dim=1, sizes=(temb_seq_len, tokens_per_temb)) *
            (1 + scale_msa) + shift_msa).flatten(1, 2)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            frame_seqlen=frame_seqlen,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # Inject camera condition after the self-attention residual update.
        if c2ws_plucker_emb is not None:
            hidden_states = self.cam_conditioner(hidden_states,
                                                 c2ws_plucker_emb)
            norm_hidden_states = self.self_attn_residual_norm.norm(
                hidden_states).to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)

        return hidden_states


class CausalLingBotWorldTransformer3DModel(CausalWanTransformer3DModel):
    _fsdp_shard_conditions = CausalLingBotWorldVideoConfig()._fsdp_shard_conditions
    _compile_conditions = CausalLingBotWorldVideoConfig()._compile_conditions
    _supported_attention_backends = CausalLingBotWorldVideoConfig(
    )._supported_attention_backends
    param_names_mapping = CausalLingBotWorldVideoConfig().param_names_mapping
    reverse_param_names_mapping = CausalLingBotWorldVideoConfig(
    ).reverse_param_names_mapping
    lora_param_names_mapping = CausalLingBotWorldVideoConfig(
    ).lora_param_names_mapping

    def __init__(self, config: CausalLingBotWorldVideoConfig,
                 hf_config: dict[str, Any]) -> None:
        BaseDiT.__init__(self, config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)
        self.patch_embedding_wancamctrl = WanCamControlPatchEmbedding(
            in_chans=config.control_dim * 64,
            embed_dim=inner_dim,
            patch_size=config.patch_size)
        self.c2ws_mlp = MLP(inner_dim,
                            inner_dim,
                            inner_dim,
                            bias=True,
                            act_type="silu")

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            CausalLingBotWorldTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.local_attn_size,
                config.sink_size,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                config.added_kv_proj_dim,
                self._supported_attention_backends,
                prefix=f"{config.prefix}.blocks.{i}")
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Causal-specific
        self.block_mask = None
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3
        self.independent_first_frame = False

        self.__post_init__()

    def _embed_camera(self, c2ws_plucker_emb: torch.Tensor | None,
                      device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor | None:
        if c2ws_plucker_emb is None:
            return None
        c2ws = self.patch_embedding_wancamctrl(
            c2ws_plucker_emb.to(device=device, dtype=dtype))
        return c2ws + self.c2ws_mlp(c2ws)

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        c2ws_plucker_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame)
        freqs_cis = (freqs_cos.to(hidden_states.device),
                     freqs_sin.to(hidden_states.device))

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        c2ws_emb = self._embed_camera(c2ws_plucker_emb, hidden_states.device,
                                      hidden_states.dtype)

        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(
                1, self.text_len - encoder_hidden_states.size(1),
                encoder_hidden_states.size(2))
        ],
                                          dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep.flatten(), encoder_hidden_states,
            encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(
            1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        assert encoder_hidden_states.dtype == orig_dtype

        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index],
                current_start=current_start,
                cache_start=cache_start,
                block_mask=self.block_mask,
                frame_seqlen=post_patch_height * post_patch_width,
                c2ws_plucker_emb=c2ws_emb,
            )

        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                          dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)
        return torch.stack(output)

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        start_frame: int = 0,
        c2ws_plucker_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame)
        freqs_cis = (freqs_cos.to(hidden_states.device),
                     freqs_sin.to(hidden_states.device))

        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size)

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        c2ws_emb = self._embed_camera(c2ws_plucker_emb, hidden_states.device,
                                      hidden_states.dtype)

        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(
                1, self.text_len - encoder_hidden_states.size(1),
                encoder_hidden_states.size(2))
        ],
                                          dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep.flatten(), encoder_hidden_states,
            encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(
            1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        assert encoder_hidden_states.dtype == orig_dtype

        for block in self.blocks:
            hidden_states = block(hidden_states,
                                  encoder_hidden_states,
                                  timestep_proj,
                                  freqs_cis,
                                  block_mask=self.block_mask,
                                  c2ws_plucker_emb=c2ws_emb)

        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                          dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)
        return torch.stack(output)


# Entry point for model registry
EntryClass = CausalLingBotWorldTransformer3DModel
