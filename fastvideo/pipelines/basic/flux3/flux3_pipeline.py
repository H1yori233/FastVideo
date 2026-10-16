# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from fastvideo.attention.selector import record_resolved_attention_backend
from fastvideo.configs.pipelines.flux3 import Flux3PipelineConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader, ProcessorLoader, TextEncoderLoader
from fastvideo.models.vaes.flux3 import Flux3AutoEncoder
from fastvideo.pipelines.basic.flux3.stages import (
    Flux3DecodingStage,
    Flux3DenoisingStage,
    Flux3InputPreparationStage,
    Flux3LatentPreparationStage,
    Flux3PromptUpsamplingStage,
    Flux3TextEncodingStage,
)
from fastvideo.pipelines.basic.flux3.state import RUNTIME_STATE_KEY, Flux3RuntimeState
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lazy_module import LazyModule
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)


class Flux3Pipeline(ComposedPipelineBase):
    pipeline_config_cls = Flux3PipelineConfig
    _required_config_modules = [
        "transformer", "vae", "text_encoder", "tokenizer", "prompt_upsampler", "prompt_upsampler_processor"
    ]
    _lazy_module_names = ("transformer", "vae", "text_encoder", "prompt_upsampler")

    @classmethod
    def get_hf_download_component_dirs(cls) -> tuple[str, ...]:
        return ("transformer", "vae", "text_encoder")

    def forward(self, batch, fastvideo_args):
        batch.extra[RUNTIME_STATE_KEY] = Flux3RuntimeState()
        try:
            return super().forward(batch, fastvideo_args)
        finally:
            batch.extra.pop(RUNTIME_STATE_KEY, None)

    def load_modules(self, fastvideo_args, loaded_modules=None):
        if fastvideo_args.num_gpus != 1 or fastvideo_args.tp_size != 1 or fastvideo_args.sp_size != 1:
            raise ValueError("FLUX.3 supports one GPU with tp_size=sp_size=1")
        if fastvideo_args.training_mode:
            raise ValueError("FLUX.3 currently supports inference only")
        if getattr(fastvideo_args, "override_text_encoder_safetensors", None) is not None:
            raise ValueError("FLUX.3 loads text encoder weights from the official component repositories")
        config = fastvideo_args.pipeline_config
        if (config.dit_config.quant_config is not None or fastvideo_args.override_text_encoder_quant is not None
                or any(encoder.quant_config is not None for encoder in config.text_encoder_configs)):
            raise ValueError("FLUX.3 currently supports unquantized BF16 checkpoints only")
        if (config.dit_precision != "bf16" or config.vae_precision != "fp32"
                or config.vae_decode_precision not in (None, "fp32")
                or any(precision != "bf16" for precision in config.text_encoder_precisions)):
            raise ValueError("FLUX.3 requires BF16 transformer/text encoders and FP32 VAE")
        if config.vae_tiling:
            raise ValueError("FLUX.3 does not support VAE tiling")
        if not config.dit_config.flow_map:
            raise ValueError("FLUX.3 requires the distilled flow-map checkpoint")
        # PUP and generator weights cannot coexist on a 32 GB GPU.
        fastvideo_args.lazy_module_load = True
        if fastvideo_args.dit_layerwise_offload or fastvideo_args.use_fsdp_inference:
            logger.info("FLUX.3 uses deferred components; disabling layerwise offload and FSDP inference")
        fastvideo_args.dit_layerwise_offload = False
        fastvideo_args.use_fsdp_inference = False
        manifest = self._load_config(self.model_path)
        if manifest.get("_class_name") != "Flux3DistilledModularPipeline":
            raise ValueError("FLUX.3 requires the official distilled Modular Diffusers checkpoint")
        transformer_config = json.loads((Path(self.model_path) / "transformer/config.json").read_text())
        if not transformer_config.get("flow_map"):
            raise ValueError("FLUX.3 requires the distilled flow-map checkpoint")

        sources = {}
        self._component_paths = {}
        for name in self.required_config_modules:
            spec = manifest[name][2]
            repo = spec["pretrained_model_name_or_path"]
            revision = spec.get("revision")
            source = (repo, revision)
            if repo == "black-forest-labs/flux-3-image-klein":
                directory = self.model_path
            else:
                if source not in sources:
                    sources[source] = maybe_download_model(repo, revision=revision)
                directory = sources[source]
            self._component_paths[name] = Path(directory) / spec.get("subfolder", "")

        def load_component(name: str) -> Any:
            path = str(self._component_paths[name])
            if name in ("text_encoder", "prompt_upsampler"):
                index = 0 if name == "text_encoder" else 1
                encoder_config = deepcopy(config.text_encoder_configs[index])
                encoder_config.update_model_arch(json.loads((Path(path) / "config.json").read_text()))
                record_resolved_attention_backend(encoder_config)
                return TextEncoderLoader().load_model(
                    path,
                    encoder_config,
                    get_local_torch_device(),
                    fastvideo_args,
                    config.text_encoder_precisions[index],
                )
            if name == "vae":
                fastvideo_args.model_paths[name] = path
                device = "cpu" if fastvideo_args.vae_cpu_offload else get_local_torch_device()
                return Flux3AutoEncoder.from_pretrained(path,
                                                        config=deepcopy(config.vae_config),
                                                        dtype=torch.float32,
                                                        device=device)
            if name == "prompt_upsampler_processor":
                return ProcessorLoader().load(path, fastvideo_args)
            return PipelineComponentLoader.load_module(
                name,
                path,
                manifest[name][0],
                fastvideo_args,
            )

        modules = dict(loaded_modules or {})
        for name in self.required_config_modules:
            if name not in modules:
                modules[name] = (LazyModule(name, lambda name=name: load_component(name))
                                 if name in self._lazy_module_names else load_component(name))
        return modules

    def create_pipeline_stages(self, fastvideo_args):
        prompts_path = self._component_paths["prompt_upsampler"] / "system_prompts.json"
        system_prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        self.add_stage("input_preparation", Flux3InputPreparationStage())
        self.add_stage(
            "prompt_upsampling",
            Flux3PromptUpsamplingStage(self.get_module("prompt_upsampler"),
                                       self.get_module("prompt_upsampler_processor"), system_prompts))
        self.add_stage("text_encoding",
                       Flux3TextEncodingStage(self.get_module("text_encoder"), self.get_module("tokenizer")))
        self.add_stage("latent_preparation", Flux3LatentPreparationStage(self.get_module("vae")))
        self.add_stage("denoising", Flux3DenoisingStage(self.get_module("transformer")))
        self.add_stage("decoding", Flux3DecodingStage(self.get_module("vae")))


EntryClass = Flux3Pipeline
