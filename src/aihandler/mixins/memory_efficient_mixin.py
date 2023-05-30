import os
import torch
from aihandler.settings import LOG_LEVEL
from aihandler.logger import logger
import logging
logging.disable(LOG_LEVEL)
logger.set_level(logger.DEBUG)


class MemoryEfficientMixin:
    enable_model_cpu_offload: bool = False
    use_attention_slicing: bool = False
    use_tf32: bool = False
    use_enable_vae_slicing: bool = False
    use_xformers: bool = False
    use_tiled_vae: bool = False
    _use_last_channels = True
    _use_enable_sequential_cpu_offload = True
    _use_attention_slicing = True
    _use_tf32 = True
    _use_enable_vae_slicing = True
    _use_xformers = False
    _use_tiled_vae = False

    @property
    def use_last_channels(self):
        return self._use_last_channels and not self.is_txt2vid

    @use_last_channels.setter
    def use_last_channels(self, value):
        self._use_last_channels = value

    @property
    def use_enable_sequential_cpu_offload(self):
        return self._use_enable_sequential_cpu_offload

    @use_enable_sequential_cpu_offload.setter
    def use_enable_sequential_cpu_offload(self, value):
        self._use_enable_sequential_cpu_offload = value

    @property
    def use_attention_slicing(self):
        return self._use_attention_slicing

    @use_attention_slicing.setter
    def use_attention_slicing(self, value):
        self._use_attention_slicing = value

    @property
    def use_tf32(self):
        return self._use_tf32

    @use_tf32.setter
    def use_tf32(self, value):
        self._use_tf32 = value

    @property
    def enable_vae_slicing(self):
        return self._enable_vae_slicing

    @enable_vae_slicing.setter
    def enable_vae_slicing(self, value):
        self._enable_vae_slicing = value

    @property
    def use_xformers(self):
        if not self.cuda_is_available:
            return False
        return self._use_xformers

    @use_xformers.setter
    def use_xformers(self, value):
        self._use_xformers = value

    @property
    def use_accelerated_transformers(self):
        return self._use_accelerated_transformers

    @use_accelerated_transformers.setter
    def use_accelerated_transformers(self, value):
        self._use_accelerated_transformers = value

    @property
    def use_torch_compile(self):
        return self._use_torch_compile

    @use_torch_compile.setter
    def use_torch_compile(self, value):
        self._use_torch_compile = value

    @property
    def use_tiled_vae(self):
        return self._use_tiled_vae

    @use_tiled_vae.setter
    def use_tiled_vae(self, value):
        self._use_tiled_vae = value

    def apply_last_channels(self):
        if self.use_last_channels:
            logger.info("Enabling torch.channels_last")
            self.pipe.unet.to(memory_format=torch.channels_last)
        else:
            logger.info("Disabling torch.channels_last")
            self.pipe.unet.to(memory_format=torch.contiguous_format)

    def apply_vae_slicing(self):
        if self.action not in ["img2img", "depth2img", "pix2pix", "outpaint", "superresolution", "controlnet", "upscale"]:
            if self.use_enable_vae_slicing:
                logger.info("Enabling vae slicing")
                self.pipe.enable_vae_slicing()
            else:
                logger.info("Disabling vae slicing")
                self.pipe.disable_vae_slicing()

    def apply_attention_slicing(self):
        if self.use_attention_slicing:
            logger.info("Enabling attention slicing")
            self.pipe.enable_attention_slicing()
        else:
            logger.info("Disabling attention slicing")
            self.pipe.disable_attention_slicing()

    def apply_tiled_vae(self):
        if self.use_tiled_vae:
            logger.info("Applying tiled vae")
            # from diffusers import UniPCMultistepScheduler
            # self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            try:
                self.pipe.vae.enable_tiling()
            except AttributeError:
                logger.warning("Tiled vae not supported for this model")

    def apply_xformers(self):
        if self.use_xformers:
            logger.info("Applying xformers")
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
            self.pipe.enable_xformers_memory_efficient_attention()
        else:
            logger.info("Disabling xformers")
            self.pipe.disable_xformers_memory_efficient_attention()

    def apply_accelerated_transformers(self):
        if self.use_accelerated_transformers:
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())

    def save_pipeline(self):
        if self.use_torch_compile:
            file_path = os.path.join(os.path.join(self.model_base_path, self.model_path, "compiled"))
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            torch.save(self.pipe.unet.state_dict(), os.path.join(file_path, "unet.pt"))

    def apply_torch_compile(self):
        """
        Torch compile has limited support
            - No support for Windows
            - Fails with the compiled version of AI Runner
        Because of this, we are disabling it until a better solution is found.

        if self.use_torch_compile:
            logger.debug("Compiling torch model")
            self.pipe.unet = torch.compile(self.pipe.unet)
        load unet state_dict from disc
        file_path = os.path.join(os.path.join(self.model_base_path, self.model_path, "compiled"))
        if os.path.exists(file_path):
            logger.debug("Loading compiled torch model")
            state_dict = torch.load(os.path.join(file_path, "unet.pt"), map_location="cpu")
            self.pipe.unet.state_dict = state_dict
        """
        return

    def move_pipe_to_cuda(self, pipe):
        if not self.use_enable_sequential_cpu_offload and not self.enable_model_cpu_offload:
            logger.info("Moving to cuda")
            pipe.to("cuda", torch.half) if self.cuda_is_available else None
        return pipe

    def move_pipe_to_cpu(self, pipe):
        logger.info("Moving to cpu")
        try:
            pipe.to("cpu", torch.float32)
        except NotImplementedError:
            logger.warning("Not implemented error when moving to cpu")
        return pipe

    def apply_cpu_offload(self):
        if self.use_enable_sequential_cpu_offload and not self.enable_model_cpu_offload:
            logger.info("Enabling sequential cpu offload")
            self.pipe = self.move_pipe_to_cpu(self.pipe)
            try:
                self.pipe.enable_sequential_cpu_offload()
            except NotImplementedError:
                logger.warning("Not implemented error when applying sequential cpu offload")
                self.pipe = self.move_pipe_to_cuda(self.pipe)

    def apply_model_offload(self):
        if self.enable_model_cpu_offload \
           and not self.use_enable_sequential_cpu_offload \
           and hasattr(self.pipe, "enable_model_cpu_offload"):
            logger.info("Enabling model cpu offload")
            self.pipe = self.move_pipe_to_cpu(self.pipe)
            self.pipe.enable_model_cpu_offload()

    def apply_memory_efficient_settings(self):
        logger.info("Applying memory efficient settings")
        self.apply_last_channels()
        self.apply_vae_slicing()
        self.apply_cpu_offload()
        self.apply_model_offload()
        self.pipe = self.move_pipe_to_cuda(self.pipe)
        self.apply_attention_slicing()
        self.apply_tiled_vae()
        self.apply_xformers()
        self.apply_accelerated_transformers()
        self.apply_torch_compile()
