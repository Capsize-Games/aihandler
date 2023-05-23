import os
import gc
import cv2
import numpy as np
import requests
from aihandler.base_runner import BaseRunner
from aihandler.controlnet_utils import ade_palette
from aihandler.qtvar import ImageVar
import traceback
import torch
import io
from aihandler.settings import LOG_LEVEL
from aihandler.logger import logger
import logging
logging.disable(LOG_LEVEL)
logger.set_level(logger.DEBUG)
from PIL import Image
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from compel import Compel
from collections import defaultdict
from safetensors.torch import load_file

os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def image_to_byte_array(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


class SDRunner(BaseRunner):
    _current_model: str = ""
    _previous_model: str = ""
    scheduler_name: str = "Euler a"
    do_nsfw_filter: bool = False
    initialized: bool = False
    seed: int = 42
    model_base_path: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    num_inference_steps: int = 20
    height: int = 512
    width: int = 512
    steps: int = 20
    ddim_eta: float = 0.5
    C: int = 4
    f: int = 8
    batch_size: int = 1
    n_samples: int = 1
    pos_x: int = 0
    pos_y: int = 0
    outpaint_box_rect = None
    hf_token: str = ""
    enable_model_cpu_offload: bool = False
    use_attention_slicing: bool = False
    use_tf32: bool = False
    use_enable_vae_slicing: bool = False
    use_xformers: bool = False
    use_tiled_vae: bool = False
    reload_model: bool = False
    action: str = ""
    options: dict = {}
    model = None
    do_cancel = False
    schedulers: dict = {
        "DDIM": "DDIMScheduler",
        "DDIM Inverse": "DDIMInverseScheduler",
        "DDPM": "DDPMScheduler",
        "DEIS": "DEISMultistepScheduler",
        "DPM Discrete": "KDPM2DiscreteScheduler",
        "DPM Discrete a": "KDPM2AncestralDiscreteScheduler",
        "Euler a": "EulerAncestralDiscreteScheduler",
        "Euler": "EulerDiscreteScheduler",
        "Heun": "HeunDiscreteScheduler",
        "IPNM": "IPNDMScheduler",
        "LMS": "LMSDiscreteScheduler",
        "Multistep DPM": "DPMSolverMultistepScheduler",
        "PNDM": "PNDMScheduler",
        "DPM singlestep": "DPMSolverSinglestepScheduler",
        "RePaint": "RePaintScheduler",
        "Karras Variance exploding": "KarrasVeScheduler",
        "UniPC": "UniPCMultistepScheduler",
        "VE-SDE": "ScoreSdeVeScheduler",
        "VP-SDE": "ScoreSdeVpScheduler",
        "VQ Diffusion": " VQDiffusionScheduler",
    }
    registered_schedulers: dict = {}
    safety_checker = None
    current_model_branch = None
    txt2img = None
    img2img = None
    pix2pix = None
    outpaint = None
    depth2img = None
    controlnet = None
    superresolution = None
    txt2vid = None
    upscale = None
    state = None
    local_files_only = True
    lora_loaded = False
    loaded_lora = []

    # memory settings
    _use_last_channels = True
    _use_enable_sequential_cpu_offload = True
    _use_attention_slicing = True
    _use_tf32 = True
    _use_enable_vae_slicing = True
    _use_xformers = False
    _use_tiled_vae = False
    _settings = None
    _action = None
    do_change_scheduler = False
    embeds_loaded = False
    controlnet_type = "canny"
    options = {}
    # active_extensions = []  TODO: extensions

    @property
    def do_mega_scale(self):
        #return self.is_superresolution
        return False

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, value):
        self._action = value

    @property
    def action_has_safety_checker(self):
        return self.action not in ["depth2img", "superresolution"]

    @property
    def is_outpaint(self):
        return self.action == "outpaint"

    @property
    def is_txt2img(self):
        return self.action == "txt2img"

    @property
    def is_txt2vid(self):
        return self.action == "txt2vid"

    @property
    def is_upscale(self):
        return self.action == "upscale"

    @property
    def is_img2img(self):
        return self.action == "img2img"

    @property
    def is_controlnet(self):
        return self.action == "controlnet"

    @property
    def is_depth2img(self):
        return self.action == "depth2img"

    @property
    def is_pix2pix(self):
        return self.action == "pix2pix"

    @property
    def is_superresolution(self):
        return self.action == "superresolution"

    @property
    def current_model(self):
        return self._current_model

    @current_model.setter
    def current_model(self, model):
        if self._current_model != model:
            self._current_model = model
            if self.initialized:
                logger.info("SDRunner initialized")
                self._load_model()

    @property
    def model_path(self):
        if self.current_model and os.path.exists(self.current_model):
            return self.current_model
        base_path = self.settings_manager.settings.model_base_path.get()
        path = None
        if self.action == "outpaint":
            path = self.settings_manager.settings.outpaint_model_path.get()
        elif self.action == "pix2pix":
            path = self.settings_manager.settings.pix2pix_model_path.get()
        elif self.action == "depth2img":
            path = self.settings_manager.settings.depth2img_model_path.get()
        if path is None or path == "":
            path = base_path
        if self.current_model:
            path = os.path.join(path, self.current_model)
        if not os.path.exists(path):
            return self.current_model
        return path

    @property
    def scheduler(self):
        return self.load_scheduler()

    def load_scheduler(self, schduler_class_name=None):
        import diffusers

        if not self.model_path or self.model_path == "":
            traceback.print_stack()
            raise Exception("Chicken / egg problem, model path not set")

        if self.is_ckpt_model or self.is_safetensors:  # skip scheduler for ckpt models
            return None
        if schduler_class_name:
            scheduler_class = getattr(diffusers, schduler_class_name)
        else:
            if self.is_txt2vid:
                scheduler_class = getattr(diffusers, "DPMSolverMultistepScheduler")
            elif self.is_upscale:
                scheduler_class = getattr(diffusers, "EulerDiscreteScheduler")
            elif self.is_superresolution:
                scheduler_class = getattr(diffusers, "DDIMScheduler")
            else:
                scheduler_class = getattr(diffusers, self.schedulers[self.scheduler_name])
        kwargs = {
            "subfolder": "scheduler"
        }
        # check if self.scheduler_name contains ++
        if self.scheduler_name.startswith("DPM"):
            kwargs["lower_order_final"] = self.num_inference_steps < 15
            if self.scheduler_name.find("++") != -1:
                kwargs["algorithm_type"] = "dpmsolver++"
            else:
                kwargs["algorithm_type"] = "dpmsolver"
        if self.current_model_branch:
            kwargs["variant"] = self.current_model_branch
        return scheduler_class.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            use_auth_token=self.data["options"]["hf_token"],
            **kwargs
        )

    @property
    def cuda_error_message(self):
        if self.is_superresolution and self.scheduler_name == "DDIM":
            return f"Unable to run the model at {self.width}x{self.height} resolution using the DDIM scheduler. Try changing the scheduler to LMS or PNDM and try again."

        return f"You may not have enough GPU memory to run the model at {self.width}x{self.height} resolution. Potential solutions: try again, restart the application, use a smaller size, upgrade your GPU."
        # clear cache

    @property
    def is_pipe_loaded(self):
        if self.is_txt2img:
            return self.txt2img is not None
        elif self.is_img2img:
            return self.img2img is not None
        elif self.is_pix2pix:
            return self.pix2pix is not None
        elif self.is_outpaint:
            return self.outpaint is not None
        elif self.is_depth2img:
            return self.depth2img is not None
        elif self.is_superresolution:
            return self.superresolution is not None
        elif self.is_controlnet:
            return self.controlnet is not None
        elif self.is_txt2vid:
            return self.txt2vid is not None
        elif self.is_upscale:
            return self.upscale is not None

    @property
    def pipe(self):
        if self.is_txt2img:
            return self.txt2img
        elif self.is_img2img:
            return self.img2img
        elif self.is_outpaint:
            return self.outpaint
        elif self.is_depth2img:
            return self.depth2img
        elif self.is_pix2pix:
            return self.pix2pix
        elif self.is_superresolution:
            return self.superresolution
        elif self.is_controlnet:
            return self.controlnet
        elif self.is_txt2vid:
            return self.txt2vid
        elif self.is_upscale:
            return self.upscale
        else:
            raise ValueError(f"Invalid action {self.action} unable to get pipe")

    @pipe.setter
    def pipe(self, value):
        if self.is_txt2img:
            self.txt2img = value
        elif self.is_img2img:
            self.img2img = value
        elif self.is_outpaint:
            self.outpaint = value
        elif self.is_depth2img:
            self.depth2img = value
        elif self.is_pix2pix:
            self.pix2pix = value
        elif self.is_superresolution:
            self.superresolution = value
        elif self.is_controlnet:
            self.controlnet = value
        elif self.is_txt2vid:
            self.txt2vid = value
        elif self.is_upscale:
            self.upscale = value
        else:
            raise ValueError(f"Invalid action {self.action} unable to set pipe")

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
    def cuda_is_available(self):
        return torch.cuda.is_available()

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

    @property
    def action_diffuser(self):
        from diffusers import (
            DiffusionPipeline,
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionUpscalePipeline,
            StableDiffusionControlNetPipeline,
            StableDiffusionLatentUpscalePipeline,
        )

        if self.is_txt2img:
            return StableDiffusionPipeline
        elif self.is_img2img:
            return StableDiffusionImg2ImgPipeline
        elif self.is_pix2pix:
            return StableDiffusionInstructPix2PixPipeline
        elif self.is_outpaint:
            return StableDiffusionInpaintPipeline
        elif self.is_depth2img:
            return StableDiffusionDepth2ImgPipeline
        elif self.is_superresolution:
            return StableDiffusionUpscalePipeline
        elif self.is_controlnet:
            return StableDiffusionControlNetPipeline
        elif self.is_txt2vid:
            return DiffusionPipeline
        elif self.is_upscale:
            return StableDiffusionLatentUpscalePipeline
        else:
            raise ValueError("Invalid action")

    @property
    def is_ckpt_model(self):
        return self._is_ckpt_file(self.model)

    @property
    def is_safetensors(self):
        return self._is_safetensor_file(self.model)

    @property
    def data_type(self):
        data_type = torch.half if self.cuda_is_available else torch.float
        data_type = torch.half if self.use_xformers else data_type
        return data_type

    @property
    def device(self):
        return "cuda" if self.cuda_is_available else "cpu"

    @property
    def controlnet_model(self):
        if self.controlnet_type == "canny":
            return "lllyasviel/control_v11p_sd15_canny"
        elif self.controlnet_type == "depth":
            return "lllyasviel/control_v11f1p_sd15_depth"
        elif self.controlnet_type == "mlsd":
            return "lllyasviel/control_v11p_sd15_mlsd"
        elif self.controlnet_type == "normal":
            return "lllyasviel/control_v11p_sd15_normalbae"
        elif self.controlnet_type == "scribble":
            return "lllyasviel/control_v11p_sd15_scribble"
        elif self.controlnet_type == "segmentation":
            return "lllyasviel/control_v11p_sd15_seg"
        elif self.controlnet_type == "lineart":
            return "lllyasviel/control_v11p_sd15_lineart"
        elif self.controlnet_type == "openpose":
            return "lllyasviel/control_v11p_sd15_openpose"
        elif self.controlnet_type == "softedge":
            return "lllyasviel/control_v11p_sd15_softedge"
        elif self.controlnet_type == "pixel2pixel":
            return "lllyasviel/control_v11e_sd15_ip2p"
        elif self.controlnet_type == "inpaint":
            return "lllyasviel/control_v11p_sd15_inpaint"
        elif self.controlnet_type == "shuffle":
            return "lllyasviel/control_v11e_sd15_shuffle"
        elif self.controlnet_type == "anime":
            return "lllyasviel/control_v11p_sd15s2_lineart_anime"

    @property
    def has_internet_connection(self):
        try:
            response = requests.get('https://huggingface.co/')
            return True
        except requests.ConnectionError:
            return False

    @property
    def txt2vid_file(self):
        return os.path.join(self.model_base_path, "videos", f"{self.prompt}_{self.seed}.mp4")

    @staticmethod
    def clear_memory():
        logger.info("Clearing memory")
        torch.cuda.empty_cache()
        gc.collect()

    def unload_unused_models(self, skip_model=None):
        """
        Unload all models except the one specified in skip_model
        :param skip_model: do not unload this model (typically the one currently in use)
        :return:
        """
        logger.info("Unloading existing model")
        do_clear_memory = False
        for model_type in [
            "txt2img",
            "img2img",
            "pix2pix",
            "outpaint",
            "depth2img",
            "superresolution",
            "controlnet",
            "txt2vid",
            "upscale",
        ]:
            if skip_model is None or skip_model != model_type:
                model = self.__getattribute__(model_type)
                if model is not None:
                    self.__setattr__(model_type, None)
                    do_clear_memory = True
        if do_clear_memory:
            self.clear_memory()

    def load_controlnet_from_ckpt(self, pipeline):
        from diffusers import ControlNetModel, UniPCMultistepScheduler
        from diffusers import StableDiffusionControlNetPipeline
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model,
            local_files_only=self.local_files_only,
            torch_dtype=self.data_type
        )
        pipeline.controlnet = controlnet
        pipeline = StableDiffusionControlNetPipeline(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            controlnet=controlnet,
            scheduler=pipeline.scheduler,
            safety_checker=pipeline.safety_checker,
            feature_extractor=pipeline.feature_extractor,
            requires_safety_checker=self.do_nsfw_filter,
        )
        if self.enable_model_cpu_offload:
            pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            pipeline.enable_model_cpu_offload()
        return pipeline

    def load_controlnet(self):
        from diffusers import ControlNetModel
        return ControlNetModel.from_pretrained(
            self.controlnet_model,
            local_files_only=self.local_files_only,
            torch_dtype=self.data_type
        )

    def load_controlnet_scheduler(self):
        if self.enable_model_cpu_offload:
            from diffusers import UniPCMultistepScheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_model_cpu_offload()

    def _load_ckpt_model(
        self, 
        path=None, 
        is_controlnet=False,
        is_safetensors=False,
        data_type=None,
        do_nsfw_filter=False,
        device=None,
        scheduler_name=None
    ):
        logger.debug(f"Loading ckpt file, is safetensors {is_safetensors}")
        if not data_type:
            data_type = self.data_type
        try:
            pipeline = self.download_from_original_stable_diffusion_ckpt(
                path=path,
                is_safetensors=is_safetensors,
                do_nsfw_filter=do_nsfw_filter,
                device=device,
                scheduler_name=scheduler_name
            )
            if is_controlnet:
                pipeline = self.load_controlnet_from_ckpt(pipeline)
        except Exception as e:
            print("Something went wrong loading the model file", e)
            self.error_handler("Unable to load ckpt file")
            raise e
        # to half
        # determine which data type to move the model to
        pipeline.vae.to(data_type)
        pipeline.text_encoder.to(data_type)
        pipeline.unet.to(data_type)
        if self.do_nsfw_filter:
            pipeline.safety_checker.half()
        return pipeline

    def download_from_original_stable_diffusion_ckpt(
        self, 
        config="v1.yaml",
        path=None,
        is_safetensors=False,
        scheduler_name=None,
        do_nsfw_filter=False,
        device=None
    ):
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import \
            download_from_original_stable_diffusion_ckpt
        print("is safetensors", is_safetensors)
        schedulers = {
            "Euler": "euler",
            "Euler a": "euler-ancestral",
            "LMS": "lms",
            "PNDM": "pndm",
            "Heun": "heun",
            "DDIM": "ddim",
            "DDPM": "DDPMScheduler",
            "DPM multistep": "dpm",
            "DPM singlestep": "dpmss",
            "DPM++ multistep": "dpm++",
            "DPM++ singlestep": "dpmss++",
            "DPM2 k": "dpm2k",
            "DPM2 a k": "dpm2ak",
            "DEIS": "deis",
        }
        if not scheduler_name:
            scheduler_name = self.scheduler_name
        if not path:
            path = f"{self.settings_manager.settings.model_base_path.get()}/{self.model}"
        if not device:
            device = self.device
        try:
            # check if config is a file
            if not os.path.exists(config):
                HERE = os.path.dirname(os.path.abspath(__file__))
                config = os.path.join(HERE, config)
            return download_from_original_stable_diffusion_ckpt(
                checkpoint_path=path,
                original_config_file=config,
                scheduler_type=schedulers[scheduler_name],
                device=device,
                from_safetensors=is_safetensors,
                load_safety_checker=do_nsfw_filter,
                local_files_only=self.local_files_only
            )
        # find exception: RuntimeError: Error(s) in loading state_dict for UNet2DConditionModel
        except RuntimeError as e:
            if e.args[0].startswith("Error(s) in loading state_dict for UNet2DConditionModel") and config  == "v1.yaml":
                logger.info("Failed to load model with v1.yaml config file, trying v2.yaml")
                return self.download_from_original_stable_diffusion_ckpt(
                    config="v2.yaml",
                    path=path,
                    is_safetensors=is_safetensors,
                    scheduler_name=scheduler_name,
                    do_nsfw_filter=do_nsfw_filter,
                    device=device
                )
            else:
                print("Something went wrong loading the model file", e)
                raise e

    def _load_model(self):
        logger.info("Loading model...")
        self.lora_loaded = False
        self.embeds_loaded = False
        if self.is_ckpt_model or self.is_safetensors:
            kwargs = {}
        else:
            kwargs = {
                "torch_dtype": self.data_type,
                "scheduler": self.scheduler,
                # "low_cpu_mem_usage": True, # default is already set to true
                "variant": self.current_model_branch
            }
            if self.current_model_branch:
                kwargs["variant"] = self.current_model_branch

        # move all models except for our current action to the CPU
        #self.unload_unused_models(skip_model=self.action)

        # special load case for img2img if txt2img is already loaded
        if self.is_img2img and self.txt2img is not None:
            self.img2img = self.action_diffuser(**self.txt2img.components)
        elif self.is_txt2img and self.img2img is not None:
            self.txt2img = self.action_diffuser(**self.img2img.components)
        elif self.pipe is None or self.reload_model:
            logger.debug("Loading model from scratch")
            if self.is_ckpt_model or self.is_safetensors:
                logger.debug("Loading ckpt or safetensors model")
                self.pipe = self._load_ckpt_model(
                    is_controlnet=self.is_controlnet,
                    is_safetensors=self.is_safetensors,
                    do_nsfw_filter=self.do_nsfw_filter
                )
            else:
                logger.debug("Loading from diffusers pipeline")
                if self.is_controlnet:
                    kwargs["controlnet"] = self.load_controlnet()
                if self.is_superresolution:
                    kwargs["low_res_scheduler"] = self.load_scheduler("DDPMScheduler")
                print(kwargs)
                self.pipe = self.action_diffuser.from_pretrained(
                    self.model_path,
                    local_files_only=self.local_files_only,
                    use_auth_token=self.data["options"]["hf_token"],
                    **kwargs
                )

            if self.is_controlnet:
                self.load_controlnet_scheduler()

            if hasattr(self.pipe, "safety_checker") and self.do_nsfw_filter:
                self.safety_checker = self.pipe.safety_checker

        # store the model_path
        self.pipe.model_path = self.model_path

        embeddings_folder = os.path.join(self.model_base_path, "embeddings")
        self.load_learned_embed_in_clip(embeddings_folder)

        self.apply_memory_efficient_settings()

    def load_learned_embed_in_clip(self, learned_embeds_path):
        if self.embeds_loaded:
            return
        self.embeds_loaded = True
        if os.path.exists(learned_embeds_path):
            logger.info("Loading embeddings...")
            tokens = []
            for f in os.listdir(learned_embeds_path):
                try:

                    text_encoder = self.pipe.text_encoder
                    tokenizer = self.pipe.tokenizer
                    token = None

                    loaded_learned_embeds = torch.load(os.path.join(learned_embeds_path, f), map_location="cpu")

                    # separate token and the embeds
                    trained_token = list(loaded_learned_embeds.keys())[0]
                    if trained_token == "string_to_token":
                        trained_token = loaded_learned_embeds["name"]
                    embeds = loaded_learned_embeds[trained_token]
                    tokens.append(trained_token)

                    # cast to dtype of text_encoder
                    # dtype = text_encoder.get_input_embeddings().weight.dtype
                    # embeds.to(dtype)

                    # add the token in tokenizer
                    token = token if token is not None else trained_token
                    num_added_tokens = tokenizer.add_tokens(token)
                    if num_added_tokens == 0:
                        raise ValueError(
                            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

                    # resize the token embeddings
                    text_encoder.resize_token_embeddings(len(tokenizer))
                    # embeds.shape == [768], convert it to [1024]
                    #embeds = torch.cat([embeds, torch.zeros(256, dtype=embeds.dtype)], dim=0)

                    # get the id for the token and assign the embeds
                    token_id = tokenizer.convert_tokens_to_ids(token)

                    try:
                        text_encoder.get_input_embeddings().weight.data[token_id] = embeds
                    except Exception as e:
                        logger.warning(e)

                    self.pipe.text_encoder = text_encoder
                    self.pipe.tokenizer = tokenizer
                except Exception as e:
                    logger.warning(e)
            self.settings_manager.settings.available_embeddings.set(", ".join(tokens))

    def apply_last_channels(self):
        if self.use_last_channels:
            logger.debug("Enabling torch.channels_last")
            self.pipe.unet.to(memory_format=torch.channels_last)
        else:
            logger.debug("Disabling torch.channels_last")
            self.pipe.unet.to(memory_format=torch.contiguous_format)

    def apply_vae_slicing(self):
        if self.action not in ["img2img", "depth2img", "pix2pix", "outpaint", "superresolution", "controlnet", "upscale"]:
            if self.use_enable_vae_slicing:
                logger.debug("Enabling vae slicing")
                self.pipe.enable_vae_slicing()
            else:
                logger.debug("Disabling vae slicing")
                self.pipe.disable_vae_slicing()

    def apply_attention_slicing(self):
        if self.use_attention_slicing:
            logger.debug("Enabling attention slicing")
            self.pipe.enable_attention_slicing()
        else:
            logger.debug("Disabling attention slicing")
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
        logger.debug("Moving to cpu")
        pipe.to("cpu", torch.float32)
        return pipe

    def apply_cpu_offload(self):
        if self.use_enable_sequential_cpu_offload and not self.enable_model_cpu_offload:
            logger.debug("Enabling sequential cpu offload")
            self.pipe = self.move_pipe_to_cpu(self.pipe)
            self.pipe.enable_sequential_cpu_offload()

    def apply_model_offload(self):
        if self.enable_model_cpu_offload and not self.use_enable_sequential_cpu_offload:
            logger.debug("Enabling model cpu offload")
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

    def _initialize(self):
        if not self.initialized or self.reload_model:
            logger.info("Initializing model")
            if self._previous_model != self.current_model:
                self.unload_unused_models(self.action)
            self._load_model()
            self.reload_model = False
            self.initialized = True

    def _is_ckpt_file(self, model):
        if not model:
            raise ValueError("ckpt path is empty")
        return model.endswith(".ckpt")

    def _is_safetensor_file(self, model):
        if not model:
            raise ValueError("safetensors path is empty")
        return model.endswith(".safetensors")

    def _do_reload_model(self):
        logger.info("Reloading model")
        if self.reload_model:
            self._load_model()

    def _prepare_model(self):
        logger.info("Prepare model")
        # get model and switch to it

        # get models from database
        model_name = self.options.get(f"{self.action}_model", None)

        self.set_message(f"Loading model {model_name}")

        self._previous_model = self.current_model

        if self._is_ckpt_file(model_name):
            self.current_model = model_name
        else:
            self.current_model = self.options.get(f"{self.action}_model_path", None)
            self.current_model_branch = self.options.get(f"{self.action}_model_branch", None)

    def _change_scheduler(self):
        if not self.do_change_scheduler:
            return
        if self.model_path and self.model_path != "":
            self.pipe.scheduler = self.scheduler
            self.do_change_scheduler = False
        else:
            logger.warning("Unable to change scheduler, model_path is not set")

    def _prepare_scheduler(self):
        scheduler_name = self.options.get(f"{self.action}_scheduler", "euler_a")
        if self.scheduler_name != scheduler_name:
            self.set_message(f"Preparing scheduler...")
            self.set_message("Loading scheduler")
            logger.info("Prepare scheduler")
            self.set_message("Preparing scheduler...")
            self.scheduler_name = scheduler_name
            if self.is_ckpt_model or self.is_safetensors:
                self.reload_model = True
            else:
                self.do_change_scheduler = True
        else:
            self.do_change_scheduler = False

    def _prepare_options(self, data):
        self.set_message(f"Preparing options...")
        try:
            action = data.get("action", "txt2img")
        except AttributeError:
            logger.error("No action provided")
            logger.error(data)
        options = data["options"]
        print(options)
        self.reload_model = False
        self.controlnet_type = self.options.get("controlnet", "canny")
        self.model_base_path = options["model_base_path"]
        model = options.get(f"{action}_model")
        if model != self.model:
            self.model = model
            self.reload_model = True
        controlnet_type = options.get(f"controlnet")
        if controlnet_type != self.controlnet_type:
            self.controlnet_type = controlnet_type
            self.reload_model = True
        self.prompt = options.get(f"{action}_prompt", self.prompt)
        self.negative_prompt = options.get(f"{action}_negative_prompt", self.negative_prompt)
        self.seed = int(options.get(f"{action}_seed", self.seed))
        self.guidance_scale = float(options.get(f"{action}_scale", self.guidance_scale))
        self.image_guidance_scale = float(options.get(f"{action}_image_scale", self.image_guidance_scale))
        self.strength = float(options.get(f"{action}_strength") or 1)
        self.num_inference_steps = int(options.get(f"{action}_steps", self.num_inference_steps))
        self.height = int(options.get(f"{action}_height", self.height))
        self.width = int(options.get(f"{action}_width", self.width))
        self.C = int(options.get(f"{action}_C", self.C))
        self.f = int(options.get(f"{action}_f", self.f))
        self.steps = int(options.get(f"{action}_steps", self.steps))
        self.ddim_eta = float(options.get(f"{action}_ddim_eta", self.ddim_eta))
        self.batch_size = int(options.get(f"{action}_n_samples", self.batch_size))
        self.n_samples = int(options.get(f"{action}_n_samples", self.n_samples))
        self.pos_x = int(options.get(f"{action}_pos_x", self.pos_x))
        self.pos_y = int(options.get(f"{action}_pos_y", self.pos_y))
        self.outpaint_box_rect = options.get(f"{action}_outpaint_box_rect", self.outpaint_box_rect)
        self.hf_token = ""
        self.enable_model_cpu_offload = options.get(f"enable_model_cpu_offload", self.enable_model_cpu_offload)
        self.use_attention_slicing = self.use_attention_slicing
        self.use_tf32 = self.use_tf32
        self.use_enable_vae_slicing = self.use_enable_vae_slicing
        self.use_xformers = self.use_xformers

        do_nsfw_filter = bool(options.get(f"do_nsfw_filter", self.do_nsfw_filter))
        self.do_nsfw_filter = do_nsfw_filter
        self.action = action
        self.options = options

        # memory settings
        self.use_last_channels = options.get("use_last_channels", True) == True
        cpu_offload = options.get("use_enable_sequential_cpu_offload", True) == True
        if self.is_pipe_loaded and cpu_offload != self.use_enable_sequential_cpu_offload:
            logger.debug("Reloading model based on cpu offload")
            self.reload_model = True
        self.use_enable_sequential_cpu_offload = cpu_offload
        self.use_attention_slicing = options.get("use_attention_slicing", True) == True
        self.use_tf32 = options.get("use_tf32", True) == True
        self.use_enable_vae_slicing = options.get("use_enable_vae_slicing", True) == True
        use_xformers = options.get("use_xformers", True) == True
        self.use_tiled_vae = options.get("use_tiled_vae", True) == True
        if self.is_pipe_loaded  and use_xformers != self.use_xformers:
            logger.debug("Reloading model based on xformers")
            self.reload_model = True
        self.use_xformers = use_xformers
        self.use_accelerated_transformers = options.get("use_accelerated_transformers", True) == True
        self.use_torch_compile = options.get("use_torch_compile", True) == True
        # print logger.info of all memory settings in use
        logger.debug("Memory settings:")
        logger.debug(f"  use_last_channels: {self.use_last_channels}")
        logger.debug(f"  use_enable_sequential_cpu_offload: {self.use_enable_sequential_cpu_offload}")
        logger.debug(f"  enable_model_cpu_offload: {self.enable_model_cpu_offload}")
        logger.debug(f"  use_tiled_vae: {self.use_tiled_vae}")
        logger.debug(f"  use_attention_slicing: {self.use_attention_slicing}")
        logger.debug(f"  use_tf32: {self.use_tf32}")
        logger.debug(f"  use_enable_vae_slicing: {self.use_enable_vae_slicing}")
        logger.debug(f"  use_xformers: {self.use_xformers}")
        logger.debug(f"  use_accelerated_transformers: {self.use_accelerated_transformers}")
        logger.debug(f"  use_torch_compile: {self.use_torch_compile}")

        self.options = options

        torch.backends.cuda.matmul.allow_tf32 = self.use_tf32

    def load_safety_checker(self, action):
        if not self.do_nsfw_filter:
            self.pipe.safety_checker = None
        else:
            self.pipe.safety_checker = self.safety_checker

    def do_sample(self, **kwargs):
        logger.info(f"Sampling {self.action}")
        self.set_message(f"Generating image...")
        # move everything but this action to the cpu
        # self.unload_unused_models(self.action)
        #if not self.is_ckpt_model and not self.is_safetensors:
        logger.info(f"Load safety checker")
        self.load_safety_checker(self.action)

        # self.apply_cpu_offload()
        try:
            if self.is_controlnet:
                logger.info(f"Setting up controlnet")
                #generator = torch.manual_seed(self.seed)
                kwargs["image"] = self._preprocess_for_controlnet(kwargs.get("image"), process_type=self.controlnet_type)
                #kwargs["generator"] = generator

                if kwargs.get("strength"):
                    kwargs["controlnet_conditioning_scale"] = kwargs["strength"]
                    del kwargs["strength"]

            logger.info(f"Generating image")
            output = self.call_pipe(**kwargs)
        except Exception as e:
            self.error_handler(e)
            if "`flshattF` is not supported because" in str(e):
                # try again
                logger.info("Disabling xformers and trying again")
                self.pipe.enable_xformers_memory_efficient_attention(attention_op=None)
                self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
                # redo the sample with xformers enabled
                return self.do_sample(**kwargs)
            output = None

        if self.is_txt2vid:
            return self.handle_txt2vid_output(output)
        else:
            image = output.images[0] if output else None
            nsfw_content_detected = None
            if output:
                if self.action_has_safety_checker:
                    try:
                        nsfw_content_detected = output.nsfw_content_detected
                    except AttributeError:
                        pass
            return image, nsfw_content_detected

    def enhance_video(self, video_frames):
        """
        Iterate over each video frame and call img2img on it using the same options that were passed
        and replace each frame with the enhanced version.
        :param video_frames: list of numpy arrays
        :return: video_frames: list of numpy arrays
        """
        new_video_frames = []
        for img in video_frames:
            pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.width, self.height), Image.LANCZOS)
            image, nsfw_detected = self.generator_sample({
                "action": "img2img",
                "options": {
                    "image": pil_image,
                    "mask": pil_image,
                    "img2img_prompt": self.prompt,
                    "img2img_negative_prompt": self.negative_prompt,
                    "img2img_steps": self.steps,
                    "img2img_ddim_eta": self.ddim_eta,
                    "img2img_n_iter": 1,
                    "img2img_width": self.width,
                    "img2img_height": self.height,
                    "img2img_n_samples": 20,
                    "img2img_strength": 0.5,
                    "img2img_scale": 7.5,
                    "img2img_seed": self.seed,
                    "img2img_model": "Stable diffusion V2",
                    "img2img_scheduler": self.scheduler_name,
                    "img2img_model_path": "stabilityai/stable-diffusion-2-1-base",
                    "img2img_model_branch": "fp16",
                    "width": self.width,
                    "height": self.height,
                    "do_nsfw_filter": self.do_nsfw_filter,
                    "model_base_path": self.model_base_path,
                    "pos_x": self.pos_x,
                    "pos_y": self.pos_y,
                    "outpaint_box_rect": self.outpaint_box_rect,
                    "hf_token": self.hf_token,
                    "enable_model_cpu_offload": self.enable_model_cpu_offload,
                    "use_attention_slicing": self.use_attention_slicing,
                    "use_tf32": self.use_tf32,
                    "use_enable_vae_slicing": self.use_enable_vae_slicing,
                    "use_xformers": self.use_xformers,
                    "use_accelerated_transformers": self.use_accelerated_transformers,
                    "use_torch_compile": self.use_torch_compile,
                    "use_tiled_vae": self.use_tiled_vae,
                }
            }, image_var=None, use_callback=False)
            if image:
                # convert to numpy array and add to new_video_frames
                new_video_frames.append(np.array(image))
        return new_video_frames if len(new_video_frames) > 0 else video_frames

    def handle_txt2vid_output(self, output):
        pil_image = None
        if output:
            from diffusers.utils import export_to_video
            video_frames = output.frames
            os.makedirs(os.path.dirname(self.txt2vid_file), exist_ok=True)
            #self.enhance_video(video_frames)
            export_to_video(video_frames, self.txt2vid_file)
            pil_image = Image.fromarray(video_frames[0])
        else:
            print("failed to get output from txt2vid")
        return pil_image, None

    '''
    TODO: extensions
    def call_pipe_extension(self, **kwargs):
        """
        This calls the call_pipe method on all active extensions
        :param kwargs:
        :return:
        """
        for extension in self.active_extensions:
            self.pipe = extension.call_pipe(self.options, self.model_base_path, self.pipe, **kwargs)
        return self.pipe
    '''

    def call_pipe(self, **kwargs):
        """
        Generate an image using the pipe
        :param kwargs:
        :return:
        """
        logger.info("Initialize compel")
        compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)
        logger.info("Initialize compel prompt")
        prompt_embeds = compel_proc(self.prompt)
        logger.info("Initialize compel negative prompt")
        negative_prompt_embeds = compel_proc(self.negative_prompt) if self.negative_prompt else None

        logger.info(f"is_txt2vid: {self.is_txt2vid}")

        if self.is_txt2vid:
            return self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                num_frames=self.batch_size,
                callback=self.callback,
                seed=self.seed,
            )
        elif self.is_upscale:
            return self.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=kwargs.get("image"),
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                callback=self.callback,
                generator=torch.manual_seed(self.seed)
            )
        elif self.is_superresolution:
            return self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=1,
                callback=self.callback,
                # cross_attention_kwargs={"scale": 0.5},
                **kwargs
            )
        else:
            # self.pipe = self.call_pipe_extension(**kwargs)  TODO: extensions
            if not self.lora_loaded:
                self.loaded_lora = []

            reload_lora = False
            if len(self.loaded_lora) > 0:
                # comparre lora in self.options[f"{self.action}_lora"] with self.loaded_lora
                # if the lora["name"] in options is not in self.loaded_lora, or lora["scale"] is different, reload lora
                for lora in self.options[f"{self.action}_lora"]:
                    lora_in_loaded_lora = False
                    for loaded_lora in self.loaded_lora:
                        if lora["name"] == loaded_lora["name"] and lora["scale"] == loaded_lora["scale"]:
                            lora_in_loaded_lora = True
                            break
                    if not lora_in_loaded_lora:
                        reload_lora = True
                        break
                if len(self.options[f"{self.action}_lora"]) != len(self.loaded_lora):
                    reload_lora = True

            if reload_lora:
                self.loaded_lora = []
                self.unload_unused_models()
                #self._load_model()
                return self.generator_sample(
                    self.data,
                    self._image_var,
                    self._error_var,
                    self._use_callback
                )
            
            if len(self.loaded_lora) == 0 and len(self.options[f"{self.action}_lora"]) > 0:
                self.apply_lora()
                self.lora_loaded = len(self.loaded_lora) > 0

            return self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=1,
                callback=self.callback,
                # cross_attention_kwargs={"scale": 0.5},
                **kwargs
            )
    
    def apply_lora(self):
        model_base_path = self.settings_manager.settings.model_base_path.get()
        lora_path = self.settings_manager.settings.lora_path.get() or "lora"
        path = os.path.join(model_base_path, lora_path) if lora_path == "lora" else lora_path
        for lora in self.options[f"{self.action}_lora"]:
            filepath = None
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.startswith(lora["name"]):
                        filepath = os.path.join(root, file)
                        break
            try:
                self.load_lora(filepath, multiplier=lora["scale"] / 100.0)
                self.loaded_lora.append({"name": lora["name"], "scale": lora["scale"]})
            except RuntimeError as e:
                print(e)
                print("Failed to load lora")

    # https://github.com/huggingface/diffusers/issues/3064
    def load_lora(self, checkpoint_path, multiplier=1.0, device="cuda", dtype=torch.float16):
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path, device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = self.pipe.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.pipe.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            try:
                alpha = elems['alpha']
                if alpha:
                    alpha = alpha.item() / weight_up.shape[1]
                else:
                    alpha = 1.0
            except KeyError:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(
                    weight_up.squeeze(3).squeeze(2),
                    weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                # print the shapes of weight_up and weight_down:
                # print(weight_up.shape, weight_down.shape)
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
    
    def _preprocess_canny(self, image):
        image = np.array(image)
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def _preprocess_depth(self, image):
        from transformers import pipeline
        depth_estimator = pipeline('depth-estimation')
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def _preprocess_hed(self, image):
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        image = hed(image)
        return image

    def _preprocess_mlsd(self, image):
        mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        image = mlsd(image)
        return image

    def _preprocess_normal(self, image):
        from transformers import pipeline
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        image = depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        bg_threhold = 0.4
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def _preprocess_segmentation(self, image):
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        return image

    def _preprocess_openpose(self, image):
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        image = openpose(image)
        return image

    def _preprocess_scribble(self, image):
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        image = hed(image, scribble=True)
        return image

    def _preprocess_for_controlnet(self, image, process_type="canny"):
        if process_type == "canny":
            image = self._preprocess_canny(image)
        elif process_type == "depth":
            image = self._preprocess_depth(image)
        elif process_type == "hed":
            image = self._preprocess_hed(image)
        elif process_type == "mlsd":
            image = self._preprocess_mlsd(image)
        elif process_type == "normal":
            image = self._preprocess_normal(image)
        elif process_type == "scribble":
            image = self._preprocess_scribble(image)
        elif process_type == "segmentation":
            image = self._preprocess_segmentation(image)
        elif process_type == "openpose":
            image = self._preprocess_openpose(image)
        return image

    def _sample_diffusers_model(self, data: dict):
        image = None
        nsfw_content_detected = None

        # disable warnings
        import warnings
        warnings.filterwarnings("ignore")
        from pytorch_lightning import seed_everything

        # disable info
        import logging
        logging.getLogger("lightning").setLevel(logging.WARNING)
        logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.WARNING)

        seed_everything(self.seed)
        action = self.action
        extra_args = {
        }

        if action == "txt2img":
            extra_args["width"] = self.width
            extra_args["height"] = self.height
        if action == "img2img":
            image = data["options"]["image"]
            extra_args["image"] = image
            extra_args["strength"] = self.strength
        elif action == "controlnet":
            image = data["options"]["image"]
            extra_args["image"] = image
            extra_args["strength"] = self.strength
        elif action == "pix2pix":
            image = data["options"]["image"]
            extra_args["image"] = image
            extra_args["image_guidance_scale"] = self.image_guidance_scale
        elif action == "depth2img":
            image = data["options"]["image"]
            # todo: get mask to work
            #mask_bytes = data["options"]["mask"]
            #mask = Image.frombytes("RGB", (self.width, self.height), mask_bytes)
            #extra_args["depth_map"] = mask
            extra_args["image"] = image
            extra_args["strength"] = self.strength
        elif action == "txt2vid":
            pass
        elif action == "upscale":
            image = data["options"]["image"]
            extra_args["image"] = image
            extra_args["image_guidance_scale"] = self.image_guidance_scale
        elif self.is_superresolution:
            image = data["options"]["image"]
            if self.do_mega_scale:
                pass
            else:
                extra_args["image"] = image
        elif action == "outpaint":
            image = data["options"]["image"]
            mask = data["options"]["mask"]
            extra_args["image"] = image
            extra_args["mask_image"] = mask
            extra_args["width"] = self.width
            extra_args["height"] = self.height

        # do the sample
        try:
            if self.do_mega_scale:
                # first we will downscale the original image using the PIL algorithm
                # called "bicubic" which is a high quality algorithm
                # then we will upscale the image using the super resolution model
                # then we will upscale the image using the PIL algorithm called "bicubic"
                # to the desired size
                # the new dimensions of scaled_w and scaled_h should be the width and height
                # of the image that current image but aspect ratio scaled to 128
                # so if the image is 256x256 then the scaled_w and scaled_h should be 128x128 but
                # if the image is 512x256 then the scaled_w and scaled_h should be 128x64

                max_in_width = 512
                scale_size = 256
                in_width = self.width
                in_height = self.height
                original_image_width = data["options"]["original_image_width"]
                original_image_height = data["options"]["original_image_height"]

                if original_image_width > max_in_width:
                    scale_factor = max_in_width / original_image_width
                    in_width = int(original_image_width * scale_factor)
                    in_height = int(original_image_height * scale_factor)
                    scale_size = int(scale_size * scale_factor)

                if in_width > max_in_width:
                    # scale down in_width and in_height by scale_size
                    # but keep the aspect ratio
                    in_width = scale_size
                    in_height = int((scale_size / original_image_width) * original_image_height)

                # now we will scale the image to the new dimensions
                # and then upscale it using the super resolution model
                # and then downscale it using the PIL bicubic algorithm
                # to the original dimensions
                # this will give us a high quality image
                scaled_w = int(in_width * (scale_size / in_height))
                scaled_h = scale_size
                downscaled_image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
                extra_args["image"] = downscaled_image
                upscaled_image = self.do_sample(**extra_args)
                # upscale back to self.width and self.height
                image = upscaled_image #.resize((original_image_width, original_image_height), Image.BILINEAR)

                return image
            else:
                image, nsfw_content_detected = self.do_sample(**extra_args)
        except Exception as e:
            if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
                raise Exception(self.cuda_error_message)
            elif "`flshattF` is not supported because" in str(e):
                # try again
                logger.info("Disabling xformers and trying again")
                self.pipe.enable_xformers_memory_efficient_attention(
                    attention_op=None)
                self.pipe.vae.enable_xformers_memory_efficient_attention(
                    attention_op=None)
                # redo the sample with xformers enabled
                return self._sample_diffusers_model(data)
            else:
                traceback.print_exc()
                self.error_handler("Something went wrong while generating image")
                logger.error(e)

        self.final_callback()

        return image, nsfw_content_detected

    def _blend_images_by_average(self, composite, original):
        # upscale the original image
        upscaled_image = original.resize((self.width * 4, self.height * 4), Image.BICUBIC)

        # blend the two images together using average pixel value of the two images
        blended_image = Image.blend(composite, upscaled_image, 0.5)
        return blended_image

    def _blend_images_with_mask(self, composite, original, alpha_amount=0.5):
        """
        1. Take the original image and upscale it using "bicubic"
        2. Create a mask that is 0 for the new image and 1 for the original image
        3. Blend the original image with the new image using the mask
        """
        # upscale the original image
        upscaled_image = original.resize((self.width * 4, self.height * 4), Image.BICUBIC)

        # both images have no alpha channel, they are RGB images
        # so we need to add an alpha channel to the composite image
        # so we can blend it with the original image
        composite = composite.convert("RGBA")

        # create a mask based on alpha_amount, where alpha_amount == 0 means the new image is used and
        # alpha_amount == 1 means the original image is used
        mask = Image.new("L", composite.size, int(255 * alpha_amount))

        # paste the mask into the composite image
        composite.putalpha(mask)

        # blend the composite image with the original image
        blended_image = Image.composite(composite, upscaled_image, mask)

        return blended_image

    def _generate(self, data: dict, image_var: ImageVar = None, use_callback: bool = True):
        logger.info("_generate called")
        self._prepare_options(data)
        self._prepare_scheduler()
        self._prepare_model()
        self._initialize()
        self._change_scheduler()

        self.apply_memory_efficient_settings()
        if self.is_txt2vid or self.is_upscale:
            total_to_generate = 1
        else:
            total_to_generate = self.batch_size
        for n in range(total_to_generate):
            image, nsfw_content_detected = self._sample_diffusers_model(data)
            if use_callback:
                self.image_handler(image, data, nsfw_content_detected)
            else:
                return image, nsfw_content_detected
            self.seed = self.seed + 1
            if self.do_cancel:
                self.do_cancel = False
                break

    def image_handler(self, image, data, nsfw_content_detected):
        if image:
            if self._image_handler:
                self._image_handler(image, data, nsfw_content_detected)
            elif self._image_var:
                self._image_var.set({
                    "image": image,
                    "data": data,
                    "nsfw_content_detected": nsfw_content_detected == True,
                })
            # self.save_pipeline()

    def final_callback(self):
        total = int(self.num_inference_steps * self.strength)
        self.tqdm_callback(total, total, self.action)

    def callback(self, step: int, _time_step, _latents):
        # convert _latents to image
        image = None
        if not self.is_txt2vid:
            image = self._latents_to_image(_latents)
        data = self.data
        if self.is_txt2vid:
            data["video_filename"] = self.txt2vid_file
        self.tqdm_callback(
            step,
            int(self.num_inference_steps * self.strength),
            self.action,
            image=image,
            data=data,
        )
        pass

    def _latents_to_image(self, latents: torch.Tensor):
        # convert tensor to image
        #image = self.pipe.vae.decoder(latents)
        image = latents.permute(0, 2, 3, 1)
        image = image.detach().cpu().numpy()
        image = image[0]
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def generator_sample(
        self,
        data: dict,
        image_var: callable,
        error_var: callable = None,
        use_callback: bool = True,
    ):
        self.data = data
        self._image_var = image_var
        self._error_var = error_var
        self._use_callback = use_callback
        self.set_message("Generating image...")

        action = "depth2img" if data["action"] == "depth" else data["action"]
        try:
            self.initialized =  self.__dict__[action] is not None
        except KeyError:
            self.initialized = False

        error = None
        try:
            self._generate(data, image_var=image_var, use_callback=use_callback)
        except OSError as e:
            err = e.args[0]
            logger.error(err)
            error = "model_not_found"
            err_obj = e
            traceback.print_exc() if self.is_dev_env else logger.error(err_obj)
        except TypeError as e:
            error = f"TypeError during generation {self.action}"
            traceback.print_exc() if self.is_dev_env else logger.error(e)
        except Exception as e:
            if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
                raise Exception(self.cuda_error_message)
            error = f"Error during generation"
            traceback.print_exc() if self.is_dev_env else logger.error(e)

        if error:
            self.initialized = False
            self.reload_model = True
            if error == "model_not_found" and self.local_files_only and self.has_internet_connection:
                # check if we have an internet connection
                self.set_message("Downloading model files...")
                self.local_files_only = False
                self._initialize()
                return self.generator_sample(data, image_var, error_var)
            elif not self.has_internet_connection:
                self.error_handler("Please check your internet connection and try again.")
            self.scheduler_name = None
            self._current_model = None
            self.local_files_only = True

            # handle the error (sends to client)
            self.error_handler(error)

    def cancel(self):
        self.do_cancel = True

    def merge_models(self, base_model_path, models_to_merge_path, weights, output_path, name, action):
        from diffusers import (
            DiffusionPipeline,
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionUpscalePipeline,
            StableDiffusionControlNetPipeline
        )
        # self.action = action
        # data = {
        #     "action": action,
        #     "options": {
        #         f"{action}_model": base_model_path,
        #         f"{action}_scheduler": "Euler a",
        #         f"{action}_model_branch": "fp16",
        #         f"model_base_path": self.model_path,
        #     }
        # }
        # self._prepare_options(data)
        # self._prepare_scheduler()
        # self._prepare_model()
        # print(data)
        # self._initialize()
        PipeCLS = StableDiffusionPipeline
        if action == "outpaint":
            PipeCLS = StableDiffusionInpaintPipeline
        elif action == "depth2img":
            PipeCLS = StableDiffusionDepth2ImgPipeline
        elif action == "pix2pix":
            PipeCLS = StableDiffusionInstructPix2PixPipeline

        print("LOADING PIPE FROM PRETRAINED", base_model_path)
        if base_model_path.endswith('.ckpt') or base_model_path.endswith('.safetensors'):
            pipe = self._load_ckpt_model(
                path=base_model_path,
                is_safetensors=base_model_path.endswith('.safetensors'),
                scheduler_name="Euler a"
            )
        else:
            pipe = PipeCLS.from_pretrained(
                base_model_path,
                local_files_only=self.local_files_only
            )
        for index in range(len(models_to_merge_path)):
            weight = weights[index]
            model_path = models_to_merge_path[index]
            print("LOADING MODEL TO MERGE FROM PRETRAINED", model_path)
            if model_path.endswith('.ckpt') or model_path.endswith('.safetensors'):
                model = self._load_ckpt_model(
                    path=model_path,
                    is_safetensors=model_path.endswith('.safetensors'),
                    scheduler_name="Euler a"
                )
            else:
                model = type(pipe).from_pretrained(
                    model_path,
                    local_files_only=self.local_files_only
                )

            pipe.vae = self.merge_vae(pipe.vae, model.vae, weight["vae"])
            pipe.unet = self.merge_unet(pipe.unet, model.unet, weight["unet"])
            pipe.text_encoder = self.merge_text_encoder(pipe.text_encoder, model.text_encoder, weight["text_encoder"])
        output_path = os.path.join(output_path, name)
        print(f"Saving to {output_path}")
        pipe.save_pretrained(output_path)
        print("merge complete")
    
    def merge_vae(self, vae_a, vae_b, weight_b=0.6):
        """
        Merge two VAE models by averaging their weights.

        Args:
            vae_a (nn.Module): First VAE model.
            vae_b (nn.Module): Second VAE model.
            weight_b (float): Weight to give to the second model. Default is 0.6.

        Returns:
            nn.Module: Merged VAE model.
        """
        # Get the state dictionaries of the two VAE models
        state_dict_a = vae_a.state_dict()
        state_dict_b = vae_b.state_dict()

        # Only merge parameters that have the same shape in both models
        merged_state_dict = {}
        for key in state_dict_a.keys():
            if key in state_dict_b and state_dict_a[key].shape == state_dict_b[key].shape:
                merged_state_dict[key] = (1 - weight_b) * state_dict_a[key] + weight_b * state_dict_b[key]
            else:
                print("shape does not match")
                merged_state_dict[key] = state_dict_a[key]

        # Load the merged state dictionary into a new VAE model
        merged_vae = type(vae_a)()
        vae_a.load_state_dict(merged_state_dict)

        return vae_a

    def merge_unet(self, unet_a, unet_b, weight_b=0.6):
        """
        Merge two U-Net models by averaging their weights.

        Args:
            unet_a (nn.Module): First U-Net model.
            unet_b (nn.Module): Second U-Net model.
            weight_b (float): Weight to give to the second model. Default is 0.6.

        Returns:
            nn.Module: Merged U-Net model.
        """
        # Get the state dictionaries of the two U-Net models
        state_dict_a = unet_a.state_dict()
        state_dict_b = unet_b.state_dict()

        # Average the weights of the two models, giving more weight to unet_b
        merged_state_dict = {}
        for key in state_dict_a.keys():
            if key in state_dict_b and state_dict_a[key].shape == state_dict_b[key].shape:
                merged_state_dict[key] = (1 - weight_b) * state_dict_a[key] + weight_b * state_dict_b[key]
            else:
                print("shape does not match")
                merged_state_dict[key] = state_dict_a[key]

        # Load the averaged weights into a new U-Net model
        merged_unet = type(unet_a)()
        unet_a.load_state_dict(merged_state_dict)

        return unet_a

    def merge_text_encoder(self, text_encoder_a, text_encoder_b, weight_b=0.6):
        """
        Merge two Text Encoder models by averaging their weights.

        Args:
            text_encoder_a (nn.Module): First Text Encoder model.
            text_encoder_b (nn.Module): Second Text Encoder model.
            weight_b (float): Weight to give to the second model. Default is 0.6.

        Returns:
            nn.Module: Merged Text Encoder model.
        """
        # Get the state dictionaries of the two Text Encoder models
        state_dict_a = text_encoder_a.state_dict()
        state_dict_b = text_encoder_b.state_dict()

        # Average the weights of the two models, giving more weight to text_encoder_b
        merged_state_dict = {}
        for key in state_dict_a.keys():
            if key in state_dict_b and state_dict_a[key].shape == state_dict_b[key].shape:
                merged_state_dict[key] = (1 - weight_b) * state_dict_a[key] + weight_b * state_dict_b[key]
            else:
                print("shape does not match")
                merged_state_dict[key] = state_dict_a[key]

        # Load the averaged weights into a new Text Encoder model
        text_encoder_a.load_state_dict(merged_state_dict)

        return text_encoder_a
