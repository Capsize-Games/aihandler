import os
import gc
import numpy as np
import requests
from aihandler.base_runner import BaseRunner
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
from compel import Compel
from aihandler.settings import AVAILABLE_SCHEDULERS_BY_ACTION
from aihandler.mixins.merge_mixin import MergeMixin
from aihandler.mixins.lora_mixin import LoraMixin
from aihandler.mixins.controlnet_mixin import ControlnetMixin
from aihandler.mixins.memory_efficient_mixin import MemoryEfficientMixin
from aihandler.mixins.embeddings_mixin import EmbeddingsMixin
from aihandler.mixins.txttovideo_mixin import TexttovideoMixin

os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class SDRunner(
    BaseRunner,
    MergeMixin,
    LoraMixin,
    ControlnetMixin,
    MemoryEfficientMixin,
    EmbeddingsMixin,
    TexttovideoMixin
):
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
    superresolution = None
    txt2vid = None
    upscale = None
    state = None
    local_files_only = True
    lora_loaded = False
    loaded_lora = []
    _settings = None
    _action = None
    do_change_scheduler = False
    embeds_loaded = False
    options = {}
    _compel_proc = None
    _prompt_embeds = None
    _negative_prompt_embeds = None
    _scheduler = None

    @property
    def compel_proc(self):
        if not self._compel_proc:
            self._compel_proc = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                truncate_long_prompts=False
            )
        return self._compel_proc

    @compel_proc.setter
    def compel_proc(self, value):
        self._compel_proc = value

    @property
    def prompt_embeds(self):
        if self._prompt_embeds is None:
            self.load_prompt_embeds()
        return self._prompt_embeds

    @prompt_embeds.setter
    def prompt_embeds(self, value):
        self._prompt_embeds = value

    @property
    def negative_prompt_embeds(self):
        if self._negative_prompt_embeds is None:
            self.load_prompt_embeds()
        return self._negative_prompt_embeds

    @negative_prompt_embeds.setter
    def negative_prompt_embeds(self, value):
        self._negative_prompt_embeds = value

    def load_prompt_embeds(self):
        logger.info("Loading prompt embeds")
        self.compel_proc = None
        self.clear_memory()
        prompt = self.prompt
        negative_prompt = self.negative_prompt if self.negative_prompt else ""
        prompt_embeds = self.compel_proc.build_conditioning_tensor(prompt)
        negative_prompt_embeds = self.compel_proc.build_conditioning_tensor(negative_prompt)
        [prompt_embeds, negative_prompt_embeds] = self.compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds

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
    def cuda_is_available(self):
        return torch.cuda.is_available()

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
    def has_internet_connection(self):
        try:
            response = requests.get('https://huggingface.co/')
            return True
        except requests.ConnectionError:
            return False

    @staticmethod
    def clear_memory():
        logger.info("Clearing memory")
        torch.cuda.empty_cache()
        gc.collect()

    def load_scheduler(self, force_scheduler_name=None):
        import diffusers
        if not force_scheduler_name and self._scheduler and not self.do_change_scheduler:
            return self._scheduler

        if not self.model_path or self.model_path == "":
            traceback.print_stack()
            raise Exception("Chicken / egg problem, model path not set")

        if self.is_ckpt_model or self.is_safetensors:  # skip scheduler for ckpt models
            return None

        scheduler_name = force_scheduler_name if force_scheduler_name else self.scheduler_name
        if not force_scheduler_name and scheduler_name not in AVAILABLE_SCHEDULERS_BY_ACTION[self.action]:
            scheduler_name = AVAILABLE_SCHEDULERS_BY_ACTION[self.action][0]
        scheduler_class_name = self.schedulers[scheduler_name]
        scheduler_class = getattr(diffusers, scheduler_class_name)
        kwargs = {
            "subfolder": "scheduler"
        }
        # check if self.scheduler_name contains ++
        if scheduler_name.startswith("DPM"):
            kwargs["lower_order_final"] = self.num_inference_steps < 15
            if scheduler_name.find("++") != -1:
                kwargs["algorithm_type"] = "dpmsolver++"
            else:
                kwargs["algorithm_type"] = "dpmsolver"
        if self.current_model_branch:
            kwargs["variant"] = self.current_model_branch
        logger.info(f"Loading scheduler {self.scheduler_name} with kwargs {kwargs}")
        self._scheduler = scheduler_class.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            use_auth_token=self.data["options"]["hf_token"],
            **kwargs
        )
        return self._scheduler

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
            print("Path", path)
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
        from diffusers import StableDiffusionImg2ImgPipeline
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
            print("path", path)
            return download_from_original_stable_diffusion_ckpt(
                checkpoint_path=path,
                original_config_file=config,
                scheduler_type=schedulers[scheduler_name],
                device=device,
                from_safetensors=is_safetensors,
                load_safety_checker=do_nsfw_filter,
                local_files_only=self.local_files_only,
                pipeline_class=StableDiffusionImg2ImgPipeline if self.is_controlnet else self.action_diffuser
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
        if not self.initialized or self.reload_model:
            self.unload_unused_models()

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
                    kwargs["low_res_scheduler"] = self.load_scheduler("DDPM")
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

        self.load_learned_embed_in_clip()
        self.apply_memory_efficient_settings()

    def _initialize(self):
        if not self.initialized or self.reload_model:
            logger.info("Initializing model")
            self.compel_proc = None
            self.prompt_embeds = None
            self.negative_prompt_embeds = None
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
        if self.prompt != options.get(f"{action}_prompt") or self.negative_prompt != options.get(f"{action}_negative_prompt"):
            self._prompt_embeds = None
            self._negative_prompt_embeds = None
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

    def call_pipe(self, **kwargs):
        """
        Generate an image using the pipe
        :param kwargs:
        :return:
        """
        if self.is_txt2vid:
            return self.pipe(
                prompt_embeds=self.prompt_embeds,
                negative_prompt_embeds=self.negative_prompt_embeds,
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
                prompt_embeds=self.prompt_embeds,
                negative_prompt_embeds=self.negative_prompt_embeds,
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
                prompt_embeds=self.prompt_embeds,
                negative_prompt_embeds=self.negative_prompt_embeds,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=1,
                callback=self.callback,
                # cross_attention_kwargs={"scale": 0.5},
                **kwargs
            )
    
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
                self.error_handler(self.cuda_error_message)
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

    def _generate(self, data: dict, image_var: ImageVar = None, use_callback: bool = True):
        logger.info("_generate called")
        self.do_cancel = False
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
                error = self.cuda_error_message
                self.clear_memory()
            else:
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
