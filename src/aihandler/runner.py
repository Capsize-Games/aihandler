import os
import gc
import re

import numpy as np
import requests
from aihandler.base_runner import BaseRunner
from aihandler.mixins.kandinsky_mixin import KandinskyMixin
from aihandler.qtvar import ImageVar
import traceback
import torch
from aihandler.logger import logger
from PIL import Image
from aihandler.mixins.merge_mixin import MergeMixin
from aihandler.mixins.lora_mixin import LoraMixin
from aihandler.mixins.controlnet_mixin import ControlnetMixin
from aihandler.mixins.memory_efficient_mixin import MemoryEfficientMixin
from aihandler.mixins.embedding_mixin import EmbeddingMixin
from aihandler.mixins.txttovideo_mixin import TexttovideoMixin
from aihandler.mixins.compel_mixin import CompelMixin
from aihandler.mixins.scheduler_mixin import SchedulerMixin
from aihandler.mixins.model_mixin import ModelMixin
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class SDRunner(
    BaseRunner,
    MergeMixin,
    LoraMixin,
    ControlnetMixin,
    MemoryEfficientMixin,
    EmbeddingMixin,
    TexttovideoMixin,
    CompelMixin,
    SchedulerMixin,
    ModelMixin,
    KandinskyMixin
):
    _current_model: str = ""
    _previous_model: str = ""
    initialized: bool = False
    _current_sample = 0
    _reload_model: bool = False
    do_cancel = False
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
    embeds_loaded = False
    _compel_proc = None
    _prompt_embeds = None
    _negative_prompt_embeds = None
    _data = {
        "options": {}
    }
    _model = None
    _controlnet_type = None
    reload_model = False

    @property
    def current_sample(self):
        return self._current_sample

    @current_sample.setter
    def current_sample(self, value):
        self._current_sample = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def options(self):
        return self.data.get("options", {})

    @property
    def seed(self):
        return self.options.get(f"{self.action}_seed", 42) + self.current_sample

    @staticmethod
    def convert_prompt_weights(prompt):
        """
        This function converts prompts in the style of automatic1111 to compel style prompts
        :param prompt:
        :return:
        """
        if not prompt:
            return prompt
        # prompt == "Example (ABC): 1.23 XYZ (DEF) (GHI:2.3)"
        # we only want to find (ABC) and (DEF) in prompt
        pattern = r"\([A-Z]+\)"
        matches = re.findall(pattern, prompt)
        # matches == ["(DEF)"]
        for match in matches:
            # replace matches with this:
            # matches == ["(DEF:1.1)"]
            # inside of prompt
            replaced_match = match.replace(")", ":1.1)")
            prompt = prompt.replace(match, replaced_match)

        # now prompt == "Example (ABC:1.1): 1.23 XYZ (DEF:1.1) (GHI:2.3)"
        # but we want it to look like this: "Example (ABC)1.1: 1.23 XYZ (DEF)1.1 (GHI)2.3"
        # so we replace all ":[0-9.]+)" with ")/1"
        pattern = r"\(([^:]+):([0-9.]+)\)"
        matches = re.findall(pattern, prompt)
        for match in matches:
            prompt = prompt.replace(f"({match[0]}:{match[1]})", f"({match[0]}){match[1]}")
        return prompt

    @property
    def prompt(self):
        prompt = self.options.get(f"{self.action}_prompt")
        if self.use_prompt_converter:
            prompt = SDRunner.convert_prompt_weights(prompt)
        return prompt

    @property
    def negative_prompt(self):
        negative_prompt = self.options.get(f"{self.action}_negative_prompt")
        if self.use_prompt_converter:
            negative_prompt = SDRunner.convert_prompt_weights(negative_prompt)
        return negative_prompt

    @property
    def use_prompt_converter(self):
        return True

    @property
    def guidance_scale(self):
        return self.options.get(f"{self.action}_scale", 7.5)

    @property
    def image_guidance_scale(self):
        return self.options.get(f"{self.action}_image_scale", 1.5)

    @property
    def height(self):
        return self.options.get(f"{self.action}_height", 512)

    @property
    def width(self):
        return self.options.get(f"{self.action}_width", 512)

    @property
    def steps(self):
        return self.options.get(f"{self.action}_steps", 20)

    @property
    def ddim_eta(self):
        return self.options.get(f"{self.action}_ddim_eta", 0.5)

    @property
    def batch_size(self):
        return self.options.get(f"{self.action}_n_samples", 1)

    @property
    def pos_x(self):
        return self.options.get(f"{self.action}_pos_x", 0)

    @property
    def pos_y(self):
        return self.options.get(f"{self.action}_pos_y", 0)

    @property
    def outpaint_box_rect(self):
        return self.options.get(f"{self.action}_box_rect", "")

    @property
    def hf_token(self):
        return self.data.get("hf_token", "")

    @property
    def strength(self):
        return self.options.get(f"{self.action}_strength", 1)

    @property
    def enable_model_cpu_offload(self):
        return self.options.get("enable_model_cpu_offload", False) == True

    @property
    def use_attention_slicing(self):
        return self.options.get("use_attention_slicing", False) == True

    @property
    def use_tf32(self):
        return self.options.get("use_tf32", False) == True

    @property
    def use_last_channels(self):
        return self.options.get("use_last_channels", True) == True

    @property
    def use_enable_sequential_cpu_offload(self):
        return self.options.get("use_enable_sequential_cpu_offload", True) == True

    @property
    def use_enable_vae_slicing(self):
        return self.options.get("use_enable_vae_slicing", False) == True

    @property
    def do_nsfw_filter(self):
        return self.options.get("do_nsfw_filter", True) == True

    @property
    def use_compel(self):
        return not self.use_enable_sequential_cpu_offload and not self.is_txt2vid

    @property
    def use_tiled_vae(self):
        if self.use_kandinsky:
            return False
        return self.options.get("use_tiled_vae", False) == True

    @property
    def use_accelerated_transformers(self):
        if self.use_kandinsky:
            return False
        return self.options.get("use_accelerated_transformers", False) == True

    @property
    def use_torch_compile(self):
        return self.options.get("use_torch_compile", False) == True

    @property
    def controlnet_type(self):
        return self.options.get("controlnet", "canny")

    @property
    def model_base_path(self):
        return self.options.get("model_base_path", None)

    @property
    def model(self):
        return self.options.get(f"{self.action}_model", None)

    @property
    def do_mega_scale(self):
        #return self.is_superresolution
        return False

    @property
    def action(self):
        return self.data.get("action", None)

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
    def use_interpolation(self):
        return self.options.get("use_interpolation", False)

    @property
    def interpolation_data(self):
        return self.options.get("interpolation_data", None)

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
    def cuda_error_message(self):
        if self.is_superresolution and self.scheduler_name == "DDIM":
            return f"Unable to run the model at {self.width}x{self.height} resolution using the DDIM scheduler. Try changing the scheduler to LMS or PNDM and try again."

        return f"You may not have enough GPU memory to run the model at {self.width}x{self.height} resolution. Potential solutions: try again, restart the application, use a smaller size, upgrade your GPU."

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

    def initialize(self):
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

    def prepare_options(self, data):
        self.set_message(f"Preparing options...")
        action = data["action"]
        options = data["options"]
        model = options.get(f"{action}_model", None)
        controlnet_type = options.get(f"{action}_controlnet", None)

        # do model reload checks here
        if (
            self.is_pipe_loaded and (  # memory options change
                self.use_enable_sequential_cpu_offload != options.get("use_enable_sequential_cpu_offload", True)
            )
        ) or (  # model change
            self.model is not None
            and self.model != model
            and model is not None
        ) or (  # controlnet change
            self.controlnet_type is not None
            and self.controlnet_type != controlnet_type
            and controlnet_type is not None
        ):
           self.reload_model = True

        if self.prompt != options.get(f"{action}_prompt") or \
           self.negative_prompt != options.get(f"{action}_negative_prompt") or \
           action != self.action or \
           self.reload_model:
            self._prompt_embeds = None
            self._negative_prompt_embeds = None

        self.data = data
        if not self.use_kandinsky:
            torch.backends.cuda.matmul.allow_tf32 = self.use_tf32

    def load_safety_checker(self, action):
        if not self.do_nsfw_filter:
            self.pipe.safety_checker = None
        else:
            self.pipe.safety_checker = self.safety_checker

    def do_sample(self, **kwargs):
        logger.info(f"Sampling {self.action}")
        self.set_message(f"Generating image...")

        if not self.use_kandinsky:
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
            error_message = str(e)
            if "Scheduler.step() got an unexpected keyword argument" in str(e):
                error_message = "Invalid scheduler"
                self.clear_scheduler()
            self.log_error(error_message)
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

    # def add_lora_to_pipe(self):
    #     # if not self.lora_loaded:
    #     #     self.loaded_lora = []
    #     #
    #     # reload_lora = False
    #     # if len(self.loaded_lora) > 0:
    #     #     # comparre lora in self.options[f"{self.action}_lora"] with self.loaded_lora
    #     #     # if the lora["name"] in options is not in self.loaded_lora, or lora["scale"] is different, reload lora
    #     #     for lora in self.options[f"{self.action}_lora"]:
    #     #         lora_in_loaded_lora = False
    #     #         for loaded_lora in self.loaded_lora:
    #     #             if lora["name"] == loaded_lora["name"] and lora["scale"] == loaded_lora["scale"]:
    #     #                 lora_in_loaded_lora = True
    #     #                 break
    #     #         if not lora_in_loaded_lora:
    #     #             reload_lora = True
    #     #             break
    #     #     if len(self.options[f"{self.action}_lora"]) != len(self.loaded_lora):
    #     #         reload_lora = True
    #     #
    #     # if reload_lora:
    #     #     self.loaded_lora = []
    #     #     self.unload_unused_models()
    #     #     # self._load_model()
    #     #     return self.generator_sample(
    #     #         self.data,
    #     #         self._image_var,
    #     #         self._error_var,
    #     #         self._use_callback
    #     #     )
    #     #
    #     # if len(self.loaded_lora) == 0 and len(self.options[f"{self.action}_lora"]) > 0:
    #     #     self.apply_lora()
    #     #     self.lora_loaded = len(self.loaded_lora) > 0
    #     return

    def call_pipe(self, **kwargs):
        """
        Generate an image using the pipe
        :param kwargs:
        :return:
        """
        args = {
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "callback": self.callback,
        }
        if not self.use_kandinsky and not self.is_txt2vid and not self.is_upscale and not self.is_superresolution:
            # self.pipe = self.call_pipe_extension(**kwargs)  TODO: extensions
            self.add_lora_to_pipe()
        if self.is_upscale:
            args["prompt"] = self.prompt
            args["negative_prompt"] = self.negative_prompt
            args["image"] = kwargs.get("image")
            args["generator"] = torch.manual_seed(self.seed)
        elif self.is_txt2vid:
            args["num_frames"] = self.batch_size
            args["prompt"] = self.prompt
            args["negative_prompt"] = self.negative_prompt
        elif not self.use_kandinsky:
            if self.use_compel:
                args["prompt_embeds"] = self.prompt_embeds
                args["negative_prompt_embeds"] = self.negative_prompt_embeds
            else:
                args["prompt"] = self.prompt
                args["negative_prompt"] = self.negative_prompt
        if not self.is_upscale:
            args.update(kwargs)
        if self.use_kandinsky:
            return self.kandinsky_call_pipe(**kwargs)
        if len(self.options[f"{self.action}_lora"]) > 0:
            args["cross_attention_kwargs"] = {"scale": 1.0}
        return self.pipe(**args)

    def prepare_extra_args(self, data, image, mask):
        action = self.action
        extra_args = {
        }
        if action == "txt2img":
            extra_args["width"] = self.width
            extra_args["height"] = self.height
        if action == "img2img":
            extra_args["image"] = image
            extra_args["strength"] = self.strength
        elif action == "controlnet":
            extra_args["image"] = image
            extra_args["strength"] = self.strength
        elif action == "pix2pix":
            extra_args["image"] = image
            extra_args["image_guidance_scale"] = self.image_guidance_scale
        elif action == "depth2img":
            extra_args["image"] = image
            extra_args["strength"] = self.strength
        elif action == "txt2vid":
            pass
        elif action == "upscale":
            extra_args["image"] = image
            extra_args["image_guidance_scale"] = self.image_guidance_scale
        elif self.is_superresolution:
            extra_args["image"] = image
        elif action == "outpaint":
            extra_args["image"] = image
            extra_args["mask_image"] = mask
            extra_args["width"] = self.width
            extra_args["height"] = self.height
        return extra_args

    def sample_diffusers_model(self, data: dict):
        from pytorch_lightning import seed_everything
        image = data["options"].get("image", None)
        mask = data["options"].get("mask", None)
        nsfw_content_detected = None
        seed_everything(self.seed)
        extra_args = self.prepare_extra_args(data, image, mask)

        # do the sample
        try:
            if self.do_mega_scale:
                return self.do_mega_scale_sample(data, image, extra_args)
            else:
                image, nsfw_content_detected = self.do_sample(**extra_args)
        except Exception as e:
            if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
                self.log_error(self.cuda_error_message)
            else:
                self.log_error(e, "Something went wrong while generating image")

        self.final_callback()

        return image, nsfw_content_detected

    def do_mega_scale_sample(self, data, image, extra_args):
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
        image = upscaled_image  # .resize((original_image_width, original_image_height), Image.BILINEAR)

        return image

    def generate(self, data: dict, image_var: ImageVar = None, use_callback: bool = True):
        logger.info("generate called")
        self.do_cancel = False
        self.prepare_options(data)
        if self.do_clear_kandinsky:
            self.clear_kandinsky()
        self._prepare_scheduler()
        self._prepare_model()
        self.initialize()
        self._change_scheduler()

        if not self.use_kandinsky:
            self.set_message(f"Applying memory settings...")
            self.apply_memory_efficient_settings()
        if self.is_txt2vid or self.is_upscale:
            total_to_generate = 1
        else:
            total_to_generate = self.batch_size
        for n in range(total_to_generate):
            self.current_sample = n
            image, nsfw_content_detected = self.sample_diffusers_model(data)
            if use_callback:
                self.image_handler(image, data, nsfw_content_detected)
            else:
                return image, nsfw_content_detected
            if self.do_cancel:
                self.do_cancel = False
                break
        self.current_sample = 0

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
        total = int(self.steps * self.strength)
        self.tqdm_callback(total, total, self.action)

    def callback(self, step: int, _time_step, _latents):
        # convert _latents to image
        image = None
        if not self.is_txt2vid:
            image = self.latents_to_image(_latents)
        data = self.data
        if self.is_txt2vid:
            data["video_filename"] = self.txt2vid_file
        self.tqdm_callback(
            step,
            int(self.steps * self.strength),
            self.action,
            image=image,
            data=data,
        )
        pass

    def latents_to_image(self, latents: torch.Tensor):
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
        error_message = ""
        try:
            self.generate(data, image_var=image_var, use_callback=use_callback)
        except OSError as e:
            error_message = "model_not_found"
            error = e
        except TypeError as e:
            error_message = f"TypeError during generation {self.action}"
            error = e
        except Exception as e:
            error = e
            if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
                error_message = self.cuda_error_message
                self.clear_memory()
            else:
                error_message = f"Error during generation"

        if error:
            self.log_error(error, error_message)
            self.initialized = False
            self.reload_model = True
            if error_message == "model_not_found" and self.local_files_only and self.has_internet_connection:
                # check if we have an internet connection
                self.set_message("Downloading model files...")
                self.local_files_only = False
                self.initialize()
                return self.generator_sample(data, image_var, error_var)
            elif not self.has_internet_connection:
                self.log_error("Please check your internet connection and try again.")
            self.scheduler_name = None
            self._current_model = None
            self.local_files_only = True

            # handle the error (sends to client)
            self.log_error(error)

    def cancel(self):
        self.do_cancel = True

    def log_error(self, error, message=None):
        message = str(error) if not message else message
        traceback.print_exc()
        self.error_handler(message)
