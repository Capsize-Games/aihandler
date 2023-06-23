import os
from aihandler.logger import logger


class ModelMixin:
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
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=path,
                original_config_file=config,
                device=device,
                scheduler_type="ddim",
                from_safetensors=is_safetensors,
                load_safety_checker=do_nsfw_filter,
                local_files_only=self.local_files_only,
                pipeline_class=StableDiffusionImg2ImgPipeline if self.is_controlnet else self.action_diffuser
            )
            pipe.scheduler = self.load_scheduler(config=pipe.scheduler.config)
            return pipe
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
        self.torch_compile_applied = False
        self.lora_loaded = False
        self.embeds_loaded = False
        if self.is_ckpt_model or self.is_safetensors:
            kwargs = {}
        else:
            kwargs = {
                "torch_dtype": self.data_type,
                "scheduler": self.load_scheduler(),
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
            logger.info(f"Loading model from scratch {self.reload_model}")
            if self.use_kandinsky:
                return
            if self.is_ckpt_model or self.is_safetensors:
                logger.info("Loading ckpt or safetensors model")
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
                    kwargs["low_res_scheduler"] = self.load_scheduler(force_scheduler_name="DDPM")
                self.pipe = self.action_diffuser.from_pretrained(
                    self.model_path,
                    local_files_only=self.local_files_only,
                    use_auth_token=self.data["options"]["hf_token"],
                    **kwargs
                )
                if self.is_upscale:
                    self.pipe.scheduler = self.load_scheduler(force_scheduler_name="Euler")

            if self.is_controlnet:
                self.load_controlnet_scheduler()

            if hasattr(self.pipe, "safety_checker") and self.do_nsfw_filter:
                self.safety_checker = self.pipe.safety_checker

        # store the model_path
        self.pipe.model_path = self.model_path

        self.load_learned_embed_in_clip()

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