import cv2
import numpy as np
from aihandler.controlnet_utils import ade_palette
import torch
from aihandler.settings import LOG_LEVEL
from aihandler.logger import logger
import logging
logging.disable(LOG_LEVEL)
logger.set_level(logger.DEBUG)
from PIL import Image
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector


class ControlnetMixin:
    controlnet = None
    controlnet_type = "canny"

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

    def load_controlnet_from_ckpt(self, pipeline):
        from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline
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