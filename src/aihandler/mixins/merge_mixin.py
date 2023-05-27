import os
from aihandler.settings import LOG_LEVEL
from aihandler.logger import logger
import logging
logging.disable(LOG_LEVEL)
logger.set_level(logger.DEBUG)


class MergeMixin:
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
        self.action = action
        # data = {
        #     "action": action,
        #     "options": {
        #         f"{action}_model": base_model_path,
        #         f"{action}_scheduler": "Euler a",
        #         f"{action}_model_branch": "fp16",
        #         f"model_base_path": self.model_path,
        #     }
        # }
        # self.prepare_options(data)
        # self._prepare_scheduler()
        # self._prepare_model()
        # print(data)
        # self.initialize()
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