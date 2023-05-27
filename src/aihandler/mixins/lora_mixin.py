import os
import torch
from aihandler.settings import LOG_LEVEL
from aihandler.logger import logger
import logging
logging.disable(LOG_LEVEL)
logger.set_level(logger.DEBUG)
from collections import defaultdict
from safetensors.torch import load_file


class LoraMixin:
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