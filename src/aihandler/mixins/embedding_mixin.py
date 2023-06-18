import os
import torch
from aihandler.logger import logger


class EmbeddingMixin:
    def load_learned_embed_in_clip(self):
        learned_embeds_path = self.settings_manager.settings.embeddings_path.get()
        if not os.path.exists(learned_embeds_path):
            learned_embeds_path = os.path.join(self.model_base_path, "embeddings")
        if self.embeds_loaded:
            return
        self.embeds_loaded = True
        if os.path.exists(learned_embeds_path):
            logger.info("Loading embeddings...")
            tokens = []
            for f in os.listdir(learned_embeds_path):
                logger.info("Loading " + os.path.join(learned_embeds_path, f))
                path = os.path.join(learned_embeds_path, f)
                token = f.split(".")[0]
                self.pipe.load_textual_inversion(path, token=token)
            self.settings_manager.settings.available_embeddings.set(", ".join(tokens))