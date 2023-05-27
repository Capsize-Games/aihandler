import os
import torch
from aihandler.logger import logger


class EmbeddingsMixin:
    def load_learned_embed_in_clip(self):
        learned_embeds_path = os.path.join(self.model_base_path, "embeddings")
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