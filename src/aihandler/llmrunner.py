import gc
import os
import threading
import time
import numpy as np
import random
import torch
from aihandler.base_runner import BaseRunner
from aihandler.settings import TEXT_MODELS as MODELS
from aihandler.logger import logger
from aihandler.settings import LOG_LEVEL
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
logger.set_level(LOG_LEVEL)


class LLMRunner(BaseRunner):
    _load_in_8bit = True
    summarizer = None
    model = None
    tokenizer = None
    model_name = None
    conversation = None

    def move_to_cpu(self):
        do_gc = False
        if self.model is not None:
            del self.model
            do_gc = True
        if self.tokenizer is not None:
            del self.tokenizer
            do_gc = True
        if do_gc:
            torch.cuda.empty_cache()
            gc.collect()

    def generate_response(self, user_query, user_id):
        if self.model is None or self.tokenizer is None:
            self.load_model(self.model_name)

        conversation_text = ""

        # Encode the conversation history for the user
        if user_id in self.conversation_history:
            conversation_text = self.conversation_history[user_id]['text']
            conversation_vector = np.mean(self.model(self.tokenizer(conversation_text, return_tensors='pt'))[0].detach().numpy(),
                                          axis=1)
        else:
            conversation_vector = np.zeros(self.model.config.n_embd)

        # Encode the user's query
        query_vector = self.tokenizer.encode(user_query, return_tensors='pt')

        # Concatenate the vectors
        combined_vector = torch.cat([conversation_vector, query_vector], dim=-1)

        # Generate a response from the LLM model
        input_ids = self.tokenizer.encode(user_query, return_tensors='pt')
        output = self.model.generate(
            input_ids=input_ids,
            max_length=50,
            pad_token_id=self.tokenizer.eos_token_id,
            context=combined_vector
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)


        # Store the response and update the conversation history
        self.conversation_history[user_id] = {'text': conversation_text + user_query + response, 'vector': combined_vector}
        return response

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value):
        self._load_in_8bit = value

    @property
    def use_gpu(self):
        return torch.cuda.is_available()

    models = MODELS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = kwargs.get("app")
        self.seed = random.randint(0, 100000)
        self.device_map = kwargs.get("device_map", "auto")
        self.load_in_8bit = kwargs.get("load_in_8bit", self._load_in_8bit)
        self.current_model = kwargs.get("model", "flan-t5-xl")
        self.model_name = MODELS[self.current_model]["path"]
        self.model_class = MODELS[self.current_model]["class"]
        self.tokenizer_class = MODELS[self.current_model]["tokenizer"]
        self.is_model_loading = False
        # self.app.message_signal.emit("initialized")

    def do_set_seed(self, seed=None):
        from transformers import set_seed as _set_seed
        seed = self.seed if seed is None else seed
        self.seed = seed
        _set_seed(self.seed)
        # set model and token seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if self.tokenizer:
            self.tokenizer.seed = self.seed
        if self.model:
            self.model.seed = self.seed

    def clear_gpu_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self, model_name, pipeline_type=None, offline=True):
        self.is_model_loading = True
        if self.model_name == model_name and self.model and self.tokenizer:
            return
        local_files_only = offline
        from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, GPTNeoForCausalLM, GPT2Tokenizer, \
            AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
        class_names = {
            "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
            "GPTNeoXTokenizerFast": GPTNeoXTokenizerFast,
            "GPTNeoForCausalLM": GPTNeoForCausalLM,
            "GPT2Tokenizer": GPT2Tokenizer,
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "T5ForConditionalGeneration": T5ForConditionalGeneration,
        }
        from transformers import pipeline
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        self.clear_gpu_cache()
        try:
            tokenizer_class = class_names[self.tokenizer_class]
            if pipeline_type == "summarize":
                self.model = pipeline(
                    "summarization",
                    model=model_name,
                    device=0
                )
            else:
                # iterate over all models and find path matching model_name then set model_class
                for model in MODELS:
                    if model_name == MODELS[model]["path"]:
                        self.model_class = MODELS[model]["class"]
                model_class = class_names[self.model_class]
                self.model = model_class.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.load_in_8bit or self.use_gpu else torch.float32,
                    local_files_only=local_files_only,
                    device_map=self.device_map,
                    load_in_8bit=self.load_in_8bit,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                self.model.eval()
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except torch.cuda.OutOfMemoryError:
            print("Out of memory")
            self.load_model(model_name)
        except OSError as e:
            print(e)
            if offline:
                return self.load_model(model_name, pipeline_type, offline=False)
        self.is_model_loading = False

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt, **properties):
        if "data" in properties:
            skip_special_tokens = properties["data"].get("skip_special_tokens", True)
        else:
            skip_special_tokens = properties.get("skip_special_tokens", True)
        model = properties.get("model", self.current_model)
        model_name = MODELS[model]["path"]
        while self.is_model_loading:
            time.sleep(0.1)
        if not self.model or self.model_name != model_name:
            self.load_model(model_name)
        if not self.tokenizer:
            print("failed to load model")
            return
        seed = properties.get("seed")
        self.do_set_seed(seed)

        properties = self.parse_properties(properties)
        response = [""]
        outputs = None
        self.do_set_seed(properties.get("seed"))
        # try:
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        try:
            outputs = self.model.generate(inputs, **properties)
        except RuntimeError:
            self.error_handler("Something went wrong during generation")
        except Exception as e:
            if "PYTORCH_CUDA_ALLOC_CONF" in str(e):
                self.error_handler("CUDA out of memory. Try to adjust your settings or using a smaller model.")
                return
            raise e
        if outputs is not None:
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)[0]
            return response
        return response

    def parse_properties(self, properties: dict):
        return {
            "max_length": properties.get("max_length", 20),
            "min_length": properties.get("min_length", 0),
            "do_sample": properties.get("do_sample", True),
            "early_stopping": properties.get("early_stopping", True),
            "num_beams": properties.get("num_beams", 1),
            "temperature": properties.get("temperature", 1.0),
            "top_k": properties.get("top_k", 1),
            "top_p": properties.get("top_p", 0.9),
            "repetition_penalty": properties.get("repetition_penalty", 50.0),
            "bad_words_ids": properties.get("bad_words_ids", None),
            "bos_token_id": properties.get("bos_token_id", None),
            "pad_token_id": properties.get("pad_token_id", None),
            "eos_token_id": properties.get("eos_token_id", None),
            "length_penalty": properties.get("length_penalty", 1.0),
            "no_repeat_ngram_size": properties.get("no_repeat_ngram_size", 1),
            "num_return_sequences": properties.get("num_return_sequences", 1),
            "attention_mask": properties.get("attention_mask", None),
            "decoder_start_token_id": properties.get("decoder_start_token_id", None),
            "use_cache": properties.get("use_cache", None),
        }
