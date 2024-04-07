"""Helper functions for chat bot"""

import enum
import os
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline

USER_ENDOF_INPUT = "[/INST]"
USER_ENDOF_INPUT_LEN = len(USER_ENDOF_INPUT)

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
assert HUGGINGFACE_API_KEY is not None

class AiResponse(enum.Enum):
    TEXT = 0
    IMAGE = 1

class Ai():
    MAX_NEW_TOKENS=1024

    def __init__(self, mission_prompt:str=None, tokenizer:AutoTokenizer=None, model:AutoModelForCausalLM=None, config:AutoConfig=None, chat_history = []) -> None:
        self.mission = mission_prompt
        self.chat_history = chat_history
        self._tokenizer = tokenizer
        self._model = model
        self._config = config
        

    def decode_tokenized(self, tokenized_outputs):
        decoded_output = self._tokenizer.decode(tokenized_outputs[0])
        returned_message = decoded_output[decoded_output.rfind(USER_ENDOF_INPUT)+USER_ENDOF_INPUT_LEN:-4]
        return returned_message

    def tell(self, message:str, **kwargs):
        raise NotImplementedError


class Mai(Ai):
    MODEL_CODELLAMA = 'codellama/CodeLlama-7b-Instruct-hf'
    mission_prompt = """You are a friendly and helpful assistant programmer. Your name is Mai.
You are passionate about programming and always in a good mood.
Ensure code block snippets do not exceed 800 characters in length. 
If snippets are larger, split code into smaller blocks using backticks as you usually do."""

    def __init__(self) -> None:
        config = AutoConfig.from_pretrained(Mai.MODEL_CODELLAMA, load_in_4bit=True)
        config.pretraining_tp = 1
        model = AutoModelForCausalLM.from_pretrained(
            Mai.MODEL_CODELLAMA,
            config=config,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map='auto',
            use_safetensors=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(Mai.MODEL_CODELLAMA, token=HUGGINGFACE_API_KEY)
        super().__init__(Mai.mission_prompt, tokenizer, model, config)

    def encode(self, message) -> List[int]:
        texts = [f'<s>[INST] <<SYS>>\n{self.mission}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in self.chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        tokenized = self._tokenizer([''.join(texts)], return_tensors='pt', add_special_tokens=False).to('cuda')
        return tokenized

    def tell(self,
            message:str,
            temperature: float = 0.1,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs) -> str:
        
        prompt = self.encode(message)
        generate_kwargs = dict(
            prompt,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        
        response = self._model.generate(**generate_kwargs)
        answer = self.decode_tokenized(response)
        self.chat_history.append(
            (message, answer)
        )
        return answer

class Mistral(Ai):
    MODEL_MISTRAL = "mistralai/Mistral-7B-Instruct-v0.2"
    mission_prompt = """You are a friendly and helpful assistant. Your name is Mistral."""

    def __init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(Mistral.MODEL_MISTRAL, token=HUGGINGFACE_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(Mistral.MODEL_MISTRAL, torch_dtype="auto").to("cuda")
        super().__init__(Mistral.mission_prompt, tokenizer, model)

    def tell(self, message:str, **kwargs):
        chat_template_msg = {'role': 'user', 'content': message }
        self.chat_history.append(chat_template_msg)

        tokenized_chat = self._tokenizer.apply_chat_template(self.chat_history, return_tensors="pt").to("cuda")

        outputs = self._model.generate(tokenized_chat, max_new_tokens=Ai.MAX_NEW_TOKENS, pad_token_id=self._tokenizer.eos_token_id)
        answer = self.decode_tokenized(outputs)
        chat_template_answer = {'role': 'assistant', 'content': answer }
        self.chat_history.append(chat_template_answer)
        return answer

class Flan(Ai):
    MODEL_FLAN = "google/flan-t5-base"

    def __init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

        super().__init__("", tokenizer, model)
            
    def tell(self, message:str, **kwargs):
        input_ids = self._tokenizer(message, return_tensors="pt").input_ids.to("cuda")

        outputs = self._model.generate(input_ids, max_new_tokens=Ai.MAX_NEW_TOKENS)
        return self.decode_tokenized(outputs)

class Diva(Ai):
    MODEL_STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"

    def __init__(self) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(Diva.MODEL_STABLE_DIFFUSION, torch_dtype=torch.float16).to("cuda")
        super().__init__()
        
    def tell(self, message:str, **kwargs):
        image = self.pipe(message).images[0]
        image.save("temp.png")
        return 


# class Mai():
#     SYS_PROMPT = """You are a friendly and helpful assistant programmer. Your name is Mai.
# Ensure code block snippets do not exceed 800 characters in length. If snippets are larger, split code into smaller blocks using backticks as you usually do."""

#     def __init__(self) -> None:
#         self.chat_history = []
#         self._mission = Mai.SYS_PROMPT
        
#     def tell(self, message) -> str:
#         response_tokenized = codellama.run(message, self.chat_history, self._mission)
#         response = codellama.get_last_message(response_tokenized).strip()
#         self.chat_history.append(
#             (message, response)
#         )
#         return response
