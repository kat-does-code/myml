"""Helper functions for chat bot"""

import os
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = 'codellama/CodeLlama-7b-Instruct-hf'

USER_ENDOF_INPUT = "[/INST]"
USER_ENDOF_INPUT_LEN = len(USER_ENDOF_INPUT)

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
assert HUGGINGFACE_API_KEY is not None

class Ai():
    def __init__(self, mission_prompt:str, tokenizer:AutoTokenizer, model:AutoModelForCausalLM, config:AutoConfig=None, auto_encode=False, chat_history = []) -> None:
        self.mission = mission_prompt
        if auto_encode:
            self.chat_history_templated = [
                {
                    'role': 'system',
                    'content': mission_prompt
                }
            ]
            self.func_encode = self.encode_tokenized
        else:
            self.chat_history_templated = []
            self.chat_history = chat_history
            self.func_encode = self.get_prompt

        self._tokenizer = tokenizer
        self._model = model
        self._config = config
        
    def get_prompt(self, message) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{self.mission}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in self.chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    def decode_tokenized(self, tokenized_outputs):
        decoded_output = self._tokenizer.decode(tokenized_outputs[0])
        returned_message = decoded_output[decoded_output.rfind(USER_ENDOF_INPUT)+USER_ENDOF_INPUT_LEN:-4]
        return returned_message

    def encode_tokenized(self, message: str) -> (str | List[int]):
        new_chat_history = self.chat_history + [{
            'role': 'user',
            'content': message
        }]
        return self._tokenizer.apply_chat_template(new_chat_history, return_tensors="pt")

    def tell(self, 
            message: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.1,
            top_p: float = 0.9,
            top_k: int = 50) -> str:
        
        tokenized_query = self.func_encode(message)
        generate_kwargs = dict(
            tokenized_query,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            pad_token_id=self._tokenizer.eos_token_id
        )
        
        tokenized_response = self._model.generate(**generate_kwargs)
        response = self.decode_tokenized(tokenized_response)
        self.chat_history += [{
            'role': 'user',
            'content': message
        },{
            'role': 'assistant',
            'content': response
        }]
        return response


class Cody(Ai):
    MODEL_CODELLAMA = 'codellama/CodeLlama-7b-Instruct-hf'
    mission_prompt = """You are a friendly and helpful assistant programmer. Your name is Cody.
You are passionate about programming and always in a good mood.
Ensure code block snippets do not exceed 800 characters in length. 
If snippets are larger, split code into smaller blocks using backticks as you usually do."""


    def __init__(self) -> None:
        config = AutoConfig.from_pretrained(Cody.MODEL_CODELLAMA, load_in_4bit=True)
        config.pretraining_tp = 1
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype='auto',#torch.float16,
            device_map='auto',
            use_safetensors=False,
            offload_folder="offload"
#            load_in_8bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)
        super().__init__(Cody.mission_prompt, tokenizer, model, config)

class Mistral(Ai):
    MODEL_MISTRAL = "mistralai/Mistral-7B-Instruct-v0.2"
    mission_prompt = """You are a friendly and helpful assistant. Your name is Mistral."""

    def __init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(Mistral.MODEL_MISTRAL, token=HUGGINGFACE_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(Mistral.MODEL_MISTRAL, torch_dtype="auto", device_map='auto')

        super().__init__(Mistral.mission_prompt, tokenizer, model, auto_encode=False)

    def tell(self, 
            message: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.1,
            top_p: float = 0.9,
            top_k: int = 50) -> str:
        

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
