import os

from datetime import datetime

from sty import bg, fg
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import dotenv

dotenv.load_dotenv()

MAX_NEW_TOKENS = 1000
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
assert HUGGINGFACE_API_KEY is not None

MODEL_GEMMA = "google/gemma-7b"
MODEL_MISTRAL = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_GPT2 = "gpt2"

CHAT_DIRECTION_OUTGOING = True
CHAT_DIRECTION_INCOMING = False

model_used = MODEL_MISTRAL

# tokenizer = AutoTokenizer.from_pretrained(MODEL_GEMMA, token=HUGGINGFACE_API_KEY)
# model = AutoModelForCausalLM.from_pretrained(MODEL_GEMMA, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2", token=HUGGINGFACE_API_KEY)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_used, token=HUGGINGFACE_API_KEY)
model = AutoModelForCausalLM.from_pretrained(model_used, torch_dtype="auto").to(device)

USER_ENDOF_INPUT = "[/INST]"
USER_ENDOF_INPUT_LEN = len(USER_ENDOF_INPUT)
CLS_CMD = 'cls' if os.name == 'nt' else 'clear'

USER_COLOUR = 201
ASSISTANT_COLOUR = 43
OTHER_COLOUR = 11
BACKGROUND_COLOUR = 0

def tt():
    return datetime.now().strftime("%H:%M:%S")

def append_chat(log, is_user_message, message):
    if is_user_message:
        role_fmtd = fg(USER_COLOUR) + 'YOU' + fg.rs
        msg_fmtd = fg(USER_COLOUR) + message + fg.rs
        msg_dir = fg(OTHER_COLOUR) + ">>" + fg.rs
    else:
        role_fmtd =  fg(ASSISTANT_COLOUR) + 'ASSISTANT' + fg.rs
        msg_fmtd = fg(ASSISTANT_COLOUR) + message + fg.rs
        msg_dir = fg(OTHER_COLOUR) + "<<" + fg.rs

    tt_fmtd = fg(OTHER_COLOUR) + tt() + fg.rs
    msg_fmtd = f"[{tt_fmtd}] {role_fmtd} {msg_dir} {msg_fmtd}\n"
    log += msg_fmtd
    os.system(CLS_CMD)
    print(bg(BACKGROUND_COLOUR) + log + bg.rs)

    chat_template_msg = {'role': 'user' if is_user_message else 'assistant', 'content': message }
    return log, chat_template_msg

def send_msg(tokenized_chat):
    outputs = model.generate(tokenized_chat, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
            
    decoded_output = tokenizer.decode(outputs[0])
    returned_message = decoded_output[decoded_output.rfind(USER_ENDOF_INPUT)+USER_ENDOF_INPUT_LEN:-4]
    return returned_message

try:
    chat_history = []
    chat_log = ""
    input_text = ''
    while input_text != 'exit':
        if input_text != '':
            chat_log, new_tokenizable = append_chat(chat_log, CHAT_DIRECTION_OUTGOING, input_text)
            chat_history.append(new_tokenizable)

            tokenized_chat = tokenizer.apply_chat_template(chat_history, return_tensors="pt").to(device)
            response = send_msg(tokenized_chat)
            
            chat_log, new_tokenizable = append_chat(chat_log, CHAT_DIRECTION_INCOMING, response)
            chat_history.append(new_tokenizable)
        input_text = input('> ')
except EOFError:
    pass
